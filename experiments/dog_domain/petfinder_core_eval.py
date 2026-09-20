"""Fixed .20 text-weight core rerun; no test-set tuning or retraining."""
import io
import json
import hashlib
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from PIL import Image
from experiments.dog_domain.petfinder_train import OUT as TRAIN, ARCHIVE, REV
from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows
from experiments.dino_fusion.alignment import projection_head_from_checkpoint
from experiments.composed_retrieval.metrics import cluster_interval
from experiments.dog_domain.notice_extension import OUT as KR
from experiments.dog_domain.taiwan_eval import OUT as TW, attributes as tw_attributes

OUT = TRAIN / 'core_evaluation'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def petfinder():
    records = [r for r in read(TRAIN/'records.json') if r['split']=='test']
    cache = OUT/'petfinder_features.npz'
    if not cache.exists():
        clip, dino = ClipEncoder(device='cuda'), DinoEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m', device='cuda', revision=REV, local_files_only=True)
        c, d = [], []
        with ZipFile(ARCHIVE) as z:
            for start in range(0,len(records),64):
                images = [Image.open(io.BytesIO(z.read(r['image']))).convert('RGB') for r in records[start:start+64]]
                with torch.inference_mode():
                    c.append(normalize_rows(clip._model.encode_image(torch.stack([clip._preprocess(im) for im in images]).cuda()).float().cpu().numpy()))
                d.append(dino.encode_batch(images))
        prompts = [f"Find a {r['size']} dog whose coat is {' and '.join(r['colors'])}." for r in records]
        t = np.concatenate([clip.encode_text_batch(prompts[i:i+64]) for i in range(0,len(prompts),64)])
        np.savez_compressed(cache, clip=np.concatenate(c), dino=np.concatenate(d), text=t, ids=np.array([r['pet_id'] for r in records]))
        del clip,dino
        torch.cuda.empty_cache()
    f = np.load(cache)
    assert list(f['ids'])==[r['pet_id'] for r in records]
    return dict(cq=f['clip'],cg=f['clip'],dq=f['dino'],dg=f['dino'],text=f['text'],
                attrs=[(r['colors'],r['size']) for r in records], groups=[r['rescuer'] for r in records],
                ids=f['ids'], exact=False, identity=False)


def external():
    r, f, m = read(KR/'records.json'), np.load(KR/'features.npz'), read(KR/'features.json')
    assert m['records_sha256']==digest(KR/'records.json') and m['features_sha256']==digest(KR/'features.npz')
    assert list(f['notice_ids'])==[x['notice_id'] for x in r]
    korea = dict(cq=f['clip_query'],cg=f['clip_gallery'],dq=f['dino_query'],dg=f['dino_gallery'],text=f['text_english'],
                 attrs=[(x['attributes']['colors'],x['attributes']['size']) for x in r], groups=[x['group'] for x in r], ids=f['notice_ids'], exact=False, identity=True)
    r, f, m = read(TW/'records.json'), np.load(TW/'images.npz'), read(TW/'images.json')
    assert m['records_sha256']==digest(TW/'records.json') and m['features_sha256']==digest(TW/'images.npz')
    assert list(f['animal_ids'])==[x['animal_id'] for x in r]
    text = np.load(TW/'text_features.npz')['template_english']
    taiwan = dict(cq=f['clip'],cg=f['clip'],dq=f['dino'],dg=f['dino'],text=text,
                  attrs=[tw_attributes(x) for x in r],groups=[x['shelter_id'] for x in r],ids=f['animal_ids'],exact=True,identity=False)
    return {'Korea':korea,'Taiwan':taiwan}


def evaluate(name, data, runs):
    n=len(data['ids'])
    attrs=data['attrs']
    rel=np.array([[a[1]==b[1] and (set(a[0])==set(b[0]) if data['exact'] else bool(set(a[0])&set(b[0]))) for b in attrs] for a in attrs])
    np.fill_diagonal(rel,False)
    known=rel.sum(1)>0
    groups=np.array(data['groups'])[known]
    relevance=torch.tensor(rel,device='cuda')
    discount=1/torch.log2(torch.arange(2,12,device='cuda',dtype=torch.float32))
    ideal=torch.tensor([float(discount[:min(10,int(k))].sum()) for k in rel.sum(1)],device='cuda')
    tensors={k:torch.tensor(data[k],device='cuda') for k in ['cq','cg','dq','dg','text']}
    metrics, identity={},{}

    def score(method, q=None, g=None, late=False):
        values=[]
        identities=[]
        if not late:
            q,g=q.cpu().numpy(),g.cpu().numpy()
        for start in range(0,n,256):
            end=min(n,start+256)
            if late:
                s=.8*(data['dq'][start:end]@data['dg'].T)+.2*(data['text'][start:end]@data['cg'].T)
            else:
                s=q[start:end]@g.T
            s=torch.tensor(s,device='cuda')
            rows=torch.arange(end-start,device='cuda')
            cols=torch.arange(start,end,device='cuda')
            if data['identity']:
                order=torch.argsort(s,descending=True,stable=True)
                rank=(order==cols[:,None]).int().argmax(1)+1
                identities.append(torch.stack([(rank<=1).float(),(rank<=5).float(),(rank<=10).float(),1/rank.float()],dim=1).cpu().numpy())
            s[rows,cols]=-torch.inf
            top=torch.argsort(s,descending=True,stable=True)[:,:10]
            matches=relevance[start:end].gather(1,top).float()
            ndcg=(matches*discount).sum(1)/ideal[start:end].clamp_min(1e-8)
            values.append(torch.stack([ndcg,matches.mean(1)],dim=1).cpu().numpy())
        metrics[method]=np.concatenate(values)
        if identities:
            identity[method]=np.concatenate(identities)

    norm=lambda x:torch.tensor(normalize_rows(x.cpu().numpy()),device='cuda')
    score('CLIP_image',tensors['cq'],tensors['cg'])
    score('DINO_image',tensors['dq'],tensors['dg'])
    # Concatenation realizes an equal-weight sum of the two cosine scores.
    score('CLIP_DINO_image',torch.cat([tensors['cq'],tensors['dq']],1)/2**.5,torch.cat([tensors['cg'],tensors['dg']],1)/2**.5)
    score('CLIP_mix',norm(.8*tensors['cq']+.2*tensors['text']),tensors['cg'])
    score('CLIP_text_only',tensors['text'],tensors['cg'])
    score('late_mix',late=True)
    for run in runs:
        ck=torch.load(run['checkpoint'],map_location='cpu',weights_only=True)
        head=projection_head_from_checkpoint(ck).cuda().eval()
        head.load_state_dict(ck['state_dict'])
        with torch.inference_mode():
            mapped=norm(head(tensors['text']))
        prefix=f"{run['target']}_{run['architecture']}_seed{run['seed']}"
        q,g=(tensors['cq'],tensors['cg']) if run['target']=='clip' else (tensors['dq'],tensors['dg'])
        score(prefix+'_mix',norm(.8*q+.2*mapped),g)
        score(prefix+'_text_only',mapped,g)
    summaries={}
    for target,arch in [('clip','linear'),('dino','linear'),('dino','mlp'),('dino','flow')]:
        for mode in ['mix','text_only']:
            key=f'{target}_{arch}_{mode}'
            entries=[metrics[f'{target}_{arch}_seed{s}_{mode}'] for s in [17,29,43]]
            metrics[key]=np.mean(entries,axis=0)
            summaries[key]={'seed_means': [v[known].mean(0).tolist() for v in entries]}
            if identity:
                identity[key]=np.mean([identity[f'{target}_{arch}_seed{s}_{mode}'] for s in [17,29,43]],axis=0)
    main=[k for k in metrics if '_seed' not in k]
    for k in main:
        summaries.setdefault(k,{})['semantic']=cluster_interval(metrics[k][known],groups)
        if identity:
            summaries[k]['identity']=cluster_interval(identity[k],np.array(data['groups']))
    paired={}
    for k in ['dino_linear_mix','dino_mlp_mix','dino_flow_mix']:
        for baseline in ['CLIP_mix','DINO_image','late_mix','clip_linear_mix']:
            paired[f'{k} minus {baseline}']=cluster_interval((metrics[k]-metrics[baseline])[known],groups)
    result=dict(dataset=name,gallery=n,evaluable_queries=int(known.sum()),text_weight=.2,
                relevance='exact color set and size' if data['exact'] else 'any shared color and same size',
                results=summaries,paired=paired,metrics=['nDCG@10','P@10'],identity_metrics=['R@1','R@5','R@10','MRR'] if identity else [])
    (OUT/f'{name}_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/f'{name}_per_query.npz',**metrics,known=known,ids=data['ids'],groups=np.array(data['groups']))
    for k in ['CLIP_mix','DINO_image','late_mix','clip_linear_mix','dino_linear_mix','dino_mlp_mix','dino_flow_mix']:
        print(name,k,round(summaries[k]['semantic']['mean'][0]*100,2),flush=True)
    return result


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    training=read(TRAIN/'training_report.json')
    assert training['complete'] and read(TRAIN/'verification.json')['passed']
    results=[]
    datasets={'PetFinder':petfinder(),**external()}
    with torch.inference_mode():
        for name,data in datasets.items():
            results.append(evaluate(name,data,training['runs']))
    lines=['# PetFinder 학습 모델 핵심 재평가','',
           '영어 조건, 글 비중 0.20 고정. 학습 모델은 시드 17/29/43의 질의별 점수 평균. 값은 nDCG@10 × 100이며 정확도가 아니다.',
           '한국은 기존 서로 다른 사진 쌍, 대만·PetFinder는 대표 사진을 입력하되 자기 공고는 후보에서 제외한다.',
           '기존 한국/대만 채점 규칙과 프롬프트를 유지한다. 대만은 색상 집합 완전 일치, 한국/PetFinder는 색상 하나 이상 겹침과 크기 일치이다. 국가 간 절대 점수 비교는 부적절하다.','',
           '| 방법 | PetFinder | 한국 | 대만 |','|---|---:|---:|---:|']
    for method in results[0]['results']:
        lines.append('| '+method+' | '+' | '.join(f"{r['results'][method]['semantic']['mean'][0]*100:.2f}" for r in results)+' |')
    lines += ['', '95% 구간은 등록자/보호소 단위 paired bootstrap 2,000회이며 다중 비교 보정 전 탐색적 구간이다. 시드 평균에 대한 구간으로 시드 변동 전체를 포괄하지 않는다.',
              '이 평가는 색상·크기 조건의 충족도를 측정한다. 얼굴 형태 보존, 주관적 취향, 자유로운 설명문 이해를 직접 검증하지 않는다.',
              '추가 조건 변경 실험·한국어 보조 실험·ResNet 기준선·새 수집 시점 확증 평가는 이번 핵심 재평가에 포함하지 않았다. 기존 논문 수치는 자동 교체하지 않았다.','']
    Path(__file__).with_name('PETFINDER_CORE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'provenance.json').write_text(json.dumps(dict(training_report_sha256=digest(TRAIN/'training_report.json'),
        evaluator_sha256=digest(Path(__file__)),checkpoints={Path(r['checkpoint']).name:digest(Path(r['checkpoint'])) for r in training['runs']}),indent=2),encoding='utf-8')
    print('CORE EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
