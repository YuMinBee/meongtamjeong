"""Freeze a 30-query pooled, blinded study and run validation-only sensitivity."""
import hashlib
import json
import random
import secrets
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import torch
from PIL import Image
from experiments.dog_domain.petfinder_expanded import BASE, OUT as TRAIN, read
from experiments.dog_domain.petfinder_expanded_eval import map_text, metrics, behavior
from experiments.dog_domain import petfinder_expanded_eval as expanded
from experiments.dog_domain.notice_extension import OUT as KR, pixel_hash
from experiments.dog_domain.external_reference_cases import CASES
from experiments.dog_domain.petfinder_train import REV
from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows

OUT=Path('D:/meongtamjeong_research/human_visual_study_v1')
SEED=20260918

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def get(url):
    response=requests.get(url,timeout=90);response.raise_for_status();return response

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'study.json').exists():
        print('Study already frozen; refusing to change live evaluation items.');return
    selection=OUT/'selection.json'
    if not selection.exists():
        tree=get('https://api.github.com/repos/ml4py/dataset-iiit-pet/git/trees/master?recursive=1').json()
        assert not tree.get('truncated')
        rng=random.Random(SEED);selected=[]
        for breed,korean,color,size in CASES:
            files=sorted(x['path'] for x in tree['tree'] if x['path'].startswith('images/'+breed+'_') and x['path'].endswith('.jpg') and Path(x['path']).name!=breed+'_1.jpg')
            for path in rng.sample(files,5):selected.append(dict(breed=breed,breed_ko=korean,color=color,size=size,path=path))
        write(selection,dict(seed=SEED,mirror_commit=tree['sha'],selection=selected,excluded='Previously inspected index-1 images; no selection based on retrieval outcomes.'))
    selected=read(selection);downloads=OUT/'source_images';downloads.mkdir(exist_ok=True)
    def download(item):
        path=downloads/Path(item['path']).name
        if not path.exists():path.write_bytes(get(f"https://raw.githubusercontent.com/ml4py/dataset-iiit-pet/{selected['mirror_commit']}/{item['path']}").content)
        with Image.open(path) as im:im.verify()
        return path
    with ThreadPoolExecutor(max_workers=5) as pool:paths=list(pool.map(download,selected['selection']))
    images=[Image.open(p).convert('RGB') for p in paths]
    print('Downloaded all 30 preselected references.',flush=True)
    encoder=ClipEncoder(device='cuda')
    runs=read(TRAIN/'training_report.json')['runs']
    linear=[r for r in runs if r['target']=='dino' and r['architecture']=='linear']
    assert len(linear)==3
    # Validation data alone: self-excluded validation gallery and queries.
    active=[r for r in read(BASE/'records.json') if r['split']!='test']
    cached=np.load(BASE/'features.npz');assert str(cached['records_sha256'])==hashlib.sha256((BASE/'records.json').read_bytes()).hexdigest()
    ix=np.array([i for i,r in enumerate(active) if r['split']=='validation']);records=[active[i] for i in ix]
    prompts=[f"Find a {r['size']} dog whose coat is {' and '.join(r['colors'])}." for r in records]
    t=np.concatenate([encoder.encode_text_batch(prompts[i:i+128]) for i in range(0,len(prompts),128)])
    rel=np.array([[a['size']==b['size'] and bool(set(a['colors'])&set(b['colors'])) for b in records] for a in records]);np.fill_diagonal(rel,False)
    known=rel.any(1);d=cached['dino'][ix];mapped=[map_text(run,t) for run in linear]
    sensitivity=[]
    for alpha in [1.,.8,.5,.2,0.]:
        values=[metrics(normalize_rows(alpha*d+(1-alpha)*m)@d.T,rel,np.arange(len(d)))[0] for m in mapped]
        sensitivity.append(dict(image_weight=alpha,ndcg10=float(np.mean(values,axis=0)[known,0].mean()),queries=int(known.sum())))
    write(OUT/'weight_sensitivity.json',dict(split='PetFinder validation only',gallery=len(d),results=sensitivity,best_attribute_alpha=max(sensitivity,key=lambda r:r['ndcg10'])['image_weight'],study_alpha=.8,interpretation='Retrospective validation sensitivity; alpha .8 was previously fixed, not selected from this sweep. Human study retains .8 to assess an image-conditioned system.'))
    print('Validation sensitivity:',sensitivity,flush=True)
    # Obtain actual Linear activity pilot values; preserve historical files.
    test_records=[r for r in read(TRAIN/'records.json') if r['split']=='test']
    pf=np.load(BASE/'core_evaluation/petfinder_features.npz');allr=read(TRAIN/'records.json');testix=[i for i,r in enumerate(allr) if r['split']=='test']
    profiles=np.load(TRAIN/'profiles.npz');expanded.EVAL=OUT/'linear_auxiliary'
    # Existing behavior function evaluates selected architectures; configure via optional argument.
    behavior(test_records,pf,{k:profiles[k][testix] for k in ['profile','description']},encoder,runs,read(BASE/'training_report.json')['runs'],architectures=[('clip','linear'),('dino','linear')])
    gallery=read(KR/'records.json');f=np.load(KR/'features.npz');assert list(f['notice_ids'])==[r['notice_id'] for r in gallery]
    hashes={pixel_hash(Image.open(KR/'images'/r['notice_id']/'gallery.png').convert('RGB')) for r in gallery}
    assert not hashes.intersection(pixel_hash(im) for im in images),'Reference duplicates gallery'
    dino=DinoEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m',device='cuda',revision=REV,local_files_only=True)
    with torch.inference_mode():c=normalize_rows(encoder._model.encode_image(torch.stack([encoder._preprocess(im) for im in images]).cuda()).float().cpu().numpy())
    d=dino.encode_batch(images)
    prompts=[f"Find a {'small' if r['size']=='small' else 'medium-sized'} dog whose coat is {r['color']}." for r in selected['selection']]
    t=encoder.encode_text_batch(prompts);mapped=[map_text(run,t) for run in linear]
    scores={'clip_image_text':normalize_rows(.8*c+.2*t)@f['clip_gallery'].T,
        'dino_linear_image_text':np.mean([normalize_rows(.8*d+.2*m)@f['dino_gallery'].T for m in mapped],axis=0),
        'dino_linear_text_only':np.mean([m@f['dino_gallery'].T for m in mapped],axis=0)}
    ranks={k:np.argsort(-v,axis=1,kind='stable')[:,:5] for k,v in scores.items()}
    assets=OUT/'assets';assets.mkdir(exist_ok=True);queries=[];pairs=[];assetmap={}
    for qi,(record,path) in enumerate(zip(selected['selection'],paths)):
        qid=f'q{qi+1:02d}';reference=hashlib.sha256(('reference:'+qid).encode()).hexdigest()[:24]+'.jpg'
        shutil.copy2(path,assets/reference);assetmap[reference]=reference
        rankings={k:[str(f['notice_ids'][j]) for j in v[qi]] for k,v in ranks.items()}
        candidates=sorted(set().union(*[set(v) for v in rankings.values()]))
        queries.append(dict(id=qid,breed=record['breed'],reference=reference,source_file=path.name,query=prompts[qi],color=record['color'],size=record['size'],rankings=rankings))
        for cid in candidates:
            pid=hashlib.sha256((qid+':'+cid).encode()).hexdigest()[:24]
            image_id=hashlib.sha256(('candidate:'+cid).encode()).hexdigest()[:24]+'.png'
            shutil.copy2(KR/'images'/cid/'gallery.png',assets/image_id);assetmap[image_id]=image_id
            attr=next(r['attributes'] for r in gallery if r['notice_id']==cid)
            pairs.append(dict(id=pid,query_id=qid,candidate_id=cid,reference=reference,image=image_id,attribute_match=int(record['color'] in attr['colors'] and record['size']==attr['size'])))
    protocol=dict(schema='dog-visual-study.v1',seed=SEED,queries=30,raters=3,topk=5,alpha=.8,ranking='mean scores over seeds 17/29/43',pool='union of all three top-5 lists per query; deduplicate within query',visual_scale=[0,1,2,3],gain='linear relevance; mean across three raters',joint='attribute_match times mean visual grade',idcg='top 5 ideal grades from the common judged pool; pooled nDCG, not exhaustive-gallery nDCG',zero_idcg='score 0 and separately report zero-ideal queries',missing='require all three raters for every pooled candidate of a query; never replace missing with zero',color_in_visual='ignore coat color as a primary criterion; prioritize face/body/coat structure; do not guess breed or real weight',status='awaiting human ratings')
    study=dict(protocol=protocol,queries=queries,pairs=pairs,assets=assetmap,selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),checkpoints={Path(r['checkpoint']).name:hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest() for r in linear})
    write(OUT/'study.json',study)
    write(OUT/'access.json',dict(admin=secrets.token_urlsafe(32),raters={f'R{i}':secrets.token_urlsafe(24) for i in range(1,4)}))
    write(OUT/'protocol.json',protocol)
    print(f"Frozen 30 queries / {len(pairs)} unique pairs per rater / {len(pairs)*3} total ratings. No human results yet.",flush=True)

if __name__=='__main__':main()
