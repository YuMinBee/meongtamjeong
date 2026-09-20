"""Predefined external dog photos against the Korean notice gallery."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from experiments.dog_domain.petfinder_expanded import OUT as TRAIN,read
from experiments.dog_domain.petfinder_expanded_eval import map_text
from experiments.dog_domain.notice_extension import OUT as KR,pixel_hash
from experiments.dino_fusion.core import ClipEncoder,DinoEncoder,normalize_rows
from experiments.dog_domain.petfinder_train import REV

OUT=Path('D:/meongtamjeong_research/external_dog_queries')
CASES=[('samoyed','사모예드','white','small'),('pomeranian','포메라니안','brown','small'),
       ('pug','퍼그','brown','small'),('beagle','비글','brown','medium'),
       ('shiba_inu','시바견','brown','medium'),('wheaten_terrier','휘튼 테리어','cream','small')]
COLOR={'white':'흰색','black':'검정','brown':'갈색','tan':'황갈색','cream':'크림색','yellow':'노랑','gray':'회색','gold':'금색','spotted':'반점'}
SIZE={'tiny':'3kg 이하','small':'3kg 초과~8kg','medium':'8kg 초과~18kg','large':'18kg 초과'}
KFONT=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family']=KFONT.get_name()
plt.rcParams['axes.unicode_minus']=False


def photo(ax,image):
    # Fixed square viewport, full photo visible; no content-dependent crop.
    w,h=image.size;m=max(w,h)
    ax.imshow(image,extent=(.5-w/m/2,.5+w/m/2,.5-h/m/2,.5+h/m/2))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_aspect('equal')
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_facecolor('#f5f5f5')
    for spine in ax.spines.values():spine.set_edgecolor('#c8c8c8');spine.set_linewidth(.7)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    config=dict(cases=CASES,source='Oxford-IIIT Pet; original filename index 1; visual quality checked before retrieval',
                gallery='fixed Korean 611 notices',text_weight=.2,selection='all six predefined cases shown regardless of results',
                semantics='conditions are hypothetical user requests, not labels inferred from reference photos',
                measurements='metadata color and recorded weight only; no visual preference ground truth')
    (OUT/'protocol.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    records=read(KR/'records.json');f=np.load(KR/'features.npz')
    assert list(f['notice_ids'])==[r['notice_id'] for r in records]
    images=[Image.open(OUT/f'{b}_1.jpg').convert('RGB') for b,*_ in CASES]
    hashes=[pixel_hash(im) for im in images]
    gallery_hashes=set()
    for r in records:
        with Image.open(KR/'images'/r['notice_id']/'gallery.png') as im:gallery_hashes.add(pixel_hash(im))
    assert not set(hashes)&gallery_hashes
    encoder=ClipEncoder(device='cuda')
    dino=DinoEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m',device='cuda',revision=REV,local_files_only=True)
    with torch.inference_mode():c=normalize_rows(encoder._model.encode_image(torch.stack([encoder._preprocess(im) for im in images]).cuda()).float().cpu().numpy())
    d=dino.encode_batch(images)
    texts=[f"Find a {'small' if size=='small' else 'medium-sized'} dog whose coat is {color}." for _,_,color,size in CASES]
    t=encoder.encode_text_batch(texts)
    scores={'CLIP 사진+글':normalize_rows(.8*c+.2*t)@f['clip_gallery'].T}
    arrays=[]
    for run in read(TRAIN/'training_report.json')['runs']:
        if run['target']=='dino' and run['architecture']=='flow':
            mapped=map_text(run,t)
            arrays.append(normalize_rows(.8*d+.2*mapped)@f['dino_gallery'].T)
    scores['CLIP–DINO 결합']=np.mean(arrays,axis=0)
    ranks={k:np.argsort(-v,axis=1,kind='stable')[:,:10] for k,v in scores.items()}
    manifest=[]
    for i,(breed,kname,color,size) in enumerate(CASES):
        fig=plt.figure(figsize=(7.5,4.3),facecolor='white')
        # Separate reference/query header and two aligned result rows.
        ax=fig.add_axes([.025,.80,.13,.175]);photo(ax,images[i])
        fig.text(.18,.935,f'{i+1:02d}  {kname} 참고 사진',fontsize=12,weight='bold')
        fig.text(.18,.872,f'요청 조건  {COLOR[color]} · {SIZE[size]}',fontsize=11)
        fig.text(.18,.825,'입력: 공고 외부 사진  |  검색 대상: 한국 공고 611건',fontsize=8.5,color='#555555')
        fig.text(.18,.787,'조건은 사용자 요청의 예시이며, 입력 사진의 정답 라벨이 아닙니다.',fontsize=7.5,color='#666666')
        rows=[]
        rel=np.array([color in r['attributes']['colors'] and r['attributes']['size']==size for r in records])
        discount=1/np.log2(np.arange(2,12))
        ideal=discount[:min(10,int(rel.sum()))].sum()
        for ri,(method,rank) in enumerate(ranks.items()):
            y=.425-ri*.38
            fig.text(.025,y+.19,method.replace(' 사진+글','\n사진+글').replace(' 결합','\n결합'),fontsize=9.5,weight='bold',va='center')
            entries=[]
            for ci,k in enumerate(rank[i,:3]):
                r=records[k];a=r['attributes'];x=.235+ci*.25
                ax=fig.add_axes([x,y+.09,.22,.235])
                with Image.open(KR/'images'/r['notice_id']/'gallery.png') as im:photo(ax,im.convert('RGB'))
                ax.set_title(f'{ci+1}위',fontsize=8,pad=2)
                cc=bool(color in a['colors']) if a['colors'] else None
                ss=bool(a['size']==size) if a['size'] else None
                status=lambda v:'충족' if v is True else '불충족' if v is False else '정보 없음'
                label='·'.join(COLOR.get(v,v) for v in a['colors'])+' / '+SIZE.get(a['size'],'정보 없음')
                fig.text(x+.11,y+.052,label,fontsize=7.4,ha='center')
                fig.text(x+.11,y+.005,f'색상 {status(cc)}  |  크기 {status(ss)}',fontsize=7.4,ha='center',color='#333333')
                entries.append(dict(notice_id=r['notice_id'],rank=ci+1,colors=a['colors'],size=a['size'],color_satisfied=cc,size_satisfied=ss))
            rows.append(dict(method=method,top3=entries,ndcg10=float((rel[rank[i]]*discount).sum()/ideal),p10=float(rel[rank[i]].mean())))
        fig.savefig(OUT/f'case_{i+1:02d}.png',dpi=260)
        plt.close(fig)
        manifest.append(dict(case=i+1,breed=breed,breed_ko=kname,input_file=f'{breed}_1.jpg',input_pixel_sha256=hashes[i],
            query_english=texts[i],requested_color=color,requested_size=size,relevant_candidates=int(rel.sum()),methods=rows))
    (OUT/'results.json').write_text(json.dumps(dict(protocol=config,cases=manifest,exact_pixel_duplicates_with_gallery=0,
        limitations=['Six illustrative references, not a preference survey or representative benchmark.','Metadata checks do not score resemblance to reference.','Near duplicates and pretraining overlap not excluded.']),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Completed all 6 predefined external-photo cases; no outcome-based selection.',flush=True)


if __name__=='__main__':main()
