"""Extend the frozen human pool with DINO image-only; never alter v1 ratings."""
import hashlib,json,secrets,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from experiments.dino_fusion.core import DinoEncoder
from experiments.dog_domain.notice_extension import OUT as KR
from experiments.dog_domain.petfinder_train import REV

ROOT=Path('D:/meongtamjeong_research')
SOURCE=ROOT/'human_visual_study_v1'
OUT=ROOT/'human_visual_dino_supplement_v1'

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'study.json').exists():
        print((OUT/'overlap.json').read_text(encoding='utf-8'));return
    torch.set_num_threads(4)
    study=json.loads((SOURCE/'study.json').read_text(encoding='utf-8'))
    gallery=json.loads((KR/'records.json').read_text(encoding='utf-8'))
    features=np.load(KR/'features.npz')
    ids=[r['notice_id'] for r in gallery]
    assert list(features['notice_ids'])==ids
    encoder=DinoEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m',device='cuda',revision=REV,local_files_only=True)
    images=[Image.open(SOURCE/'source_images'/q['source_file']).convert('RGB') for q in study['queries']]
    d=encoder.encode_batch(images)
    scores=d@features['dino_gallery'].T
    ranks=np.argsort(-scores,axis=1,kind='stable')[:,:5]
    np.savez_compressed(OUT/'dino_image_scores.npz',reference_dino=d,scores=scores,notice_ids=features['notice_ids'])
    merged=json.loads(json.dumps(study));existing={p['id'] for p in study['pairs']};new=[];overlap=[]
    attrs={r['notice_id']:r['attributes'] for r in gallery}
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    for qi,q in enumerate(merged['queries']):
        found=[ids[j] for j in ranks[qi]];q['rankings']['dino_image_only']=found;new_count=0
        shutil.copy2(SOURCE/'assets'/q['reference'],assets/q['reference'])
        for cid in found:
            pid=hashlib.sha256((q['id']+':'+cid).encode()).hexdigest()[:24]
            if pid in existing:continue
            image=hashlib.sha256(('candidate:'+cid).encode()).hexdigest()[:24]+'.png'
            attr=attrs[cid]
            pair=dict(id=pid,query_id=q['id'],candidate_id=cid,reference=q['reference'],image=image,attribute_match=int(q['color'] in attr['colors'] and q['size']==attr['size']))
            new.append(pair);merged['pairs'].append(pair);merged['assets'][image]=image
            shutil.copy2(KR/'images'/cid/'gallery.png',assets/image);new_count+=1
        overlap.append(dict(query=q['id'],already_rated=5-new_count,new_pairs=new_count))
    summary=dict(source_study_sha256=hashlib.sha256((SOURCE/'study.json').read_bytes()).hexdigest(),dino_revision=REV,top5_total=150,reused_pairs=150-len(new),new_pairs=len(new),new_ratings_required=len(new)*5,expanded_pool_pairs=len(merged['pairs']),per_query=overlap)
    merged['protocol'].update(raters=5,pool='union of four top-5 lists per query',gain='linear relevance; mean across five raters',zero_idcg='report positive-ideal-only primary and original zero-filled sensitivity',status='awaiting supplemental judgments')
    merged['source_study_sha256']=summary['source_study_sha256']
    write(OUT/'merged_study.json',merged)
    supplement=json.loads(json.dumps(merged));supplement['pairs']=new
    active={p['query_id'] for p in new};supplement['queries']=[q for q in merged['queries'] if q['id'] in active]
    supplement['assets']={name:name for name in {p[k] for p in new for k in ['reference','image']}}
    supplement['protocol'].update(kind='supplement',queries=len(active),public_raters=['R1','R2','R3','R4','R5'],pool='new, previously unjudged query-candidate pairs only',status='awaiting supplemental judgments')
    write(OUT/'study.json',supplement)
    write(OUT/'access.json',dict(admin=secrets.token_urlsafe(32),raters={f'R{i}':secrets.token_urlsafe(24) for i in range(1,6)}))
    write(OUT/'overlap.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
