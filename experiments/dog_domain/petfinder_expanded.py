"""Train full-notice alignment without overwriting the color/size experiment."""
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import json
import re
import time
import hashlib
import numpy as np
import torch
from experiments.dog_domain.petfinder_train import OUT as BASE, ARCHIVE
from experiments.dino_fusion.core import ClipEncoder, normalize_rows
from experiments.dino_fusion.alignment import make_projection_head, project_embeddings
from experiments.dino_fusion.train_alignment import multipositive_loss, save_checkpoint_atomic

OUT=BASE.parent/'english_v2_full_notice'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def clean_description(s):
    s=re.sub(r'https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]+',' ',s)
    s=re.sub(r'(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)',' ',s)
    return ' '.join(s.split())


def prepare():
    records=read(BASE/'records.json')
    with ZipFile(ARCHIVE) as z:
        breeds={r['BreedID']:r['BreedName'] for r in csv.DictReader(io.StringIO(z.read('BreedLabels.csv').decode('utf-8-sig')))}
        states={r['StateID']:r['StateName'] for r in csv.DictReader(io.StringIO(z.read('StateLabels.csv').decode('utf-8-sig')))}
        raw={r['PetID']:r for split in ['train','test'] for r in csv.DictReader(io.StringIO(z.read(f'{split}/{split}.csv').decode('utf-8-sig')))}
    for r in records:
        source=raw[r['pet_id']]
        attrs={k:source[k] for k in ['Age','Breed1','Breed2','Gender','FurLength','Vaccinated','Dewormed','Sterilized','Health','Fee','State']}
        description=clean_description(source['Description'])
        parts=[f"A {r['size']} dog with {' and '.join(r['colors'])} fur."]
        breed=[breeds[x] for x in [attrs['Breed1'],attrs['Breed2']] if x in breeds]
        if breed: parts.append('Breed: '+', '.join(dict.fromkeys(breed))+'.')
        age=int(attrs['Age'])
        if age>0: parts.append(f'Age: {age} months.')
        for k,label,mapping in [
            ('Gender','Sex',{'1':'male','2':'female'}),('FurLength','Coat length',{'1':'short','2':'medium','3':'long'}),
            ('Vaccinated','Vaccinated',{'1':'yes','2':'no'}),('Dewormed','Dewormed',{'1':'yes','2':'no'}),
            ('Sterilized','Neutered or spayed',{'1':'yes','2':'no'}),
            ('Health','Reported health',{'1':'healthy','2':'minor injury','3':'serious injury'})]:
            if attrs[k] in mapping: parts.append(label+': '+mapping[attrs[k]]+'.')
        if attrs['State'] in states: parts.append('Location: '+states[attrs['State']]+'.')
        if attrs['Fee'].isdigit(): parts.append(f"Adoption fee: {attrs['Fee']} Malaysian ringgit.")
        r.update(fields=attrs,structured=' '.join(parts),description=description,breed_names=breed)
    write('records.json',records)
    return records


def encode_long(encoder,texts):
    from clip.simple_tokenizer import SimpleTokenizer
    tok=SimpleTokenizer()
    chunks,owners=[],[]
    lengths=[]
    for i,text in enumerate(texts):
        ids=tok.encode(text)
        lengths.append(len(ids))
        for offset in range(0,max(1,len(ids)),60):
            chunks.append(tok.decode(ids[offset:offset+60]) if ids else 'unknown')
            owners.append(i)
    accum=np.zeros((len(texts),512),np.float32)
    for offset in range(0,len(chunks),256):
        vectors=encoder.encode_text_batch(chunks[offset:offset+256])
        np.add.at(accum,np.array(owners[offset:offset+256]),vectors)
    accum/=np.bincount(owners,minlength=len(texts))[:,None]
    return normalize_rows(accum),dict(texts=len(texts),chunks=len(chunks),longer_than_75_tokens=sum(x>75 for x in lengths),max_tokens=max(lengths))


def features(records):
    fingerprint=hashlib.sha256((OUT/'records.json').read_bytes()).hexdigest()
    if (OUT/'profiles.npz').exists():
        a=np.load(OUT/'profiles.npz');assert str(a['records_sha256'])==fingerprint
        return a
    encoder=ClipEncoder(device='cuda')
    started=time.perf_counter()
    structured,sm=encode_long(encoder,[r['structured'] for r in records])
    print('Structured profiles encoded',sm,flush=True)
    description,dm=encode_long(encoder,[r['description'] for r in records])
    profile=normalize_rows(.5*structured+.5*description)
    np.savez_compressed(OUT/'profiles.npz',structured=structured,description=description,profile=profile,ids=np.array([r['pet_id'] for r in records]),records_sha256=np.array(fingerprint))
    write('profiles_manifest.json',dict(structured=sm,description=dm,seconds=time.perf_counter()-started,records_sha256=fingerprint))
    del encoder
    torch.cuda.empty_cache()
    print('All profiles encoded',dm,flush=True)
    return np.load(OUT/'profiles.npz')


def pair_loss(mapped,visual):
    logits=mapped@visual.T/.07
    ids=torch.arange(len(mapped),device='cuda')
    return .5*(torch.nn.functional.cross_entropy(logits,ids)+torch.nn.functional.cross_entropy(logits.T,ids))


def train(records,profiles):
    old=np.load(BASE/'features.npz')
    active=[r for r in records if r['split']!='test']
    idx=[i for i,r in enumerate(records) if r['split']!='test']
    assert list(profiles['ids'][idx])==[r['pet_id'] for r in active]
    texts=torch.tensor(old['text'],device='cuda')
    full=torch.tensor(np.stack([profiles[k][idx] for k in ['profile','structured','description']],axis=1),device='cuda')
    signatures=list(old['signatures'])
    labels=torch.tensor([signatures.index(r['signature']) for r in active],device='cuda')
    tr=torch.tensor([i for i,r in enumerate(active) if r['split']=='train'],device='cuda')
    va=torch.tensor([i for i,r in enumerate(active) if r['split']=='validation'],device='cuda')
    report=dict(version='full-notice-v2',runs=[],seeds=[17,29,43],objective='0.5 attribute multipositive + 0.5 full-notice instance contrast',
                same_split_as_v1=True,test_evaluated=False,encoders_frozen=True,max_epochs=150,learning_rate=.002,batch_size=512,
                validation='mean of attribute loss without centroid and full-profile instance loss',profile_variants=['profile','structured','description'])
    for target,arch in [('clip','linear'),('dino','linear'),('dino','mlp'),('dino','flow')]:
        visual=torch.tensor(old[target],device='cuda')
        centroids=torch.zeros((len(signatures),visual.shape[1]),device='cuda')
        for i in range(len(signatures)):
            matches=tr[labels[tr]==i]
            if len(matches): centroids[i]=visual[matches].mean(0)
        centroids=torch.nn.functional.normalize(centroids,dim=-1)
        for seed in report['seeds']:
            torch.manual_seed(seed)
            head=make_projection_head(arch,input_dim=512,output_dim=visual.shape[1]).cuda()
            opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.0001)
            best,stale,state,history=float('inf'),0,None,[]
            start=time.perf_counter()
            for epoch in range(1,151):
                head.train()
                perm=tr[torch.randperm(len(tr),device='cuda')]
                for offset in range(0,len(tr),512):
                    b=perm[offset:offset+512]
                    attr=project_embeddings(head,texts[labels[b]*4+torch.randint(3,(len(b),),device='cuda')])
                    variant=torch.randint(3,(len(b),),device='cuda')
                    mapped=project_embeddings(head,full[b,variant])
                    loss=.5*multipositive_loss(torch,attr,visual[b],labels[b],labels[b],centroids,temperature=.07,centroid_weight=.25,symmetric=True)+.5*pair_loss(mapped,visual[b])
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if epoch%5: continue
                head.eval()
                with torch.inference_mode():
                    attr=project_embeddings(head,texts[labels[va]*4+3])
                    mapped=project_embeddings(head,full[va,0])
                    value=float(.5*multipositive_loss(torch,attr,visual[va],labels[va],labels[va],centroids,temperature=.07,centroid_weight=0.,symmetric=False)+.5*pair_loss(mapped,visual[va]))
                assert np.isfinite(value)
                history.append(dict(epoch=epoch,validation_loss=value))
                if value<best-1e-5:
                    best,stale,best_epoch=value,0,epoch
                    state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                else: stale+=1
                if epoch%25==0: print(target,arch,seed,'epoch',epoch,'val',round(value,4),flush=True)
                if stale>=6: break
            assert state is not None
            cp=OUT/f'{target}_{arch}_seed{seed}.pt'
            save_checkpoint_atomic(dict(architecture=arch,input_dim=512,output_dim=visual.shape[1],state_dict=state,seed=seed,target=target),cp,torch)
            report['runs'].append(dict(target=target,architecture=arch,seed=seed,best_epoch=best_epoch,epochs_completed=epoch,validation_loss=best,seconds=time.perf_counter()-start,checkpoint=str(cp),history=history))
            write('training_report.json',report)
            print('FINISHED',target,arch,seed,round(time.perf_counter()-start,1),'s',flush=True)
    report['complete']=True
    write('training_report.json',report)
    print('FULL NOTICE TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    records=prepare()
    train(records,features(records))
