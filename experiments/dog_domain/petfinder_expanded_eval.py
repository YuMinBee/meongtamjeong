"""Full-notice core, attribute and explicitly recorded behavior evaluation."""
import json
import re
import hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.dog_domain.petfinder_expanded import OUT,BASE,read
from experiments.dog_domain import petfinder_core_eval as core
from experiments.dino_fusion.core import ClipEncoder,normalize_rows
from experiments.dino_fusion.alignment import projection_head_from_checkpoint
from experiments.composed_retrieval.metrics import cluster_interval

EVAL=OUT/'evaluation'
AXES={
 'activity':dict(pos=r'\b(?:active|energetic|high[ -]energy|full of energy)\b',neg=r'\b(?:calm|laid[ -]back|low[ -]energy|not (?:very )?active)\b',
                 positive='a dog that enjoys exercise and has lots of energy',negative='a relaxed dog with a low activity level'),
 'people':dict(pos=r'\b(?:friendly (?:with|to|towards) (?:people|humans|children|kids)|loves? (?:people|humans|children|kids)|affectionate|loves? (?:cuddles|cuddling))\b',
               neg=r'\b(?:afraid of (?:people|humans|strangers)|fearful of (?:people|strangers)|shy (?:with|around) (?:people|strangers)|not friendly (?:with|to|towards) (?:people|humans))\b',
               positive='a dog that enjoys human company and being close to its adopter',negative='a dog that feels fearful around unfamiliar humans'),
 'dogs':dict(pos=r'\b(?:good|friendly|great) (?:with|to|towards) (?:other )?dogs\b|\bgets? along (?:well )?with (?:other )?dogs\b',
             neg=r'\b(?:not good|not friendly|aggressive) (?:with|to|towards) (?:other )?dogs\b|\b(?:does not|doesn.t|cannot) get along with (?:other )?dogs\b',
             positive='a dog that can comfortably share a home with canine companions',negative='a dog that has difficulty getting along with canine companions')}


def write(name,value):
    EVAL.mkdir(parents=True,exist_ok=True)
    (EVAL/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def map_text(run,text):
    cp=torch.load(run['checkpoint'],map_location='cpu',weights_only=True)
    model=projection_head_from_checkpoint(cp).cuda().eval()
    model.load_state_dict(cp['state_dict'])
    with torch.inference_mode():
        v=normalize_rows(model(torch.tensor(text,device='cuda')).cpu().numpy())
    assert np.isfinite(v).all()
    return v


def metrics(scores,relevance,refs=None):
    scores=scores.copy()
    if refs is not None: scores[np.arange(len(refs)),refs]=-np.inf
    top=np.argsort(-scores,axis=1,kind='stable')[:,:min(10,scores.shape[1])]
    weights=1/np.log2(np.arange(2,top.shape[1]+2))
    ideal=np.array([weights[:min(len(weights),int(r.sum()))].sum() for r in relevance])
    hit=np.take_along_axis(relevance,top,axis=1)
    ndcg=np.divide((hit*weights).sum(1),ideal,out=np.full(len(ideal),np.nan),where=ideal>0)
    return np.column_stack([ndcg,hit.mean(1)]),top


def attributes(records,f,profiles,encoder,newruns,oldruns):
    def value(r,kind):
        x=r['fields'][kind]
        if kind=='Age':
            n=int(x);return None if n==0 else 'young' if n<=12 else 'adult' if n<=84 else 'senior'
        return None if x in ('0','3') and kind=='Gender' or x=='0' else x
    prompts={'Gender':{'1':'male','2':'female'},'FurLength':{'1':'short-haired','2':'medium-haired','3':'long-haired'},
             'Age':{'young':'aged at most 12 months','adult':'aged over 12 months and at most 84 months','senior':'aged over 84 months'}}
    results={}
    all_metrics={}
    for kind in ['Age','Gender','FurLength','Breed1']:
        refs=np.array([i for i,r in enumerate(records) if value(r,kind) is not None])
        extra=[(records[i]['breed_names'][0] if records[i]['breed_names'] else 'unknown breed') if kind=='Breed1' else prompts[kind][value(records[i],kind)] for i in refs]
        texts=[f"Find a {records[i]['size']} dog with {' and '.join(records[i]['colors'])} fur, {x}." for i,x in zip(refs,extra)]
        t=np.concatenate([encoder.encode_text_batch(texts[o:o+128]) for o in range(0,len(texts),128)])
        rel=np.array([[r['size']==s['size'] and bool(set(r['colors'])&set(s['colors'])) and value(r,kind)==value(s,kind) for s in records] for r in [records[i] for i in refs]])
        rel[np.arange(len(refs)),refs]=False
        known=rel.any(1); groups=np.array([records[i]['rescuer'] for i in refs])[known]
        vals={}; rankings={}
        def add(name,score):
            v,rank=metrics(score,rel,refs);vals[name]=v;rankings[name]=rank
        add('CLIP_mix',normalize_rows(.8*f['clip'][refs]+.2*t)@f['clip'].T)
        add('DINO_image',f['dino'][refs]@f['dino'].T)
        add('CLIP_profile_mix',.8*(f['clip'][refs]@f['clip'].T)+.2*(t@profiles['profile'].T))
        add('DINO_profile_mix',.8*(f['dino'][refs]@f['dino'].T)+.2*(t@profiles['profile'].T))
        for version,runs in [('v1',oldruns),('v2',newruns)]:
            for target,arch in [('clip','linear'),('dino','linear'),('dino','mlp'),('dino','flow')]:
                scores=[];vv=[]
                for run in runs:
                    if run['target']!=target or run['architecture']!=arch:continue
                    mapped=map_text(run,t)
                    score=normalize_rows(.8*f[target][refs]+.2*mapped)@f[target].T
                    v,_=metrics(score,rel,refs);vv.append(v);scores.append(score)
                name=f'{version}_{target}_{arch}'
                vals[name]=np.mean(vv,axis=0)
                if arch=='flow' and version=='v2':
                    rankings[name]=np.argsort(-np.mean(scores,axis=0),axis=1,kind='stable')[:,:11]
        result=dict(queries=len(refs),evaluable_queries=int(known.sum()),methods={k:cluster_interval(v[known],groups) for k,v in vals.items()},
                    paired_v2_minus_v1=cluster_interval((vals['v2_dino_flow']-vals['v1_dino_flow'])[known],groups))
        results[kind]=result
        all_metrics.update({kind+'/'+k:v for k,v in vals.items()})
        np.savez_compressed(EVAL/f'{kind}_rankings.npz',**rankings,refs=refs,relevance=rel,known=known,texts=np.array(texts))
        print('ATTRIBUTE',kind,result['evaluable_queries'],{k:round(v['mean'][0]*100,2) for k,v in result['methods'].items()},flush=True)
    write('attribute_results.json',results)
    np.savez_compressed(EVAL/'attribute_per_query.npz',**all_metrics)
    return results


def behavior_label(text,axis):
    text=text.lower()
    neg=list(re.finditer(axis['neg'],text));pos=[]
    for match in re.finditer(axis['pos'],text):
        prefix=text[max(0,match.start()-24):match.start()]
        if re.search(r'\b(?:not|never|isn.t|no)\s+(?:very\s+)?$',prefix):continue
        if any(m.start()<=match.start()<m.end() for m in neg):continue
        pos.append(match)
    if bool(pos)==bool(neg):return 0,None
    m=(pos or neg)[0]
    return (1 if pos else -1),text[max(0,m.start()-40):min(len(text),m.end()+60)]


def behavior(records,f,profiles,encoder,newruns,oldruns,architectures=None):
    result={};evidence=[]
    for key,axis in AXES.items():
        labels=[]
        for r in records:
            label,snippet=behavior_label(r['description'],axis);labels.append(label)
            if label:evidence.append(dict(pet_id=r['pet_id'],axis=key,label=label,snippet=snippet))
        labels=np.array(labels);ix=np.flatnonzero(labels)
        counts=dict(positive=int((labels==1).sum()),negative=int((labels==-1).sum()),unknown_or_conflicting=int((labels==0).sum()))
        if min(counts['positive'],counts['negative'])<3:
            result[key]=dict(counts=counts,status='insufficient opposing examples');continue
        query=[axis['positive'],axis['negative']]
        t=encoder.encode_text_batch(query)
        rel=np.array([labels[ix]==1,labels[ix]==-1])
        def summarize(score):
            v,_=metrics(score,rel)
            order=np.argsort(-score,axis=1,kind='stable');hits=np.take_along_axis(rel,order,1)
            precisions=hits.cumsum(1)/np.arange(1,len(ix)+1)
            ap=(precisions*hits).sum(1)/hits.sum(1)
            return dict(ndcg10=v[:,0].tolist(),p10=v[:,1].tolist(),ap=ap.tolist(),macro_ap=float(ap.mean()))
        results={'CLIP_image_text':summarize(t@f['clip'][ix].T),
                 'description_text':summarize(t@profiles['description'][ix].T),
                 'profile_text':summarize(t@profiles['profile'][ix].T)}
        for version,runs in [('v1',oldruns),('v2',newruns)]:
            for target,arch in (architectures or [('clip','linear'),('dino','flow')]):
                seed_results=[]
                for run in runs:
                    if run['target']==target and run['architecture']==arch:
                        seed_results.append(summarize(map_text(run,t)@f[target][ix].T))
                results[f'{version}_{target}_{arch}']={k:np.mean([r[k] for r in seed_results],axis=0).tolist() for k in seed_results[0]}
        result[key]=dict(status='evaluated',counts=counts,queries=query,methods=results)
        print('BEHAVIOR',key,counts,{k:round(v['macro_ap']*100,2) for k,v in results.items()},flush=True)
    write('behavior_results.json',dict(axes=result,label_source='fixed explicit-description rules; not human-validated behavior',
                                      unknown_not_negative=True,queries_per_axis=2,confidence_intervals=False))
    write('behavior_evidence.json',evidence)
    return result


def main():
    torch.set_num_threads(4)
    EVAL.mkdir(parents=True,exist_ok=True)
    nr=read(OUT/'training_report.json');oldruns=read(BASE/'training_report.json')['runs'];assert nr['complete']
    core.OUT=EVAL
    # Preserve the already encoded PetFinder test cache.
    pf_cache=BASE/'core_evaluation/petfinder_features.npz'
    f=np.load(pf_cache)
    records_all=read(OUT/'records.json');ix=[i for i,r in enumerate(records_all) if r['split']=='test']
    records=[records_all[i] for i in ix]
    assert list(f['ids'])==[r['pet_id'] for r in records]
    profiles_all=np.load(OUT/'profiles.npz')
    profiles={k:profiles_all[k][ix] for k in ['profile','structured','description']}
    pf=dict(cq=f['clip'],cg=f['clip'],dq=f['dino'],dg=f['dino'],text=f['text'],attrs=[(r['colors'],r['size']) for r in records],groups=[r['rescuer'] for r in records],ids=f['ids'],exact=False,identity=False)
    with torch.inference_mode():
        for name,data in {'PetFinder':pf,**core.external()}.items():core.evaluate(name,data,nr['runs'])
    encoder=ClipEncoder(device='cuda')
    attributes(records,f,profiles,encoder,nr['runs'],oldruns)
    behavior(records,f,profiles,encoder,nr['runs'],oldruns)
    write('provenance.json',dict(training_report_sha256=hashlib.sha256((OUT/'training_report.json').read_bytes()).hexdigest(),
                               evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_ids_sha256=hashlib.sha256('\n'.join(f['ids']).encode()).hexdigest(),complete=True))
    print('EXPANDED EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()
