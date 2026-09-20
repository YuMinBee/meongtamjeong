"""Freeze all five raters and estimate paired breed/condition block intervals."""
import json, sqlite3, hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from experiments.dog_domain.visual_study_metrics import summarize

def main():
    root=Path('D:/meongtamjeong_research/human_visual_study_v1')
    study=json.loads((root/'study.json').read_text(encoding='utf-8'))
    out=root/'audits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_five_raters')
    out.mkdir(parents=True)
    with sqlite3.connect('file:'+str(root/'ratings.sqlite3')+'?mode=ro',uri=True) as db:
        with sqlite3.connect(out/'ratings_snapshot.sqlite3') as snapshot: db.backup(snapshot)
    with sqlite3.connect(out/'ratings_snapshot.sqlite3') as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT * FROM ratings')]
    raters=['R1','R2','R3','R4','R5']
    report=summarize(study,rows,raters)
    assert report['status']=='complete'
    report['analysis_status']='All five raters complete; frozen snapshot for the manuscript.'
    report['study_sha256']=hashlib.sha256((root/'study.json').read_bytes()).hexdigest()
    report['individual']={r:summarize(study,rows,[r])['methods'] for r in raters}
    lookup={(r['rater'],r['pair_id']):r['score'] for r in rows}
    pairs={(p['query_id'],p['candidate_id']):p['id'] for p in study['pairs']}
    for method in report['methods']:
        values=[lookup[r,pairs[q['id'],c]] for q in study['queries'] for c in q['rankings'][method] for r in raters]
        report['methods'][method]['visual_mean']=sum(values)/len(values)
    # Retain each block's five images and all five raters in paired resampling.
    breeds=sorted({q['breed'] for q in study['queries']})
    qbreed={q['id']:q['breed'] for q in study['queries']}
    draws=np.random.default_rng(20260918).integers(0,len(breeds),(10000,len(breeds)))
    report['paired_block_bootstrap']={'replicates':10000,'seed':20260918,'blocks':breeds,'raters_resampled':False,'comparisons':{}}
    for baseline in ['clip_image_text','dino_linear_text_only']:
        comparison={}
        for metric in ['visual_ndcg5','joint_ndcg5']:
            base={v['query']:v[metric] for v in report['per_query'][baseline]}
            delta={v['query']:v[metric]-base[v['query']] for v in report['per_query']['dino_linear_image_text']}
            block=np.array([np.mean([v for q,v in delta.items() if qbreed[q]==b]) for b in breeds])
            comparison[metric]={'difference':float(block.mean()),'percentile95':np.quantile(block[draws].mean(axis=1),[.025,.975]).tolist()}
        report['paired_block_bootstrap']['comparisons'][baseline]=comparison
    (out/'five_raters_final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'audits/latest_five_raters_path.txt').write_text(str(out/'five_raters_final.json'),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_query','protocol')},indent=2))
    print(out)

if __name__=='__main__':main()
