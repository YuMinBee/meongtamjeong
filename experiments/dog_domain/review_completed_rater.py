"""Explicitly requested single-rater audit; never overwrite the primary report."""
import argparse,json,sqlite3
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from experiments.dog_domain.visual_study_metrics import ndcg

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rater',default='R1',choices=['R1','R2','R3','R4','R5'])
    rater=parser.parse_args().rater
    root=Path('D:/meongtamjeong_research/human_visual_study_v1')
    study=json.loads((root/'study.json').read_text(encoding='utf-8'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=root/'audits'/stamp;out.mkdir(parents=True)
    with sqlite3.connect(root/'ratings.sqlite3') as db:
        with sqlite3.connect(out/'ratings_snapshot.sqlite3') as snapshot:db.backup(snapshot)
    with sqlite3.connect(out/'ratings_snapshot.sqlite3') as snapshot:
        snapshot.row_factory=sqlite3.Row
        rows=[dict(r) for r in snapshot.execute('SELECT * FROM ratings WHERE rater=?',(rater,))]
    grades={r['pair_id']:r['score'] for r in rows if r['score'] is not None}
    assert set(grades)=={p['id'] for p in study['pairs']},f'{rater} incomplete'
    pool=defaultdict(list)
    for p in study['pairs']:pool[p['query_id']].append(p)
    methods=list(study['queries'][0]['rankings']);results=[]
    for q in study['queries']:
        visual={p['candidate_id']:grades[p['id']] for p in pool[q['id']]}
        attrs={p['candidate_id']:p['attribute_match'] for p in pool[q['id']]}
        joint={cid:visual[cid]*a for cid,a in attrs.items()}
        for method,rank in q['rankings'].items():
            vn,zv=ndcg([visual[c] for c in rank],list(visual.values()))
            jn,zj=ndcg([joint[c] for c in rank],list(joint.values()))
            results.append(dict(query=q['id'],breed=q['breed'],method=method,visual_ndcg5=vn,joint_ndcg5=jn,attribute_p5=sum(attrs[c] for c in rank)/5,visual_mean=sum(visual[c] for c in rank)/5,zero_visual_ideal=zv,zero_joint_ideal=zj))
    def average(items):
        return {m:{k:sum(r[k] for r in items if r['method']==m)/sum(r['method']==m for r in items) for k in ['visual_ndcg5','joint_ndcg5','attribute_p5','visual_mean']} for m in methods}
    report=dict(status='single-rater exploratory audit, not the final multi-rater result',rater=rater,ratings=len(rows),grade_counts=dict(Counter(grades.values())),queries=30,overall=average(results),by_breed={b:average([r for r in results if r['breed']==b]) for b in sorted({q['breed'] for q in study['queries']})},zero_visual_ideal=sum(r['zero_visual_ideal'] for r in results if r['method']==methods[0]),zero_joint_ideal=sum(r['zero_joint_ideal'] for r in results if r['method']==methods[0]),per_query=results,agreement=None)
    (out/f'{rater}_exploratory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='per_query'},ensure_ascii=False,indent=2))
    print('Audit saved:',out)

if __name__=='__main__':main()
