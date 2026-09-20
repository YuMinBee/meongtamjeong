"""Snapshot the requested R1--R4 interim analysis without altering live ratings."""
import json, sqlite3, hashlib
from pathlib import Path
from datetime import datetime, timezone
from experiments.dog_domain.visual_study_metrics import summarize

def main():
    root=Path('D:/meongtamjeong_research/human_visual_study_v1')
    study=json.loads((root/'study.json').read_text(encoding='utf-8'))
    out=root/'audits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_four_raters')
    out.mkdir(parents=True)
    with sqlite3.connect('file:'+str(root/'ratings.sqlite3')+'?mode=ro',uri=True) as db:
        with sqlite3.connect(out/'ratings_snapshot.sqlite3') as snapshot: db.backup(snapshot)
    with sqlite3.connect(out/'ratings_snapshot.sqlite3') as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT * FROM ratings')]
    raters=['R1','R2','R3','R4']
    report=summarize(study,rows,raters)
    assert report['status']=='complete'
    report['analysis_status']='Four completed raters, interim analysis; R5 excluded regardless of progress.'
    report['study_sha256']=hashlib.sha256((root/'study.json').read_bytes()).hexdigest()
    report['individual']={r:summarize(study,rows,[r])['methods'] for r in raters}
    lookup={(r['rater'],r['pair_id']):r['score'] for r in rows}
    pairs={(p['query_id'],p['candidate_id']):p['id'] for p in study['pairs']}
    for method in report['methods']:
        values=[lookup[r,pairs[q['id'],c]] for q in study['queries'] for c in q['rankings'][method] for r in raters]
        report['methods'][method]['visual_mean']=sum(values)/len(values)
    (out/'four_raters_interim.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'audits/latest_four_raters_path.txt').write_text(str(out/'four_raters_interim.json'),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_query','protocol')},indent=2))
    print(out)

if __name__=='__main__':main()
