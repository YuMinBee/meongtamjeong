"""Reuse frozen original grades; require all added grades before four-method scores."""
import json,sqlite3,hashlib
from pathlib import Path
from datetime import datetime,timezone
from experiments.dog_domain.visual_study_metrics import summarize

ROOT=Path('D:/meongtamjeong_research')
SUP=ROOT/'human_visual_dino_supplement_v1'

def main():
    original=ROOT/'human_visual_study_v1'
    study=json.loads((SUP/'merged_study.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((original/'study.json').read_bytes()).hexdigest()==study['source_study_sha256']
    final=Path((original/'audits/latest_five_raters_path.txt').read_text())
    with sqlite3.connect(final.parent/'ratings_snapshot.sqlite3') as db:
        db.row_factory=sqlite3.Row;rows=[dict(r) for r in db.execute('SELECT * FROM ratings')]
    original_ids={p['id'] for p in json.loads((original/'study.json').read_text(encoding='utf-8'))['pairs']}
    new_ids={p['id'] for p in study['pairs']}-original_ids
    assert len(new_ids)==23
    if (SUP/'ratings.sqlite3').exists():
        with sqlite3.connect('file:'+str(SUP/'ratings.sqlite3')+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row;extra=[dict(r) for r in db.execute('SELECT * FROM ratings')]
    else:extra=[]
    assert all(r['pair_id'] in new_ids for r in extra)
    raters=['R1','R2','R3','R4','R5']
    progress={r:sum(x['rater']==r and x['score'] is not None for x in extra) for r in raters}
    if any(n!=len(new_ids) for n in progress.values()):
        print(json.dumps(dict(status='pending',required_per_rater=len(new_ids),progress=progress)))
        return
    report=summarize(study,rows+extra,raters)
    assert report['status']=='complete'
    report['positive_joint_only']={m:sum(x['joint_ndcg5'] for x in vals if not x['zero_joint_ideal'])/sum(not x['zero_joint_ideal'] for x in vals) for m,vals in report['per_query'].items()}
    out=SUP/'audits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True)
    (out/'merged_ratings.json').write_text(json.dumps(rows+extra),encoding='utf-8')
    (out/'four_method_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['methods'],indent=2))
    print('Saved:',out)

if __name__=='__main__':main()
