"""Local/LAN blinded evaluation server. Ratings persist in SQLite on D:."""
import argparse
import hashlib
import json
import random
import sqlite3
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from experiments.dog_domain.visual_study_metrics import summarize

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=Path('D:/meongtamjeong_research/human_visual_study_v1')

def connect(root):
    db=sqlite3.connect(root/'ratings.sqlite3',timeout=20)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS ratings(rater TEXT NOT NULL,pair_id TEXT NOT NULL,score INTEGER CHECK(score BETWEEN 0 AND 3 OR score IS NULL),skipped INTEGER NOT NULL DEFAULT 0,updated TEXT NOT NULL,PRIMARY KEY(rater,pair_id))')
    db.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,rater TEXT,pair_id TEXT,score INTEGER,skipped INTEGER,updated TEXT)')
    db.commit();return db

def make_handler(root, public_raters_only=False):
    study=json.loads((root/'study.json').read_text(encoding='utf-8'))
    access=json.loads((root/'access.json').read_text(encoding='utf-8'))
    pairs={p['id']:p for p in study['pairs']}
    study_id=hashlib.sha256((root/'study.json').read_bytes()).hexdigest()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,data,status=200,kind='application/json; charset=utf-8'):
            content=json.dumps(data,ensure_ascii=False,allow_nan=False).encode() if kind.startswith('application/json') else data
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.send_header('X-Robots-Tag','noindex, nofollow');self.end_headers();self.wfile.write(content)
        def user(self):
            token=self.headers.get('Authorization','').removeprefix('Bearer ')
            if token==access['admin']:return None if public_raters_only else 'admin'
            public_raters=study['protocol'].get('public_raters',[r for r in access['raters'] if r!='R1'])
            return next((r for r,t in access['raters'].items() if t==token and (not public_raters_only or r in public_raters)),None)
        def rows(self):
            with connect(root) as db:return [dict(r) for r in db.execute('SELECT * FROM ratings ORDER BY rater,pair_id')]
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/':return self.send((ROOT/'app/visual_study.html').read_bytes(),kind='text/html; charset=utf-8')
            if path.startswith('/assets/'):
                name=path.removeprefix('/assets/')
                if name not in study['assets']:return self.send({'error':'Not found'},404)
                return self.send((root/'assets'/name).read_bytes(),kind='image/png' if name.endswith('.png') else 'image/jpeg')
            user=self.user()
            if not user:return self.send({'error':'평가 링크를 확인해 주세요.'},401)
            if path=='/api/session':
                if user=='admin':return self.send(dict(role='admin',study_id=study_id))
                rng=random.Random(int(hashlib.sha256(user.encode()).hexdigest()[:12],16)+study['protocol']['seed'])
                query_ids=[q['id'] for q in study['queries']];rng.shuffle(query_ids);tasks=[]
                for q in query_ids:
                    block=[p for p in study['pairs'] if p['query_id']==q];rng.shuffle(block)
                    tasks.extend([{k:p[k] for k in ['id','query_id','reference','image']} for p in block])
                rows=[r for r in self.rows() if r['rater']==user]
                return self.send(dict(role='rater',rater=user,study_id=study_id,tasks=tasks,ratings=rows,study_kind=study['protocol'].get('kind','original')))
            if path=='/api/export':
                rows=self.rows();rows=rows if user=='admin' else [r for r in rows if r['rater']==user]
                return self.send(dict(schema='dog-visual-ratings.v1',study_id=study_id,exported=datetime.now(timezone.utc).isoformat(),ratings=rows))
            if path=='/api/admin' and user=='admin':
                rows=self.rows();progress={r:{'scored':sum(x['rater']==r and x['score'] is not None for x in rows),'skipped':sum(x['rater']==r and bool(x['skipped']) for x in rows),'total':len(pairs)} for r in access['raters']}
                if study['protocol'].get('kind')=='supplement':
                    report=dict(status='supplement_pending_merge',complete_queries=0,total_queries=len(study['queries']))
                else:report=summarize(study,rows,access['raters'])
                return self.send(dict(progress=progress,links={r:'/?key='+t for r,t in access['raters'].items()},report=report))
            return self.send({'error':'Not found'},404)
        def do_POST(self):
            user=self.user()
            if user not in access['raters']:return self.send({'error':'평가자 링크가 필요합니다.'},401)
            if urlsplit(self.path).path!='/api/rating':return self.send({'error':'Not found'},404)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=4096:raise ValueError('invalid size')
                payload=json.loads(self.rfile.read(size))
                pid=payload['pair_id'];score=payload.get('score');skipped=payload.get('skipped',False)
                if payload.get('study_id')!=study_id or pid not in pairs:raise ValueError('invalid study or pair')
                if type(skipped) is not bool:raise ValueError('invalid skip')
                if skipped:
                    if score is not None:raise ValueError('skip must not have grade')
                elif type(score) is not int or score not in range(4):raise ValueError('grade must be 0–3')
                now=datetime.now(timezone.utc).isoformat()
                with connect(root) as db:
                    db.execute('INSERT INTO ratings VALUES(?,?,?,?,?) ON CONFLICT(rater,pair_id) DO UPDATE SET score=excluded.score,skipped=excluded.skipped,updated=excluded.updated',(user,pid,score,int(skipped),now))
                    db.execute('INSERT INTO events(rater,pair_id,score,skipped,updated) VALUES(?,?,?,?,?)',(user,pid,score,int(skipped),now))
                return self.send(dict(saved=True,pair_id=pid,score=score,skipped=skipped))
            except (ValueError,KeyError,json.JSONDecodeError):return self.send({'error':'올바른 점수와 평가 항목을 보내 주세요.'},400)
            except sqlite3.Error:return self.send({'error':'저장하지 못했습니다. 다시 시도해 주세요.'},503)
    return Handler

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DEFAULT);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);p.add_argument('--public-raters-only',action='store_true');args=p.parse_args()
    connect(args.data).close()
    access=json.loads((args.data/'access.json').read_text());url=f'http://localhost:{args.port}/#'+access['admin']
    if not args.public_raters_only:
        (args.data/'OPEN_DASHBOARD.url').write_text('[InternetShortcut]\nURL='+url+'\n',encoding='utf-8')
    print(f'Evaluation server ready on {args.host}:{args.port}. Open {args.data / "OPEN_DASHBOARD.url"}',flush=True)
    ThreadingHTTPServer((args.host,args.port),make_handler(args.data,args.public_raters_only)).serve_forever()

if __name__=='__main__':main()
