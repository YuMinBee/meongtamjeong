import json
import math
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pytest
from experiments.dog_domain.visual_study_metrics import ndcg,ordinal_alpha,summarize
from experiments.dog_domain.visual_study_server import make_handler,connect

def fixture_study():
    return dict(protocol={},queries=[dict(id='q1',rankings={'a':['c1','c2'],'b':['c2','c1']})],pairs=[dict(id='p1',query_id='q1',candidate_id='c1',reference='ref.jpg',image='1.png',attribute_match=0),dict(id='p2',query_id='q1',candidate_id='c2',reference='ref.jpg',image='2.png',attribute_match=1)],assets={})

def test_common_pool_and_zero_ideal():
    assert ndcg([1,0],[3,1,0])[0]<.3
    assert ndcg([0,0],[0,0])==(0,True)
    assert ndcg([3,1],[1,3])[0]==1

def test_alpha_agreement_and_undefined():
    assert ordinal_alpha([[0,0,0],[1,1,1],[3,3,3]])==1
    assert ordinal_alpha([[1,1],[1,1]]) is None
    assert ordinal_alpha([[0,3],[0,3]])<0

def test_pending_is_not_fabricated_zero_and_joint_is_gated():
    s=fixture_study();rows=[dict(rater=r,pair_id=p,score=3 if p=='p1' else 1) for r in ['R1','R2','R3'] for p in ['p1','p2']]
    assert summarize(s,rows[:-1])['status']=='pending'
    assert 'methods' not in summarize(s,rows[:-1])
    result=summarize(s,rows)
    assert result['status']=='complete'
    assert result['methods']['a']['visual_ndcg5']==1
    assert result['methods']['b']['joint_ndcg5']==1
    assert result['methods']['a']['joint_ndcg5']==pytest.approx(1/math.log2(3))

def test_api_blinding_validation_isolation_and_persistence(tmp_path):
    (tmp_path/'study.json').write_text(json.dumps(fixture_study()))
    (tmp_path/'access.json').write_text(json.dumps(dict(admin='admin-secret',raters={'R1':'one','R2':'two','R3':'three'})))
    # Seed added for task shuffle; no real ratings are touched.
    s=fixture_study();s['protocol']['seed']=1;(tmp_path/'study.json').write_text(json.dumps(s))
    connect(tmp_path).close()
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(tmp_path));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def request(path,token='one',body=None):
        r=Request(f'http://127.0.0.1:{server.server_port}'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with urlopen(r) as result:return json.load(result)
    try:
        session=request('/api/session');raw=json.dumps(session)
        assert not any(key in raw for key in ['attribute_match','rankings','candidate_id','breed'])
        payload=dict(study_id=session['study_id'],pair_id='p1',score=2)
        assert request('/api/rating',body=payload)['saved']
        assert request('/api/session')['ratings'][0]['score']==2
        assert request('/api/session',token='two')['ratings']==[]
        request('/api/rating',body={**payload,'score':3})
        with connect(tmp_path) as db:
            assert db.execute('SELECT count(*) FROM ratings').fetchone()[0]==1
            assert db.execute('SELECT count(*) FROM events').fetchone()[0]==2
        for score in [-1,4,True,'2']:
            with pytest.raises(HTTPError) as e:request('/api/rating',body={**payload,'score':score})
            assert e.value.code==400
        with pytest.raises(HTTPError):request('/api/admin')
        with pytest.raises(HTTPError):request('/api/session',token='wrong')
        assert request('/api/admin',token='admin-secret')['report']['status']=='pending'
        request('/api/rating',body={**payload,'score':None,'skipped':True})
        assert request('/api/export')['ratings'][0]['score'] is None
    finally:server.shutdown();server.server_close();thread.join()
