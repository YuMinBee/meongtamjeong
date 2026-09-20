"""Pooled visual/joint nDCG and ordinal Krippendorff alpha."""
import math
from collections import defaultdict

def ndcg(grades, pool, k=5):
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(grades[:k]))
    ideal=sum(v/math.log2(i+2) for i,v in enumerate(sorted(pool,reverse=True)[:k]))
    return (dcg/ideal if ideal else 0.),not bool(ideal)

def ordinal_alpha(units):
    # Coincidence matrix with ordinal distances from pooled category frequencies.
    o=[[0.]*4 for _ in range(4)]
    for unit in units:
        values=[v for v in unit if v is not None]
        if len(values)<2:continue
        for a,va in enumerate(values):
            for b,vb in enumerate(values):
                if a!=b:o[va][vb]+=1/(len(values)-1)
    counts=[sum(row) for row in o];n=sum(counts)
    if n<=1:return None
    mids=[];cumulative=0
    for c in counts:mids.append(cumulative+c/2);cumulative+=c
    observed=expected=0.
    for i in range(4):
        for j in range(4):
            delta=(mids[i]-mids[j])**2
            observed+=o[i][j]*delta
            expected+=counts[i]*(counts[j]-(i==j))/(n-1)*delta
    return 1-observed/expected if expected else None

def summarize(study, rows, raters=('R1','R2','R3')):
    expected=set(raters)
    votes=defaultdict(dict)
    for r in rows:
        if r['score'] is not None and r['rater'] in expected:votes[r['pair_id']][r['rater']]=r['score']
    byquery=defaultdict(list)
    for pair in study['pairs']:byquery[pair['query_id']].append(pair)
    methods=list(study['queries'][0]['rankings']);values={m:[] for m in methods};finished=[];units=[]
    for q in study['queries']:
        pool=byquery[q['id']]
        if any(set(votes[p['id']])!=expected for p in pool):continue
        finished.append(q['id'])
        visual={p['candidate_id']:sum(votes[p['id']].values())/len(expected) for p in pool}
        joint={p['candidate_id']:visual[p['candidate_id']]*p['attribute_match'] for p in pool}
        attr={p['candidate_id']:p['attribute_match'] for p in pool}
        units.extend([list(votes[p['id']].values()) for p in pool])
        for m in methods:
            rank=q['rankings'][m]
            v,zv=ndcg([visual[c] for c in rank],list(visual.values()))
            j,zj=ndcg([joint[c] for c in rank],list(joint.values()))
            values[m].append(dict(query=q['id'],visual_ndcg5=v,joint_ndcg5=j,attribute_p5=sum(attr[c] for c in rank)/5,zero_visual_ideal=zv,zero_joint_ideal=zj))
    result=dict(status='complete' if len(finished)==len(study['queries']) else 'pending',complete_queries=len(finished),total_queries=len(study['queries']),rated_pairs=sum(set(v)==expected for v in votes.values()),total_pairs=len(study['pairs']),protocol=study['protocol'])
    result['raters']=sorted(expected)
    # Do not reveal interim comparative scores to influence ongoing evaluations.
    if result['status']=='complete':
        result.update(ordinal_krippendorff_alpha=ordinal_alpha(units),methods={m:{key:sum(v[key] for v in vals)/len(vals) for key in ['visual_ndcg5','joint_ndcg5','attribute_p5']} for m,vals in values.items()},per_query=values,
            zero_visual_ideal_queries=sum(v['zero_visual_ideal'] for v in values[methods[0]]),zero_joint_ideal_queries=sum(v['zero_joint_ideal'] for v in values[methods[0]]))
    return result
