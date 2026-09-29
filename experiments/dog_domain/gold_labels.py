"""Blind colour/identity labelling UI for Korean notices; see GOLD_LABEL_PROTOCOL.md.

  python -m experiments.dog_domain.gold_labels build            # writes gold_label_ui.html next to the images
  python -m experiments.dog_domain.gold_labels build --rater2   # separate storage for an independent second rater
"""
import argparse
import json
import random
from pathlib import Path
from experiments.dog_domain.notice_extension import OUT as KR

SEED = 20260928
COLORS = [('white', '흰색'), ('black', '검은색'), ('brown', '갈색'), ('tan', '황갈색'), ('cream', '크림'),
          ('yellow', '노란색'), ('gold', '골드'), ('gray', '회색'), ('spotted', '점박이')]
# Post-hoc sensitivity (GOLD_LABEL_PROTOCOL.md change log): merge colour names with fuzzy boundaries.
MERGED = {'white': 'light', 'cream': 'light', 'brown': 'brownish', 'tan': 'brownish', 'gold': 'brownish', 'yellow': 'brownish'}
MERGED_LABELS = {'light': '밝은색(흰색·크림)', 'brownish': '갈색 계열', 'black': '검은색', 'gray': '회색', 'spotted': '점박이'}
UI = KR / 'gold_label_ui.html'
ORDER = Path(__file__).with_name('gold_label_order.json')


def build(rater2=False):
    ids = [r['notice_id'] for r in json.loads((KR / 'records.json').read_text(encoding='utf-8'))]
    random.Random(SEED).shuffle(ids)
    ORDER.write_text(json.dumps(dict(seed=SEED, order=ids), indent=1), encoding='utf-8')
    html = TEMPLATE.replace('__IDS__', json.dumps(ids)).replace('__COLORS__', json.dumps(COLORS, ensure_ascii=False))
    target = UI
    if rater2:
        # Separate browser storage so the first rater's saved labels are never shown to the second rater.
        html = html.replace("KEY='gold_labels_v1'", "KEY='gold_labels_v1_rater2'").replace('<b>털색 라벨링</b>', '<b>털색 라벨링 (두 번째 평가자)</b>')
        target = UI.with_name('gold_label_ui_rater2.html')
    target.write_text(html, encoding='utf-8')
    print('Wrote', target, len(ids), 'items')


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>털색 라벨링</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#dedbd4;--accent:#2f5d50;--on:#e3eee9;--warn:#9a3b2e}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--card:#222220;--ink:#eceae4;--muted:#a19f98;--line:#3a3935;--accent:#7fb8a4;--on:#27372f;--warn:#e08a78}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0}
header b{font-size:16px}.bar{flex:1;min-width:160px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
main{max-width:1100px;margin:0 auto;padding:16px}
.photos{display:grid;grid-template-columns:1fr 1fr;gap:12px}.photos figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.photos img{width:100%;height:min(46vh,440px);object-fit:contain;background:#0001;border-radius:4px}.photos figcaption{color:var(--muted);font-size:13px;margin-top:4px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-top:12px}
h2{font-size:14px;margin:0 0 8px;color:var(--muted);font-weight:600}
.opts{display:flex;flex-wrap:wrap;gap:8px}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
button.on{background:var(--on);border-color:var(--accent);color:var(--accent);font-weight:600}
button kbd{font:12px ui-monospace,monospace;color:var(--muted);margin-right:6px}
.nav{display:flex;gap:8px;justify-content:space-between;align-items:center;margin-top:12px;flex-wrap:wrap}
.primary{background:var(--accent);color:var(--card);border-color:var(--accent)}
.msg{color:var(--warn);font-size:13px;min-height:1.5em}.muted{color:var(--muted);font-size:13px}
input{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
@media (max-width:640px){.photos{grid-template-columns:1fr}.photos img{height:34vh}}
</style></head><body>
<header><b>털색 라벨링</b><span id="count" class="muted"></span><div class="bar"><i id="prog"></i></div>
<span id="raterid" class="muted"></span><button id="export">JSON 내보내기</button></header>
<main>
<div class="photos"><figure><img id="a" alt="사진 1"><figcaption>사진 1</figcaption></figure><figure><img id="b" alt="사진 2"><figcaption>사진 2</figcaption></figure></div>
<section><h2>1. 사진에 보이는 털색 (여러 개 선택 가능)</h2><div class="opts" id="colors"></div></section>
<section><h2>2. 두 사진이 같은 개인가요?</h2><div class="opts" id="same"></div></section>
<section><h2>3. 사진 문제 (해당 시)</h2><div class="opts" id="flags"></div></section>
<div class="nav"><button id="prev"><kbd>←</kbd>이전</button><span class="msg" id="msg"></span><button id="next" class="primary"><kbd>Enter</kbd>저장 후 다음</button></div>
<p class="muted">단축키: 숫자 1–9 털색 · Y/N/U 같은 개 예/아니오/판단불가 · M 여러 마리 · V 잘 안 보임 · Enter 다음 · ← 이전. 입력은 이 브라우저에 자동 저장되며, 끝나면 'JSON 내보내기'로 파일을 저장해 주세요. 앞에서부터 순서대로 진행해 주세요.</p>
</main>
<script>
const IDS=__IDS__, COLORS=__COLORS__, KEY='gold_labels_v1';
const SAME=[['yes','예','Y'],['no','아니오','N'],['unsure','판단 불가','U']], FLAGS=[['multiple','여러 마리가 찍힘','M'],['unclear','개가 잘 안 보임','V']];
let st={rater:'',labels:{},pos:0};
try{const s=JSON.parse(localStorage.getItem(KEY)||'null');if(s&&s.labels)st=s}catch(e){}
// Anonymous per-browser rater id; no names are collected.
if(!st.rater||!st.rater.startsWith('rater-'))st.rater='rater-'+Math.random().toString(16).slice(2,6);
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const $=id=>document.getElementById(id);
let cur;
function btn(host,label,key,on,fn){const b=document.createElement('button');b.innerHTML=(key?`<kbd>${key}</kbd>`:'')+label;if(on)b.className='on';b.onclick=fn;host.appendChild(b)}
function render(){
  const id=IDS[st.pos];cur=JSON.parse(JSON.stringify(st.labels[id]||{colors:[],same:null,flags:[]}));
  $('a').src=`images/${id}/gallery.png`;$('b').src=`images/${id}/query.png`;
  draw();const done=firstGap();$('count').textContent=`${st.pos+1} / ${IDS.length} · 순서대로 완료 ${done}건`;
  $('prog').style.width=(100*done/IDS.length)+'%';$('raterid').textContent='익명 ID '+st.rater;$('msg').textContent='';
}
function draw(){
  for(const h of ['colors','same','flags'])$(h).innerHTML='';
  COLORS.forEach(([v,l],i)=>btn($('colors'),l,i+1,cur.colors.includes(v),()=>{toggle(cur.colors,v);draw()}));
  SAME.forEach(([v,l,k])=>btn($('same'),l,k,cur.same===v,()=>{cur.same=v;draw()}));
  FLAGS.forEach(([v,l,k])=>btn($('flags'),l,k,cur.flags.includes(v),()=>{toggle(cur.flags,v);draw()}));
}
function toggle(a,v){const i=a.indexOf(v);i<0?a.push(v):a.splice(i,1)}
function firstGap(){let i=0;while(i<IDS.length&&st.labels[IDS[i]])i++;return i}
function next(){
  if(!cur.colors.length&&!cur.flags.includes('unclear')){$('msg').textContent='털색을 하나 이상 고르거나 "잘 안 보임"을 체크해 주세요.';return}
  if(!cur.same){$('msg').textContent='같은 개인지 골라 주세요.';return}
  st.labels[IDS[st.pos]]={...cur,time:new Date().toISOString()};if(st.pos<IDS.length-1)st.pos++;save();render();
}
$('next').onclick=next;$('prev').onclick=()=>{if(st.pos>0){st.pos--;save();render()}};
$('export').onclick=()=>{const out={protocol:'GOLD_LABEL_PROTOCOL.md',seed:__SEED__,rater:st.rater,exported:new Date().toISOString(),order:IDS,labels:st.labels,contiguous_completed:firstGap()};
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download=`gold_labels_${st.rater}_${firstGap()}.json`;a.click()};
document.addEventListener('keydown',e=>{const k=e.key.toUpperCase();
  if(/^[1-9]$/.test(k)){toggle(cur.colors,COLORS[+k-1][0]);draw()}
  else if(k==='Y'||k==='N'||k==='U'){cur.same=SAME.find(s=>s[2]===k)[0];draw()}
  else if(k==='M'||k==='V'){toggle(cur.flags,FLAGS.find(s=>s[2]===k)[0]);draw()}
  else if(e.key==='Enter'){next()}else if(e.key==='ArrowLeft'){$('prev').click()}});
if(!st.labels[IDS[st.pos]])st.pos=Math.min(firstGap(),IDS.length-1);save();render();
</script></body></html>
""".replace('__SEED__', str(SEED))


def merge(colours, merged):
    return {MERGED.get(c, c) for c in colours} if merged else set(colours)


def agreement(records, labels, merged=False):
    """Analysis 1: listing (silver) colours vs photo-based (gold) colours."""
    keys = list(MERGED_LABELS) if merged else [c for c, _ in COLORS]
    rows = [(merge(r['attributes']['colors'], merged), merge(labels[r['notice_id']]['colors'], merged)) for r in records]
    usable = [(s, g) for s, g in rows if g]
    per_color = {}
    for c in keys:
        tp = sum(c in s and c in g for s, g in usable)
        fp = sum(c in s and c not in g for s, g in usable)
        fn = sum(c not in s and c in g for s, g in usable)
        per_color[c] = dict(silver=tp + fp, gold=tp + fn, precision=tp / (tp + fp) if tp + fp else None,
                            recall=tp / (tp + fn) if tp + fn else None)
    return dict(labelled=len(rows), with_visible_colour=len(usable),
                any_overlap=sum(bool(s & g) for s, g in usable) / len(usable),
                exact=sum(s == g for s, g in usable) / len(usable),
                mean_jaccard=sum(len(s & g) / len(s | g) for s, g in usable) / len(usable),
                per_color=per_color)


def analyze(label_file, merged=False):
    import numpy as np
    import torch
    from experiments.dog_domain import petfinder_core_eval as core
    from experiments.dog_domain import related_baselines as rb
    from experiments.composed_retrieval.metrics import cluster_interval
    exported = json.loads(Path(label_file).read_text(encoding='utf-8'))
    assert exported['order'] == json.loads(ORDER.read_text(encoding='utf-8'))['order']
    labels = exported['labels']
    records = json.loads((KR / 'records.json').read_text(encoding='utf-8'))
    # Protocol: only the contiguous prefix of the shuffled order is analysed.
    prefix = set(exported['order'][:exported['contiguous_completed']])
    assert all(r['notice_id'] in prefix for r in records), 'analysis below assumes all 611 were labelled'
    result = dict(rater=exported['rater'], merged_colours=merged, agreement=agreement(records, labels, merged))

    same = [labels[r['notice_id']]['same'] for r in records]
    flags = [labels[r['notice_id']]['flags'] for r in records]
    result['identity_check'] = dict(yes=same.count('yes'), no=same.count('no'), unsure=same.count('unsure'),
                                    multiple_dogs=sum('multiple' in f for f in flags), unclear=sum('unclear' in f for f in flags))

    data = core.external()['Korea']
    assert list(data['ids']) == [r['notice_id'] for r in records]
    size = [r['attributes']['size'] for r in records]
    silver = [merge(r['attributes']['colors'], merged) for r in records]
    gold = [merge(labels[r['notice_id']]['colors'], merged) for r in records]
    n = len(records)
    relevance = {
        'silver': [[size[i] == size[j] and bool(silver[i] & silver[j]) for j in range(n)] for i in range(n)],
        'requested_vs_gold': [[size[i] == size[j] and bool(silver[i] & gold[j]) for j in range(n)] for i in range(n)],
        'gold_vs_gold': [[size[i] == size[j] and bool(gold[i] & gold[j]) for j in range(n)] for i in range(n)],
    }
    f_l, f_x, f_t = (np.load(rb.OUT / f'features_{k}.npz') for k in ['dinoL', 'dinotxt', 't2d'])
    text = data['text']
    methods = {'CLIP_mix': (rb.mix(data['cq'], text), data['cg']), 'DINO_image': (data['dq'], data['dg']),
               'dinotxt_mix': (rb.mix(f_x['Korea_query'], f_x['Korea_text']), f_x['Korea_gallery']),
               't2d_mix': (rb.mix(f_t['Korea_query_avg'], f_t['Korea_text']), f_t['Korea_gallery_avg'])}
    seeded = {}
    v2 = [r for r in rb.read(rb.V2 / 'training_report.json')['runs'] if r['target'] == 'dino' and r['architecture'] == 'linear']
    new = [r for r in rb.read(rb.OUT / 'training_report.json')['runs'] if r['target'] == 'dinoL' and r['architecture'] == 'linear']
    for key, runs, q, g in [('dino_linear_mix', v2, data['dq'], data['dg']), ('dinoL_linear_mix', new, f_l['Korea_query'], f_l['Korea_gallery'])]:
        for run in runs:
            methods[f'{key}_seed{run["seed"]}'] = (rb.mix(q, rb.map_text(run['checkpoint'], text)), g)
            seeded.setdefault(key, []).append(f'{key}_seed{run["seed"]}')
    groups = np.array(data['groups'])
    queries_with_gold = np.array([bool(g) for g in gold])
    result['retrieval'] = {}
    with torch.inference_mode():
        for name, rel in relevance.items():
            metrics, _, known = rb.evaluate_dataset('Korea', dict(data, relevance=rel, identity=False), methods)
            for key, parts in seeded.items():
                metrics[key] = np.mean([metrics.pop(p) for p in parts], axis=0)
            keep = known & queries_with_gold if name == 'gold_vs_gold' else known
            result['retrieval'][name] = dict(
                evaluable_queries=int(keep.sum()),
                methods={k: cluster_interval(v[keep], groups[keep]) for k, v in metrics.items()},
                paired={f'{a} minus {b}': cluster_interval((metrics[a] - metrics[b])[keep], groups[keep])
                        for a, b in [('dino_linear_mix', 'CLIP_mix'), ('dino_linear_mix', 'DINO_image'), ('dinoL_linear_mix', 'dinotxt_mix'), ('dino_linear_mix', 't2d_mix')]})
            print(name, {k: round(v['mean'][0] * 100, 2) for k, v in result['retrieval'][name]['methods'].items()}, flush=True)

    per_query = np.load(rb.OUT / 'Korea_per_query.npz')
    not_different = np.array([s != 'no' for s in same])
    result['identity_excluding_different_dogs'] = {
        k: dict(all=float(per_query[f'identity_{k}'][:, 0].mean()), excluding_no=float(per_query[f'identity_{k}'][not_different, 0].mean()))
        for k in ['DINO_image', 'DINOL_image', 'dinotxt_image', 't2d_image', 'dino_linear_mix', 'dinoL_linear_mix']}
    rb.write('gold_label_results_merged.json' if merged else 'gold_label_results.json', result)
    report(result, merged)


def report(result, merged=False):
    a = result['agreement']
    i = result['identity_check']
    pct = lambda d: f"{d['mean'][0] * 100:.2f}"
    ci = lambda d: f"{d['mean'][0] * 100:+.2f} [{d['ci95_low'][0] * 100:+.2f}, {d['ci95_high'][0] * 100:+.2f}]"
    lines = ['# 한국 공고 색상 정답 검증 결과' + (' — 비슷한 색 묶음 (사후 민감도 분석)' if merged else ''), '',
             *(['밝은색 = 흰색·크림, 갈색 계열 = 갈색·황갈색·골드·노란색. 결과 확인 후 추가한 분석이며 주 결과는 [원래 분석](GOLD_LABEL_RESULTS.md)이다.', ''] if merged else []),
             f"[사전 프로토콜](GOLD_LABEL_PROTOCOL.md). 평가자 1명(저자, 공고 색 비공개), {a['labelled']}건 전체 라벨링.", '',
             '## 1. 공고 색 vs 사진 기준 사람 라벨', '',
             f"- 사진에서 색을 판단할 수 있었던 건: {a['with_visible_colour']} / {a['labelled']}",
             f"- 하나 이상 겹침(주 평가의 정답 규칙): **{a['any_overlap'] * 100:.1f}%**",
             f"- 색 집합 완전 일치: {a['exact'] * 100:.1f}%, 평균 Jaccard {a['mean_jaccard']:.3f}", '',
             '| 색 | 공고 | 사람 | 정밀도 | 재현율 |', '|---|---:|---:|---:|---:|']
    for c, label in (MERGED_LABELS.items() if merged else COLORS):
        p = a['per_color'][c]
        f = lambda v: '–' if v is None else f'{v * 100:.1f}%'
        lines.append(f"| {label} | {p['silver']} | {p['gold']} | {f(p['precision'])} | {f(p['recall'])} |")
    lines += ['', '정밀도 = 공고에 그 색이 적힌 건 중 사진에서도 보인 비율, 재현율 = 사진에 보인 건 중 공고에도 적힌 비율.', '',
              '## 2. 사람 라벨로 다시 채점한 검색 결과 (nDCG@10 × 100)', '',
              '| 방법 | 공고 색 (기존) | 요청 색이 사진에 보이는가 (주) | 사람 색끼리 (보조) |', '|---|---:|---:|---:|']
    r = result['retrieval']
    for k in ['CLIP_mix', 'DINO_image', 'dino_linear_mix', 'dinoL_linear_mix', 'dinotxt_mix', 't2d_mix']:
        lines.append(f'| {k} | ' + ' | '.join(pct(r[n]['methods'][k]) for n in ['silver', 'requested_vs_gold', 'gold_vs_gold']) + ' |')
    lines += ['', '| 비교 (차이 × 100, 95% 구간) | 공고 색 | 주 | 보조 |', '|---|---|---|---|']
    for k in r['silver']['paired']:
        lines.append(f'| {k} | ' + ' | '.join(ci(r[n]['paired'][k]) for n in ['silver', 'requested_vs_gold', 'gold_vs_gold']) + ' |')
    lines += ['', f"평가 가능한 질의 수: 공고 색 {r['silver']['evaluable_queries']}, 주 {r['requested_vs_gold']['evaluable_queries']}, 보조 {r['gold_vs_gold']['evaluable_queries']}.", '',
              '## 3. 같은 개 정답 점검', '',
              f"- 두 사진이 같은 개: 예 {i['yes']}, 아니오 {i['no']}, 판단 불가 {i['unsure']}",
              f"- 사진 문제: 여러 마리 {i['multiple_dogs']}, 잘 안 보임 {i['unclear']}", '',
              '| 방법 | R@1 전체 | R@1 "다른 개" 제외 |', '|---|---:|---:|']
    for k, v in result['identity_excluding_different_dogs'].items():
        lines.append(f"| {k} | {v['all'] * 100:.2f} | {v['excluding_no'] * 100:.2f} |")
    lines += ['', '## 해석상 주의', '', '- 평가자는 저자 1명이다. 공고 색은 보지 않았지만 독립 평가자가 아니며 평가자 간 일치도는 없다.',
              '- 사람 라벨도 사진 조명·화질의 영향을 받는 두 번째 기준이지 절대 정답이 아니다.', '- 크기는 공고 체중 구간을 그대로 쓴다.', '']
    Path(__file__).with_name('GOLD_LABEL_RESULTS_MERGED.md' if merged else 'GOLD_LABEL_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'analyze'])
    parser.add_argument('--labels')
    parser.add_argument('--merged', action='store_true', help='post-hoc: merge fuzzy colour names')
    parser.add_argument('--rater2', action='store_true', help='build: separate storage for a second rater')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.rater2)
    else:
        analyze(args.labels, args.merged)
