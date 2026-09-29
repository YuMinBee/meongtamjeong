"""Same-animal retrieval on PetFinder test listings; see PETFINDER_IDENTITY_PROTOCOL.md.

Requires related_baselines features/heads (dino.txt, Talk2DINO, DINOv3-L runs) to exist.
"""
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from PIL import Image
from experiments.dog_domain.petfinder_train import OUT as V1, ARCHIVE, REV
from experiments.dog_domain.petfinder_expanded import OUT as V2
from experiments.dog_domain import related_baselines as rb
from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows
from experiments.composed_retrieval.metrics import cluster_interval

OUT = rb.OUT / 'petfinder_identity'
PROTOCOL = Path(__file__).with_name('PETFINDER_IDENTITY_PROTOCOL.md')
RESULTS = Path(__file__).with_name('PETFINDER_IDENTITY_RESULTS.md')


def near_copy(a, b):
    """Taiwan rule: 9x8 dHash Hamming <= 3 and 32x32 thumbnail mean abs diff < 0.035."""
    def dhash(im):
        g = np.asarray(im.convert('L').resize((9, 8)), dtype=np.float32)
        return (g[:, 1:] > g[:, :-1]).flatten()
    thumb = lambda im: np.asarray(im.resize((32, 32)), dtype=np.float32)
    return int((dhash(a) != dhash(b)).sum()) <= 3 and np.abs(thumb(a) - thumb(b)).mean() / 255 < .035


class Clip:
    def __init__(self):
        self.encoder = ClipEncoder(device='cuda')

    @torch.inference_mode()
    def images(self, images):
        x = torch.stack([self.encoder._preprocess(im) for im in images]).cuda()
        return normalize_rows(self.encoder._model.encode_image(x).float().cpu().numpy())


class DinoBase:
    def __init__(self):
        self.encoder = DinoEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m', device='cuda', revision=REV, local_files_only=True)

    def images(self, images):
        return self.encoder.encode_batch(images)


def features(records, pairs):
    """Encode gallery (first photo) and query (second photo) on this machine for all encoders."""
    cache = OUT / 'features.npz'
    if cache.exists():
        return np.load(cache)
    arrays = {}
    with ZipFile(ARCHIVE) as z:
        load = lambda names: [Image.open(io.BytesIO(z.read(n))).convert('RGB') for n in names]
        for name, build in [('clip', Clip), ('dino', DinoBase), ('dinoL', rb.DinoLarge), ('dinotxt', rb.DinoTxt), ('t2d', rb.Talk2Dino)]:
            encoder = build()
            for view, names in [('gallery', [r['photos'][0] for r in records]), ('query', [records[i]['photos'][1] for i in pairs])]:
                chunks = [encoder.images(load(names[s:s + 32])) for s in range(0, len(names), 32)]
                if isinstance(chunks[0], dict):
                    arrays[f'{name}_{view}'] = np.concatenate([c['avg'] for c in chunks])
                else:
                    arrays[f'{name}_{view}'] = np.concatenate(chunks)
            print('encoded', name, flush=True)
            del encoder
            torch.cuda.empty_cache()
    tmp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(cache)
    return np.load(cache)


def identity(query, gallery, targets):
    q, g = torch.tensor(query, device='cuda'), torch.tensor(gallery, device='cuda')
    t = torch.tensor(targets, device='cuda')
    out = []
    for s in range(0, len(q), 256):
        scores = q[s:s + 256] @ g.T
        target = scores.gather(1, t[s:s + 256, None])
        # Rank = 1 + number of gallery items scoring strictly higher than the target.
        rank = (scores > target).sum(1) + 1
        out.append(torch.stack([(rank <= 1).float(), (rank <= 5).float(), (rank <= 10).float(), 1 / rank.float()], 1).cpu().numpy())
    return np.concatenate(out)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    records = [r for r in rb.read(V1 / 'records.json') if r['split'] == 'test']
    multi = [i for i, r in enumerate(records) if len(r['photos']) >= 2]
    flags = []
    with ZipFile(ARCHIVE) as z:
        for i in multi:
            a, b = (z.read(records[i]['photos'][k]) for k in (0, 1))
            same_bytes = hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
            flags.append(same_bytes or near_copy(Image.open(io.BytesIO(a)).convert('RGB'), Image.open(io.BytesIO(b)).convert('RGB')))
    flags = np.array(flags)
    f = features(records, multi)
    cached = np.load(V1 / 'core_evaluation/petfinder_features.npz')
    consistency = {k: float(np.abs(f[f'{k}_gallery'] - cached[k]).max()) for k in ['clip', 'dino']}
    cosine = {k: float((f[f'{k}_gallery'] * cached[k]).sum(1).min()) for k in ['clip', 'dino']}
    print('4090 cache vs this machine: max abs diff', consistency, 'min cosine', cosine, flush=True)
    base_text = cached['text']
    xt, tt = (np.load(rb.OUT / f'features_{k}.npz')['PetFinder_text'] for k in ['dinotxt', 't2d'])
    targets = np.array(multi)
    methods = {}
    for name in ['clip', 'dino', 'dinoL', 'dinotxt', 't2d']:
        label = {'clip': 'CLIP_image', 'dino': 'DINO_image', 'dinoL': 'DINOL_image', 'dinotxt': 'dinotxt_image', 't2d': 't2d_image'}[name]
        methods[label] = (f[f'{name}_query'], f[f'{name}_gallery'])
    methods['CLIP_mix'] = (rb.mix(f['clip_query'], base_text[multi]), f['clip_gallery'])
    methods['dinotxt_mix'] = (rb.mix(f['dinotxt_query'], xt[multi]), f['dinotxt_gallery'])
    methods['t2d_mix'] = (rb.mix(f['t2d_query'], tt[multi]), f['t2d_gallery'])
    v2_runs = [r for r in rb.read(V2 / 'training_report.json')['runs'] if r['target'] == 'dino' and r['architecture'] == 'linear']
    new_runs = [r for r in rb.read(rb.OUT / 'training_report.json')['runs'] if r['target'] == 'dinoL' and r['architecture'] == 'linear']
    scores = {k: identity(q, g, targets) for k, (q, g) in methods.items()}
    for key, runs, visual in [('dino_linear_mix', v2_runs, 'dino'), ('dinoL_linear_mix', new_runs, 'dinoL')]:
        per_seed = [identity(rb.mix(f[f'{visual}_query'], rb.map_text(r['checkpoint'], base_text)[multi]), f[f'{visual}_gallery'], targets) for r in runs]
        scores[key] = np.mean(per_seed, axis=0)
    groups = np.array([records[i]['rescuer'] for i in multi])
    comparisons = [('DINO_image', 'CLIP_image'), ('dino_linear_mix', 'CLIP_mix'), ('dino_linear_mix', 'DINO_image'),
                   ('DINOL_image', 'dinotxt_image'), ('dinoL_linear_mix', 'dinotxt_mix'), ('dinoL_linear_mix', 'DINOL_image'),
                   ('dinotxt_mix', 'dinotxt_image'), ('t2d_mix', 't2d_image')]
    results = {}
    for subset, keep in [('primary_excluding_near_copies', ~flags), ('all_pairs', np.ones_like(flags))]:
        results[subset] = dict(queries=int(keep.sum()), rescuers=len(set(groups[keep])),
                               methods={k: cluster_interval(v[keep], groups[keep]) for k, v in scores.items()},
                               paired={f'{a} minus {b}': cluster_interval((scores[a] - scores[b])[keep], groups[keep]) for a, b in comparisons})
    rb.write('petfinder_identity/results.json', dict(protocol_sha256=rb.digest(PROTOCOL), evaluator_sha256=rb.digest(Path(__file__)),
                                                    environment=rb.environment(), gallery=len(records), multi_photo=len(multi),
                                                    near_copy_excluded=int(flags.sum()), cache_consistency_max_abs=consistency,
                                                    cache_consistency_min_cosine=cosine, metrics=['R@1', 'R@5', 'R@10', 'MRR'], results=results))
    np.savez_compressed(OUT / 'per_query.npz', **scores, near_copy=flags, groups=groups, targets=targets)
    report(results, len(records), len(multi), int(flags.sum()), cosine)
    print('PETFINDER IDENTITY COMPLETE', flush=True)


LABELS = [('CLIP_image', 'CLIP B/32 사진만'), ('DINO_image', 'DINOv3-B 사진만'), ('DINOL_image', 'DINOv3-L 사진만'),
          ('dinotxt_image', 'dino.txt 사진만'), ('t2d_image', 'Talk2DINO 사진만'),
          ('CLIP_mix', 'CLIP 사진+설명'), ('dino_linear_mix', 'DINOv3-B + Linear 사진+설명 (논문 주 방법)'),
          ('dinoL_linear_mix', 'DINOv3-L + Linear 사진+설명'), ('dinotxt_mix', 'dino.txt 사진+설명'), ('t2d_mix', 'Talk2DINO 사진+설명')]


def report(results, gallery, multi, excluded, cosine):
    p = results['primary_excluding_near_copies']
    a = results['all_pairs']
    lines = ['# PetFinder 같은 개 다시 찾기 결과', '',
             f'[사전 프로토콜](PETFINDER_IDENTITY_PROTOCOL.md). 갤러리 {gallery}장(test 대표 사진), 사진 2장 이상 개체 {multi}마리 중 '
             f'거의 같은 사진 {excluded}쌍 제외 → 주 분석 {p["queries"]}개 질의, 등록자 {p["rescuers"]}명.',
             f'값은 × 100. 장비 간 점검: 갤러리 특징의 기존 4090 캐시 대비 최소 코사인 CLIP {cosine["clip"]:.6f}, DINOv3-B {cosine["dino"]:.6f}.', '',
             '| 방법 | R@1 | R@5 | R@10 | MRR | R@1 (제외 없음) |', '|---|---:|---:|---:|---:|---:|']
    for key, label in LABELS:
        m = p['methods'][key]['mean']
        lines.append(f'| {label} | ' + ' | '.join(f'{v * 100:.2f}' for v in m) + f" | {a['methods'][key]['mean'][0] * 100:.2f} |")
    lines += ['', '## 사전 지정 비교 (R@1 차이 × 100, 등록자 단위 paired bootstrap 95% 구간)', '', '| 비교 | 주 분석 | 제외 없음 |', '|---|---|---|']
    fmt = lambda d: f"{d['mean'][0] * 100:+.2f} [{d['ci95_low'][0] * 100:+.2f}, {d['ci95_high'][0] * 100:+.2f}]"
    for key in p['paired']:
        lines.append(f'| {key} | {fmt(p["paired"][key])} | {fmt(a["paired"][key])} |')
    lines += ['', '## 해석상 주의', '',
              '- 같은 공고의 사진은 같은 날·같은 장소에서 찍힌 경우가 많아 배경이 단서가 될 수 있다. 서로 다른 날 찍은 실종·보호 사진 매칭 성능을 직접 뜻하지 않는다.',
              '- "사진+설명"의 설명은 공고 기재 색·크기로 만든 템플릿이다. 실제 보호자가 쓴 자유 서술이 아니다.',
              '- 구간은 다중비교 보정 전 탐색적 구간이다.', '']
    RESULTS.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    assert torch.cuda.is_available()
    main()
