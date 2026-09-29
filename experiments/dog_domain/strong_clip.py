"""SigLIP 2 / CLIP ViT-L baselines; see STRONG_CLIP_PROTOCOL.md.

Requires related_baselines (features, heads) and petfinder_identity outputs to exist.
"""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from experiments.dog_domain import related_baselines as rb
from experiments.dog_domain import petfinder_core_eval as core
from experiments.dog_domain import petfinder_identity as pi
from experiments.dog_domain.petfinder_train import ARCHIVE
from experiments.dino_fusion.core import normalize_rows
from experiments.composed_retrieval.metrics import cluster_interval

OUT = rb.OUT / 'strong_clip'
PROTOCOL = Path(__file__).with_name('STRONG_CLIP_PROTOCOL.md')
RESULTS = Path(__file__).with_name('STRONG_CLIP_RESULTS.md')
MODELS = {'siglip2L': 'google/siglip2-so400m-patch14-384', 'siglip2B': 'google/siglip2-base-patch16-224',
          'clipL': 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'}


def pooled(output):
    """transformers returns a tensor or a ModelOutput depending on version."""
    if torch.is_tensor(output):
        return output
    return output.pooler_output if getattr(output, 'pooler_output', None) is not None else output[0]


class Dual:
    def __init__(self, model_id):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(model_id).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.siglip = 'siglip' in model_id
        self.meta = dict(model=model_id, revision=getattr(self.model.config, '_commit_hash', None))

    @torch.inference_mode()
    def images(self, images):
        x = self.processor(images=images, return_tensors='pt')['pixel_values'].cuda()
        return normalize_rows(pooled(self.model.get_image_features(pixel_values=x)).float().cpu().numpy())

    @torch.inference_mode()
    def texts(self, texts):
        options = dict(padding='max_length', max_length=64, truncation=True) if self.siglip else dict(padding=True, truncation=True)
        t = self.processor(text=texts, return_tensors='pt', **options)
        t = {k: v.cuda() for k, v in t.items() if k in ('input_ids', 'attention_mask')}
        return normalize_rows(pooled(self.model.get_text_features(**t)).float().cpu().numpy())


def features():
    image_views, prompts = rb.views()
    records = [r for r in rb.read(rb.V1 / 'records.json') if r['split'] == 'test']
    multi = [i for i, r in enumerate(records) if len(r['photos']) >= 2]
    image_views = {k: v for k, v in image_views.items() if k != 'PetFinder_train'}
    image_views['PetFinder_second'] = [('zip', records[i]['photos'][1]) for i in multi]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model_id in MODELS.items():
        cache = OUT / f'features_{name}.npz'
        if cache.exists():
            continue
        encoder = Dual(model_id)
        arrays = {}
        with ZipFile(ARCHIVE) as archive:
            for view, items in image_views.items():
                arrays[view] = np.concatenate([encoder.images(rb.load_images(items[s:s + 32], archive)) for s in range(0, len(items), 32)])
        for dataset, texts in prompts.items():
            arrays[f'{dataset}_text'] = np.concatenate([encoder.texts(texts[i:i + 64]) for i in range(0, len(texts), 64)])
        tmp = cache.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        tmp.replace(cache)
        rb.write(f'strong_clip/features_{name}.json', dict(**encoder.meta, sha256=rb.digest(cache), environment=rb.environment(),
                                                        shapes={k: list(v.shape) for k, v in arrays.items()}))
        print('Encoded', name, flush=True)
        del encoder
        torch.cuda.empty_cache()
    return multi


def main():
    torch.set_num_threads(4)
    multi = features()
    feats = {k: np.load(OUT / f'features_{k}.npz') for k in MODELS}
    base = {'PetFinder': core.petfinder(), **core.external()}
    keys = {'PetFinder': ('PetFinder_image', 'PetFinder_image'), 'Korea': ('Korea_query', 'Korea_gallery'), 'Taiwan': ('Taiwan_image', 'Taiwan_image')}
    results = {}
    comparisons = [('dino_linear_mix', 'siglip2L_mix'), ('dinoL_linear_mix', 'siglip2L_mix'), ('dino_linear_mix', 'siglip2B_mix'), ('dino_linear_mix', 'clipL_mix')]
    with torch.inference_mode():
        for name, data in base.items():
            qk, gk = keys[name]
            methods = {}
            for m, f in feats.items():
                methods.update({f'{m}_image': (f[qk], f[gk]), f'{m}_text_only': (f[f'{name}_text'], f[gk]), f'{m}_mix': (rb.mix(f[qk], f[f'{name}_text']), f[gk])})
            metrics, identity, known = rb.evaluate_dataset(name, data, methods)
            # Existing methods come from the stored per-query arrays of related_baselines (same queries, same order).
            stored = np.load(rb.OUT / f'{name}_per_query.npz')
            assert np.array_equal(stored['known'], known)
            for k in ['CLIP_mix', 'DINO_image', 'dino_linear_mix', 'DINOL_image', 'dinoL_linear_mix', 'dinotxt_image', 'dinotxt_mix']:
                metrics[k] = stored[k]
                if identity:
                    identity[k] = stored[f'identity_{k}']
            groups = np.array(data['groups'])
            results[name] = dict(evaluable_queries=int(known.sum()),
                                 methods={k: dict(semantic=cluster_interval(v[known], groups[known]), **({'identity': cluster_interval(identity[k], groups)} if identity else {}))
                                          for k, v in metrics.items()},
                                 paired={f'{a} minus {b}': cluster_interval((metrics[a] - metrics[b])[known], groups[known]) for a, b in comparisons})
            if identity:
                results[name]['identity_paired'] = {f'{a} minus {b}': cluster_interval(identity[a] - identity[b], groups)
                                                    for a, b in [('DINO_image', 'siglip2L_image'), ('DINOL_image', 'siglip2L_image')]}
            np.savez_compressed(OUT / f'{name}_per_query.npz', **metrics, **{f'identity_{k}': v for k, v in identity.items()}, known=known, groups=groups)
            print(name, {k: round(results[name]['methods'][k]['semantic']['mean'][0] * 100, 2) for k in ['siglip2L_mix', 'siglip2B_mix', 'clipL_mix', 'dino_linear_mix']}, flush=True)
        # PetFinder same-animal retrieval, reusing the petfinder_identity query set and near-copy flags.
        stored = np.load(pi.OUT / 'per_query.npz')
        assert list(stored['targets']) == multi
        keep = ~stored['near_copy']
        groups = stored['groups']
        scores = {f'{m}_image': pi.identity(f['PetFinder_second'], f['PetFinder_image'], stored['targets']) for m, f in feats.items()}
        scores.update({k: stored[k] for k in ['CLIP_image', 'DINO_image', 'DINOL_image', 'dinotxt_image']})
        results['PetFinder_identity'] = dict(queries=int(keep.sum()), methods={k: cluster_interval(v[keep], groups[keep]) for k, v in scores.items()},
                                             paired={f'{a} minus {b}': cluster_interval((scores[a] - scores[b])[keep], groups[keep])
                                                     for a, b in [('DINO_image', 'siglip2L_image'), ('DINOL_image', 'siglip2L_image')]})
        np.savez_compressed(OUT / 'PetFinder_identity_per_query.npz', **scores, keep=keep, groups=groups)
    rb.write('strong_clip/results.json', dict(protocol_sha256=rb.digest(PROTOCOL), evaluator_sha256=rb.digest(Path(__file__)), environment=rb.environment(), results=results))
    report(results)
    print('STRONG CLIP COMPLETE', flush=True)


def report(results):
    pct = lambda d: f"{d['mean'][0] * 100:.2f}"
    ci = lambda d: f"{d['mean'][0] * 100:+.2f} [{d['ci95_low'][0] * 100:+.2f}, {d['ci95_high'][0] * 100:+.2f}]"
    rows = [('CLIP_mix', 'CLIP B/32 사진+글 (기존 기준선)'), ('clipL_image', 'CLIP L/14 (LAION) 사진만'), ('clipL_mix', 'CLIP L/14 (LAION) 사진+글'),
            ('siglip2B_image', 'SigLIP 2 B/16 사진만'), ('siglip2B_mix', 'SigLIP 2 B/16 사진+글'),
            ('siglip2L_image', 'SigLIP 2 So400m 사진만'), ('siglip2L_text_only', 'SigLIP 2 So400m 글만'), ('siglip2L_mix', 'SigLIP 2 So400m 사진+글'),
            ('DINO_image', 'DINOv3-B 사진만'), ('dino_linear_mix', 'DINOv3-B + Linear (논문 주 방법)'), ('dinoL_linear_mix', 'DINOv3-L + Linear')]
    lines = ['# 강한 CLIP 계열 기준선 결과', '', '[사전 프로토콜](STRONG_CLIP_PROTOCOL.md). 모든 기준선은 공개 가중치 zero-shot, 사진 0.8 + 글 0.2.', '',
             '## 1. 속성 조건 검색 (nDCG@10 × 100)', '', '| 방법 | PetFinder | 한국 | 대만 |', '|---|---:|---:|---:|']
    for k, label in rows:
        lines.append(f'| {label} | ' + ' | '.join(pct(results[d]['methods'][k]['semantic']) for d in rb.DATASETS) + ' |')
    lines += ['', '| 비교 (차이 × 100, 95% 구간) | PetFinder | 한국 | 대만 |', '|---|---|---|---|']
    for k in results['PetFinder']['paired']:
        lines.append(f'| {k} | ' + ' | '.join(ci(results[d]['paired'][k]) for d in rb.DATASETS) + ' |')
    lines += ['', '## 2. 같은 개 찾기 (R@1 × 100)', '', '| 방법 | 한국 | PetFinder |', '|---|---:|---:|']
    for k in ['CLIP_image', 'clipL_image', 'siglip2B_image', 'siglip2L_image', 'dinotxt_image', 'DINO_image', 'DINOL_image']:
        korea = results['Korea']['methods'].get(k)
        lines.append(f"| {k} | {pct(korea['identity']) if korea else '–'} | {pct(results['PetFinder_identity']['methods'][k])} |")
    lines += ['', '| 비교 (R@1 차이 × 100) | 한국 | PetFinder |', '|---|---|---|']
    for k in results['PetFinder_identity']['paired']:
        lines.append(f"| {k} | {ci(results['Korea']['identity_paired'][k])} | {ci(results['PetFinder_identity']['paired'][k])} |")
    lines += ['', '한국 CLIP B/32 사진만(CLIP_image)은 기존 한국 평가 파일에 없어 비워 둔다.', '',
              '## 해석상 주의', '', '- 기준선은 모두 zero-shot이고 우리 헤드는 PetFinder로 학습했다.',
              '- SigLIP 2 So400m은 입력 384px, 학습 데이터·모델 크기가 달라 순수한 구조 비교가 아니다.', '- 구간은 다중비교 보정 전이며, 보정 결과는 MULTIPLICITY_RESULTS.md에 있다.', '']
    RESULTS.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    assert torch.cuda.is_available()
    main()
