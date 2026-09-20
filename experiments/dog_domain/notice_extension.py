"""Reproducible, fixed-weight evaluation on newly collected Korean notices."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from app.heldout_image_evaluation import SafePublicImageDownloader, ImageDownloadError
from experiments.composed_retrieval.download import ROOT, sha256
from experiments.composed_retrieval.metrics import cluster_interval
from experiments.dino_fusion.alignment import alignment_attributes, evaluation_prompts, collect_heldout_notice_ids
from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows
from experiments.dino_fusion.evaluate_alignment import load_heads

OUT = ROOT / 'tmp/notice_extension_20260916'
HEADS = ROOT / 'experiments/dino_fusion/artifacts/dinode_flow'
PROTOCOL = Path(__file__).with_name('NOTICE_EXTENSION_PROTOCOL.md')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def pixel_hash(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def near_copy(a, b):
    def bits(im):
        arr = np.asarray(im.convert('L').resize((9, 8)), dtype=float)
        return arr[:, 1:] > arr[:, :-1]
    distance = np.count_nonzero(bits(a) != bits(b))
    difference = np.abs(np.asarray(a.convert('RGB').resize((32, 32)), dtype=float) -
                        np.asarray(b.convert('RGB').resize((32, 32)), dtype=float)).mean() / 255
    return bool(distance <= 3 and difference < .035)


def old_ids():
    paths = list((ROOT / 'data').glob('dog_metas*.json'))
    paths += list((ROOT / 'tmp/dino_fusion_refresh_20260814').glob('*.json'))
    paths += list((ROOT / 'experiments/dino_fusion').glob('artifacts/**/dino_metas.json'))
    found, provenance = set(), {}
    for path in paths:
        obj = read(path)
        rows = obj if isinstance(obj, list) else obj.get('items', [])
        ids = {str(r.get('desertionNo') or r.get('notice_id') or '') for r in rows if isinstance(r, dict)} - {''}
        if ids:
            found |= ids
            provenance[str(path.relative_to(ROOT))] = {'sha256': sha256(path), 'ids': len(ids)}
    path = ROOT / 'docs/evaluation/heldout_image_retrieval.appearance_v1.json'
    if path.exists():
        found |= collect_heldout_notice_ids(read(path))
        provenance[str(path.relative_to(ROOT))] = {'sha256': sha256(path)}
    return found, provenance


def fetch_pair(row):
    notice_id = str(row['desertionNo'])
    target = OUT / 'images' / notice_id
    cache = target / 'record.json'
    if cache.exists():
        return read(cache)
    target.mkdir(parents=True, exist_ok=True)
    downloader = SafePublicImageDownloader(timeout=12, retries=0, allowed_hosts=['openapi.animal.go.kr'])
    try:
        urls = list(dict.fromkeys([row['image_url']] + row['image_urls']))[:4]
        primary = downloader.fetch(urls[0])
        failures = Counter()
        for url in urls[1:]:
            try:
                alternate = downloader.fetch(url)
            except ImageDownloadError as exc:
                failures[str(exc)] += 1
                continue
            if pixel_hash(primary.image) == pixel_hash(alternate.image) or near_copy(primary.image, alternate.image):
                failures['duplicate_or_near_copy'] += 1
                continue
            primary.image.save(target / 'gallery.png')
            alternate.image.save(target / 'query.png')
            result = {'notice_id': notice_id, 'status': 'ok', 'attributes': alignment_attributes(row),
                      'group': hashlib.sha256((row['org_name'] + '|' + row['care_name']).encode()).hexdigest(),
                      'description': row['desc'], 'happen_date': row['happen_date'],
                      'urls': [primary.final_url, alternate.final_url],
                      'payload_hashes': [primary.payload_sha256, alternate.payload_sha256],
                      'pixel_hashes': [pixel_hash(primary.image), pixel_hash(alternate.image)],
                      'image_sha256': [sha256(target / 'gallery.png'), sha256(target / 'query.png')],
                      'alternate_exclusions': dict(failures)}
            write(cache, result)
            return result
        result = {'notice_id': notice_id, 'status': 'no_distinct_alternate', 'reasons': dict(failures)}
    except ImageDownloadError as exc:
        result = {'notice_id': notice_id, 'status': 'primary_download_failed', 'reason': str(exc)}
    finally:
        downloader.close()
    write(cache, result)
    return result


def prepare():
    source = read(OUT / 'notices.json')
    excluded, provenance = old_ids()
    counts, eligible = Counter(), []
    for row in source['items']:
        if str(row['desertionNo']) in excluded:
            counts['old_notice'] += 1
        elif str(row['happen_date']).replace('-', '') < '20260816':
            counts['before_date'] += 1
        elif alignment_attributes(row) is None:
            counts['unknown_color_or_size'] += 1
        else:
            eligible.append(row)
    eligible.sort(key=lambda r: hashlib.sha256(('notice-extension-v1:' + str(r['desertionNo'])).encode()).hexdigest())
    selected = eligible[:1000]
    selection = {'source_sha256': sha256(OUT / 'notices.json'), 'protocol_sha256': sha256(PROTOCOL),
                 'old_sources': provenance, 'old_id_count': len(excluded), 'source_count': len(source['items']),
                 'eligible': len(eligible), 'excluded': dict(counts),
                 'selected_ids': [str(r['desertionNo']) for r in selected]}
    write(OUT / 'selection.json', selection)
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(fetch_pair, selected), 1):
            results.append(result)
            if i % 100 == 0:
                print('downloaded', i, dict(Counter(r['status'] for r in results)), flush=True)
    accepted = [r for r in results if r['status'] == 'ok']
    hashes = Counter(h for r in accepted for h in r['pixel_hashes'])
    records = [r for r in accepted if all(hashes[h] == 1 for h in r['pixel_hashes'])]
    write(OUT / 'records.json', records)
    write(OUT / 'preparation.json', {'selection_sha256': sha256(OUT / 'selection.json'),
          'download_status': dict(Counter(r['status'] for r in results)),
          'cross_notice_exact_duplicate_removed': len(accepted) - len(records),
          'gallery_and_query_notices': len(records), 'shelters': len({r['group'] for r in records}),
          'records_sha256': sha256(OUT / 'records.json')})
    print(read(OUT / 'preparation.json'), flush=True)


def features():
    import torch
    torch.set_num_threads(4)
    records = read(OUT / 'records.json')
    clip = ClipEncoder(device='cuda')
    dino = DinoEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m', device='cuda',
                      revision='5931719e67bbdb9737e363e781fb0c67687896bc', local_files_only=True)
    arrays = {}
    for view in ['query', 'gallery']:
        cf, df = [], []
        for start in range(0, len(records), 32):
            images = []
            for r in records[start:start+32]:
                path = OUT / 'images' / r['notice_id'] / f'{view}.png'
                assert sha256(path) == r['image_sha256'][view == 'query']
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            with torch.inference_mode():
                inputs = torch.stack([clip._preprocess(im) for im in images]).to('cuda')
                cf.append(normalize_rows(clip._model.encode_image(inputs).float().cpu().numpy()))
            df.append(dino.encode_batch(images))
        arrays[f'clip_{view}'] = np.concatenate(cf)
        arrays[f'dino_{view}'] = np.concatenate(df)
        print('encoded', view, len(records), flush=True)
    text_metadata = {}
    for language in ['korean', 'english', 'description']:
        texts = [r['description'] if language == 'description' else evaluation_prompts(r['attributes'])[language] for r in records]
        arrays[f'text_{language}'] = np.concatenate([clip.encode_text_batch(texts[i:i+64]) for i in range(0, len(texts), 64)])
        truncated = 0
        for text in texts:
            try:
                clip._clip.tokenize(text, truncate=False)
            except RuntimeError:
                truncated += 1
        text_metadata[language] = {'truncated': truncated, 'empty': sum(not t.strip() for t in texts)}
    np.savez_compressed(OUT / 'features.npz', **arrays, notice_ids=np.array([r['notice_id'] for r in records]))
    write(OUT / 'features.json', {'records_sha256': sha256(OUT / 'records.json'), 'features_sha256': sha256(OUT / 'features.npz'),
          'protocol_sha256': sha256(PROTOCOL), 'texts': text_metadata, 'dino_revision': dino.resolved_revision,
          'clip_weights_sha256': sha256(Path.home() / '.cache/clip/ViT-B-32.pt')})


def per_query(scores, relevance):
    n = len(scores)
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = np.argmax(order == np.arange(n)[:, None], axis=1) + 1
    masked = scores.copy()
    np.fill_diagonal(masked, -np.inf)
    top = np.argsort(-masked, axis=1, kind='stable')[:, :min(10, n-1)]
    discount = 1 / np.log2(np.arange(2, top.shape[1]+2))
    dcg = (np.take_along_axis(relevance, top, axis=1) * discount).sum(axis=1)
    ideal = np.array([discount[:min(int(row.sum()), len(discount))].sum() for row in relevance])
    ndcg = np.divide(dcg, ideal, out=np.full(n, np.nan), where=ideal > 0)
    return np.column_stack([ranks <= 1, ranks <= 5, ranks <= 10, 1/ranks, ndcg])


def evaluate():
    import torch
    torch.set_num_threads(4)
    records, manifest = read(OUT / 'records.json'), read(OUT / 'features.json')
    assert manifest['records_sha256'] == sha256(OUT / 'records.json')
    assert manifest['features_sha256'] == sha256(OUT / 'features.npz')
    assert manifest['protocol_sha256'] == sha256(PROTOCOL)
    with np.load(OUT / 'features.npz') as cache:
        f = {k: cache[k] for k in cache.files}
    assert list(f['notice_ids']) == [r['notice_id'] for r in records]
    report = read(HEADS / 'alignment_training_report.json')
    heads = load_heads(report=report, artifact_dir=HEADS, device='cuda', torch=torch)
    cq, cg, dq, dg = [f[k] for k in ['clip_query', 'clip_gallery', 'dino_query', 'dino_gallery']]
    scores = {'CLIP_image': cq @ cg.T, 'DINO_image': dq @ dg.T}
    for lang in ['korean', 'english', 'description']:
        text = f[f'text_{lang}']
        scores[f'{lang}/CLIP_mix'] = normalize_rows(.8*cq + .2*text) @ cg.T
        scores[f'{lang}/late_mix'] = .8*(dq @ dg.T) + .2*(text @ cg.T)
        scores[f'{lang}/CLIP_shift'] = normalize_rows(.8*cq + .2*np.roll(text, 1, axis=0)) @ cg.T
        scores[f'{lang}/CLIP_text_only'] = text @ cg.T
        for name, head in heads.items():
            with torch.inference_mode():
                mapped = normalize_rows(head(torch.tensor(text, device='cuda')).cpu().numpy())
            scores[f'{lang}/{name}_mix'] = normalize_rows(.8*dq + .2*mapped) @ dg.T
            scores[f'{lang}/{name}_text_only'] = mapped @ dg.T
            if name == 'flow':
                scores[f'{lang}/flow_shift'] = normalize_rows(.8*dq + .2*np.roll(mapped, 1, axis=0)) @ dg.T
    attributes = [r['attributes'] for r in records]
    relevance = np.array([[bool(set(a['colors']) & set(b['colors'])) and a['size'] == b['size'] for b in attributes] for a in attributes], dtype=float)
    np.fill_diagonal(relevance, 0)
    groups = np.array([r['group'] for r in records])
    arrays = {name: per_query(matrix, relevance) for name, matrix in scores.items()}
    known = relevance.sum(axis=1) > 0
    def summary(values):
        return {'identity': cluster_interval(values[:, :4], groups),
                'semantic': cluster_interval(values[known, 4], groups[known])}
    result = {'status': 'completed', 'n': len(records), 'semantic_n': int(known.sum()), 'shelters': len(set(groups)),
              'metrics': ['R@1', 'R@5', 'R@10', 'MRR', 'color_size_nDCG@10_excluding_self'],
              'preparation': read(OUT / 'preparation.json'), 'features': manifest,
              'training_report_sha256': sha256(HEADS / 'alignment_training_report.json'),
              'shift_same_color_size_fraction': float(np.mean([bool(set(a['colors']) & set(b['colors'])) and a['size'] == b['size'] for a,b in zip(attributes, attributes[-1:] + attributes[:-1])])),
              'results': {name: summary(values) for name, values in arrays.items()}, 'paired': {}}
    for lang in ['korean', 'english', 'description']:
        first = f'{lang}/flow_mix'
        for second in [f'{lang}/CLIP_mix', 'DINO_image', f'{lang}/late_mix', f'{lang}/linear_mix', f'{lang}/mlp_mix', f'{lang}/flow_shift']:
            result['paired'][f'{first} minus {second}'] = summary(arrays[first] - arrays[second])
    write(OUT / 'results.json', result)
    np.savez_compressed(OUT / 'per_query.npz', **arrays, notice_ids=f['notice_ids'], groups=groups)
    np.savez_compressed(OUT / 'scores.npz', **scores, relevance=relevance)
    lines = ['# New Korean notice results', '', f"{len(records)} new notice pairs; {len(set(groups))} shelters; full-photo gallery of {len(records)}.",
             'Fixed text weight .20; frozen original heads; no test tuning. See NOTICE_EXTENSION_PROTOCOL.md.', '',
             '| Method | R@1 | R@5 | R@10 | MRR | Color+size nDCG@10 |', '|---|---:|---:|---:|---:|---:|']
    for name, value in result['results'].items():
        lines.append('| '+name+' | '+' | '.join(f'{v*100:.2f}' for v in value['identity']['mean']+value['semantic']['mean'])+' |')
    lines += ['', 'Paired shelter-bootstrap 95% intervals, percentage points; exploratory contrasts are not multiplicity-adjusted.', '',
              '| Contrast | R@1 difference [95% CI] | Semantic difference [95% CI] |','|---|---|---|']
    for name, value in result['paired'].items():
        cells = [f"{x['mean'][0]*100:+.2f} [{x['ci95_low'][0]*100:+.2f}, {x['ci95_high'][0]*100:+.2f}]" for x in [value['identity'],value['semantic']]]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', 'Public attributes are silver relevance labels, not human query judgments. The query notice is excluded from semantic ranking.',
              'Descriptions are original public text. Their semantic scores measure only color/size agreement, not full-description relevance.',
              'Same-notice identity is a proxy, not verified same-animal identification. Near copies across different notices and repeat animals remain possible.',
              'The original pilot used crops and a different gallery; absolute scores should not be directly compared.', '']
    Path(__file__).with_name('NOTICE_EXTENSION_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:21]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'features', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'features': features, 'evaluate': evaluate}[args.stage]()
