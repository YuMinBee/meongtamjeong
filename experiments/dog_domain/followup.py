"""Frozen-fusion paired verification and independent DogFaceNet transfer."""

from collections import Counter, defaultdict
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
from PIL import Image

from experiments.composed_retrieval.download import ROOT, sha256
from experiments.composed_retrieval.features import canonical_provenance, digest
from experiments.composed_retrieval.metrics import cluster_interval
from experiments.composed_retrieval.prepare import load_json, write_json
from experiments.dog_domain.mpdd import OUT

DATA = ROOT / 'tmp/dog_retrieval'
DEST = OUT / 'followup'
METRICS = ['R@1', 'R@5', 'R@10', 'MRR']
SEED = 20260916


def exact_sign_test(values):
    values = np.asarray(values)
    wins, losses = int((values > 1e-12).sum()), int((values < -1e-12).sum())
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    return {'wins': wins, 'losses': losses, 'ties': len(values) - n, 'two_sided_exact_p': p}


def permutation_p(values, samples=100000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.
    observed = abs(values.sum())
    rng = np.random.default_rng(SEED)
    extreme = 0
    for start in range(0, samples, 1000):
        n = min(1000, samples - start)
        signs = rng.integers(0, 2, (n, len(values)), dtype=np.int8) * 2 - 1
        extreme += int((np.abs(signs @ values) >= observed - 1e-12).sum())
    return (extreme + 1) / (samples + 1)


def prepare_dogface():
    expected = {'DogFaceNet_224resized.zip': '010c207e202bb499039452aa7363b015',
                'classes_test.txt': '0afc77355916e658a00b050979534ad7',
                'classes_train.txt': '67f7bc3aecb967e43aa2585a1a4879d6'}
    hashes = {}
    for name, checksum in expected.items():
        path = DATA / name
        if hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Publisher MD5 mismatch: {name}')
        hashes[name] = sha256(path)
    test = set((DATA / 'classes_test.txt').read_text().split())
    train = set((DATA / 'classes_train.txt').read_text().split())
    if test & train:
        raise ValueError('Published identity split overlap')
    previous = load_json(OUT / 'data_audit.json')['records']
    mpdd_dev = {r['pixels_sha256'] for r in previous if r['split'] in ('train', 'val')}
    mpdd_all = {r['pixels_sha256'] for r in previous}
    records = []
    with zipfile.ZipFile(DATA / 'DogFaceNet_224resized.zip') as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC failure')
        for member in sorted(archive.namelist()):
            if not member.lower().endswith('.jpg'):
                continue
            identity = PurePosixPath(member).parent.name
            if identity not in test | train:
                raise ValueError(f'Unlisted identity: {identity}')
            payload = archive.read(member)
            with Image.open(io.BytesIO(payload)) as opened:
                image = opened.convert('RGB')
                image.load()
            pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            records.append({'member': member, 'identity': identity, 'size': list(image.size),
                'official_split': 'test' if identity in test else 'train',
                'pixels_sha256': pixel_hash, 'sha256': hashlib.sha256(payload).hexdigest()})
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record['pixels_sha256']].append(record)
    excluded, kept = [], []
    conflicts = []
    for pixel_hash, group in by_hash.items():
        if len({r['identity'] for r in group}) > 1:
            conflicts.append([r['member'] for r in group])
            excluded.extend({'member': r['member'], 'reason': 'conflicting_identity_duplicate'} for r in group)
        elif pixel_hash in mpdd_dev:
            excluded.extend({'member': r['member'], 'reason': 'MPDD_development_overlap'} for r in group)
        else:
            kept.append(group[0])
            excluded.extend({'member': r['member'], 'reason': 'within_identity_duplicate'} for r in group[1:])
    counts = Counter(r['identity'] for r in kept)
    excluded.extend({'member': r['member'], 'reason': 'fewer_than_two_unique_photos'} for r in kept if counts[r['identity']] < 2)
    kept = sorted([r for r in kept if counts[r['identity']] >= 2], key=lambda r: r['member'])
    report = {'source': 'https://zenodo.org/records/12578449', 'sha256': hashes,
              'published_test_ids': len(test), 'published_train_ids': len(train),
              'decoded_images': len(records), 'kept_images': len(kept),
              'kept_identities': len({r['identity'] for r in kept}),
              'official_test_images': sum(r['official_split'] == 'test' for r in kept),
              'official_test_ids': len({r['identity'] for r in kept if r['official_split'] == 'test'}),
              'overlap_MPDD_all_exact_pixels': sum(r['pixels_sha256'] in mpdd_all for r in records),
              'conflicting_duplicate_groups': conflicts, 'excluded': excluded, 'records': kept}
    write_json(DEST / 'dogface_audit.json', report)
    print('DogFaceNet audit:', {k: v for k, v in report.items() if k not in ('records', 'excluded', 'conflicting_duplicate_groups', 'sha256')}, flush=True)
    return kept


def encode(records):
    import torch
    from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows

    torch.set_num_threads(4)
    clip = ClipEncoder(device='cuda')
    # Match the revision used to select the original MPDD fusion weight.
    source = load_json(OUT / 'encoders.json')['provenance']
    dino = DinoEncoder(device='cuda', revision=source['revision'], local_files_only=True)
    if sha256(Path.home() / '.cache/clip/ViT-B-32.pt') != source['clip_sha256']:
        raise ValueError('CLIP weights changed')
    provenance = canonical_provenance({'records_hash': digest(records), 'clip_preprocess': str(clip._preprocess),
        'dino_preprocess': dino._processor.to_dict(), 'revision': dino.resolved_revision,
        'clip_sha256': source['clip_sha256'], 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'protocol_sha256': sha256(Path(__file__).with_name('FOLLOWUP_PROTOCOL.md'))})
    signature = digest(provenance)
    path = DEST / 'dogface_features.npz'
    if path.exists():
        with np.load(path) as cache:
            if str(cache['signature']) != signature:
                raise ValueError('Stale DogFaceNet features')
            return {k: cache[k] for k in ('clip', 'dino')}
    rows = {'clip': [], 'dino': []}
    started = time.monotonic()
    with zipfile.ZipFile(DATA / 'DogFaceNet_224resized.zip') as archive:
        for start in range(0, len(records), 64):
            images = []
            for r in records[start:start + 64]:
                with Image.open(io.BytesIO(archive.read(r['member']))) as opened:
                    images.append(opened.convert('RGB'))
            batch = torch.stack([clip._preprocess(image) for image in images]).cuda()
            with torch.inference_mode():
                rows['clip'].append(normalize_rows(clip._model.encode_image(batch).float().cpu().numpy()))
            rows['dino'].append(dino.encode_batch(images))
            if start % 1024 == 0:
                print(f'DogFaceNet encoding {min(start+64,len(records))}/{len(records)}', flush=True)
    result = {k: np.concatenate(v) for k, v in rows.items()}
    with path.with_suffix('.partial').open('wb') as handle:
        np.savez(handle, **result, signature=signature)
    path.with_suffix('.partial').replace(path)
    write_json(DEST / 'encoders.json', {'signature': signature, 'provenance': provenance,
                                      'encoding_seconds': time.monotonic() - started})
    return result


def gallery_draw(records, rng=None):
    by_id = defaultdict(list)
    for i, row in enumerate(records):
        by_id[row['identity']].append(i)
    gallery = []
    for identity in sorted(by_id):
        indices = by_id[identity]
        if rng is None:
            index = min(indices, key=lambda i: hashlib.sha256(f"{SEED}|{records[i]['member']}".encode()).hexdigest())
        else:
            index = int(rng.choice(indices))
        gallery.append(index)
    gallery_set = set(gallery)
    queries = [i for i in range(len(records)) if i not in gallery_set]
    return np.array(queries), np.array(gallery)


def scores_to_identity_metrics(scores, target_positions, query_identity_indices, identity_count):
    import torch

    target = scores[torch.arange(len(scores), device=scores.device), target_positions]
    columns = torch.arange(scores.shape[1], device=scores.device)[None, :]
    ranks = 1 + ((scores > target[:, None]) | ((scores == target[:, None]) & (columns < target_positions[:, None]))).sum(dim=1)
    ranks = ranks.cpu().numpy()
    per_query = np.column_stack([ranks <= 1, ranks <= 5, ranks <= 10, 1. / ranks])
    sums = np.zeros((identity_count, 4))
    np.add.at(sums, query_identity_indices, per_query)
    counts = np.bincount(query_identity_indices, minlength=identity_count)
    if np.any(counts == 0):
        raise ValueError('Identity with no query')
    return sums / counts[:, None], per_query


def evaluate_cohort(name, records, features, repeats):
    import torch

    destination = DEST / f'{name}.json'
    signature = digest({'records': records, 'feature_signature': load_json(OUT / 'encoders.json')['signature'],
        'selection': sha256(OUT / 'frozen_selection.json'), 'protocol': sha256(Path(__file__).with_name('FOLLOWUP_PROTOCOL.md')),
        'code': sha256(Path(__file__)), 'repeats': repeats})
    if destination.exists():
        result = load_json(destination)
        if result['signature'] != signature:
            raise ValueError('Stale follow-up result')
        return result
    ids = sorted({r['identity'] for r in records})
    id_positions = {value: i for i, value in enumerate(ids)}
    clip = torch.tensor(features['clip'], device='cuda')
    dino = torch.tensor(features['dino'], device='cuda')
    rng = np.random.default_rng(SEED)
    weight = load_json(OUT / 'frozen_selection.json')['selected_dino_weight']
    if weight != .75:
        raise ValueError('Prespecified fusion weight changed')
    accumulated = {k: [] for k in ('CLIP', 'DINO', 'fusion')}
    primary = {}
    for draw in range(repeats + 1):
        query, gallery = gallery_draw(records, rng=None if draw == 0 else rng)
        labels = np.array([id_positions[records[i]['identity']] for i in query])
        targets = torch.tensor(labels, device='cuda')
        ci = clip[query] @ clip[gallery].T
        di = dino[query] @ dino[gallery].T
        for method, scores in (('CLIP', ci), ('DINO', di), ('fusion', (1-weight)*ci + weight*di)):
            values, per_query = scores_to_identity_metrics(scores, targets, labels, len(ids))
            if draw == 0:
                primary[method] = values
                np.savez_compressed(DEST / f'{name}_{method}_primary.npz', per_identity=values, per_query=per_query,
                    query_members=np.array([records[i]['member'] for i in query]),
                    gallery_members=np.array([records[i]['member'] for i in gallery]), identities=np.array(ids))
            else:
                accumulated[method].append(values)
        if draw % 10 == 0:
            print(f'{name}: gallery draw {draw}/{repeats}', flush=True)
    def summarize(arrays):
        delta = arrays['fusion'] - arrays['DINO']
        return {'systems': {k: cluster_interval(v, ids) for k, v in arrays.items()},
                'fusion_minus_DINO': cluster_interval(delta, ids),
                'DINO_minus_CLIP': cluster_interval(arrays['DINO'] - arrays['CLIP'], ids),
                'r1_identity_sign_test': exact_sign_test(delta[:, 0]),
                'r1_identity_signflip_p': permutation_p(delta[:, 0])}
    result = {'signature': signature, 'dataset': name, 'identities': len(ids), 'images': len(records),
              'gallery_images_per_draw': len(ids), 'query_images_per_draw': len(query), 'metrics': METRICS,
              'frozen_DINO_weight': weight, 'primary': summarize(primary),
              'gallery_repeat_count': repeats,
              'repeated_gallery_identity_averaged': summarize({k: np.mean(v, axis=0) for k, v in accumulated.items()}),
              'repeat_mean_distributions': {k: {'mean': np.mean(v, axis=(0,1)).tolist(),
                 'min': np.asarray(v).mean(axis=1).min(axis=0).tolist(),
                 'max': np.asarray(v).mean(axis=1).max(axis=0).tolist()} for k,v in accumulated.items()}}
    np.savez_compressed(DEST / f'{name}_repeats.npz', **{k: np.stack(v) for k,v in accumulated.items()}, identities=np.array(ids))
    write_json(destination, result)
    print(name, 'primary', {k: v['mean'] for k,v in result['primary']['systems'].items()}, flush=True)
    return result


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    # Original same-ID paired evidence, before expanding the query/gallery protocol.
    with np.load(OUT / 'clean_per_query.npz') as original:
        a, b = original['DINO_per_query'][:,0], original['selected_fusion_per_query'][:,0]
        delta = original['selected_fusion'][:,0] - original['DINO'][:,0]
        paired = {'both_correct': int(((a==1)&(b==1)).sum()), 'both_wrong': int(((a==0)&(b==0)).sum()),
                  'fusion_only_correct': int(((a==0)&(b==1)).sum()), 'DINO_only_correct': int(((a==1)&(b==0)).sum()),
                  'changed_queries': original['query_members'][a != b].tolist(),
                  'identity_sign_test': exact_sign_test(delta), 'identity_signflip_p': permutation_p(delta)}
    write_json(DEST / 'original_mpdd_paired.json', paired)
    dog_records = prepare_dogface()
    dog_features = encode(dog_records)
    results = []
    positions = [i for i,r in enumerate(dog_records) if r['official_split'] == 'test']
    results.append(evaluate_cohort('dogface_official_test', [dog_records[i] for i in positions],
                                  {k: v[positions] for k,v in dog_features.items()}, 20))
    results.append(evaluate_cohort('dogface_all_frozen', dog_records, dog_features, 20))
    records = load_json(OUT / 'data_audit.json')['records']
    positions = [i for i,r in enumerate(records) if r['split'] in ('query', 'gallery')]
    with np.load(OUT / 'features_clean.npz') as features:
        results.append(evaluate_cohort('mpdd_gallery_sensitivity', [records[i] for i in positions],
                                      {k: features[k][positions] for k in ('clip','dino')}, 100))
    lines = ['# Frozen fusion: follow-up validation', '',
             'Weight fixed from the original MPDD validation: 25% CLIP + 75% DINO. No training or new tuning.',
             'Original MPDD: fusion-only correct = '+str(paired['fusion_only_correct'])+', DINO-only correct = '+str(paired['DINO_only_correct'])+'.',
             f"Original exact paired ID sign-test p={paired['identity_sign_test']['two_sided_exact_p']:.4f}.", '',
             'DogFaceNet official-test cohort is the primary external test; all-IDs is secondary.',
             'All use one gallery photo per identity and all remaining photos as queries. These are custom retrieval protocols.',
             'MPDD resampling reuses the same identities and is a sensitivity analysis, not new external evidence.', '',
             '| Cohort | IDs | Images | Gallery draws | CLIP R@1 | DINO R@1 | Fusion R@1 | Fusion−DINO [95% CI], pp |',
             '|---|---:|---:|---|---:|---:|---:|---|']
    for result in results:
        for label, key in [('hash-fixed', 'primary'), (f"{result['gallery_repeat_count']} repeats", 'repeated_gallery_identity_averaged')]:
            summary = result[key]
            means = [summary['systems'][k]['mean'][0]*100 for k in ('CLIP','DINO','fusion')]
            diff = summary['fusion_minus_DINO']
            lines.append(f"| {result['dataset']} | {result['identities']} | {result['images']} | {label} | " +
                         ' | '.join(f'{v:.2f}' for v in means) + f" | {diff['mean'][0]*100:+.2f} [{diff['ci95_low'][0]*100:+.2f}, {diff['ci95_high'][0]*100:+.2f}] |")
    lines.extend(['', '## Paired R@1 tests', '', '| Cohort / analysis | Winning / tied / losing IDs | Sign-flip p |', '|---|---|---:|'])
    for result in results:
        for key in ('primary', 'repeated_gallery_identity_averaged'):
            s = result[key]['r1_identity_sign_test']
            lines.append(f"| {result['dataset']} / {key} | {s['wins']} / {s['ties']} / {s['losses']} | {result[key]['r1_identity_signflip_p']:.5f} |")
    lines.extend(['', 'Bootstrap: 2,000 identity resamples after averaging repeats within ID. Repeats are not independent sample-size increases.',
                  'Sign-flip: paired identity-level mean differences, 100,000 random sign draws, +1 correction.',
                  'Only hash-fixed official-test R@1 is the primary hypothesis; remaining comparisons are exploratory without multiplicity correction.',
                  'Exact duplicate checks do not eliminate same-session near duplicates, foundation pretraining overlap, or unknown identity overlap across corpora.',
                  'DogFaceNet faces are pre-aligned and 224px; results do not establish full-body lost-dog deployment performance or text/Flow efficacy.',
                  'Data source: [Mougeot, DogFaceNet datasets](https://zenodo.org/records/12578449). No raw images redistributed.',
                  'All counts, source hashes, per-ID/per-query metrics and repeats are in artifacts/followup/.', ''])
    Path(__file__).with_name('FOLLOWUP_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
