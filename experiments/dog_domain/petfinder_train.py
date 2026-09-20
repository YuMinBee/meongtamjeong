"""English attribute alignment on PetFinder; frozen encoders, cached features."""
from pathlib import Path
import csv
import io
import json
import hashlib
import time
from collections import defaultdict, Counter
from zipfile import ZipFile
import numpy as np
from PIL import Image

OUT = Path('D:/meongtamjeong_research/petfinder/english_v1')
ARCHIVE = OUT.parent / 'petfinder-adoption-prediction.zip'
REV = '5931719e67bbdb9737e363e781fb0c67687896bc'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def prepare():
    records, owners = [], defaultdict(set)
    with ZipFile(ARCHIVE) as z:
        photos = defaultdict(list)
        for name in z.namelist():
            if name.endswith('.jpg') and '_images/' in name:
                photos[Path(name).stem.rsplit('-', 1)[0]].append(name)
        colors = {str(i): c for i, c in enumerate(['', 'black', 'brown', 'golden', 'yellow', 'cream', 'gray', 'white'])}
        sizes = {'1': 'small', '2': 'medium', '3': 'large', '4': 'extra-large'}
        for split in ('train', 'test'):
            rows = csv.DictReader(io.StringIO(z.read(f'{split}/{split}.csv').decode('utf-8-sig')))
            for r in rows:
                pid = r['PetID']
                if r['Type'] != '1' or r['Quantity'] != '1' or not photos[pid] or not r['Description'].strip():
                    continue
                color = [colors[r[k]] for k in ('Color1', 'Color2', 'Color3') if r[k] in colors and colors[r[k]]]
                if not color or r['MaturitySize'] not in sizes:
                    continue
                names = sorted(photos[pid], key=lambda n: int(Path(n).stem.rsplit('-', 1)[1]))
                for name in names:
                    owners['bytes:' + hashlib.sha256(z.read(name)).hexdigest()].add(pid)
                try:
                    im = Image.open(io.BytesIO(z.read(names[0]))).convert('RGB')
                    pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                except Exception:
                    continue
                owners['pixels:' + pixel].add(pid)
                records.append(dict(pet_id=pid, original_split=split, rescuer=r['RescuerID'],
                                    image=names[0], photos=names, colors=sorted(set(color)),
                                    size=sizes[r['MaturitySize']], description_present=True))
    ambiguous = set().union(*(ids for ids in owners.values() if len(ids) > 1))
    records = [r for r in records if r['pet_id'] not in ambiguous]
    tr = [i for i, r in enumerate(records) if r['original_split'] == 'train']
    groups = [records[i]['rescuer'] or records[i]['pet_id'] for i in tr]
    unique_groups = sorted(set(groups), key=lambda g: hashlib.sha256(('20260917|' + g).encode()).hexdigest())
    validation_groups = set(unique_groups[:max(1, round(len(unique_groups)*.15))])
    val = {i for i,g in zip(tr, groups) if g in validation_groups}
    train = set(tr) - val
    for i, r in enumerate(records):
        r['split'] = 'train' if i in train else 'validation' if i in val else 'test'
        r['signature'] = r['size'] + '|' + ','.join(r['colors'])
    assert not ({r['rescuer'] for r in records if r['split'] == 'train'} & {r['rescuer'] for r in records if r['split'] == 'validation'})
    write('records.json', records)
    write('preparation.json', dict(counts=dict(Counter(r['split'] for r in records)),
          excluded_duplicate_pet_ids=len(ambiguous),
          duplicate_check='SHA256 of every eligible photo; decoded RGB hash of representative photos',
          test_rescuers_shared_with_train=len({r['rescuer'] for r in records if r['split']=='test'} & {r['rescuer'] for r in records if r['split']=='train'}),
          limitations=['Near-duplicate images and re-registered animals are not fully audited.',
                       'Official test partition preserved; rescuers may overlap train and test.']))
    print('Prepared:', Counter(r['split'] for r in records), flush=True)
    return records


def features(records):
    import torch
    from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows
    fingerprint = hashlib.sha256((OUT / 'records.json').read_bytes()).hexdigest()
    cache = OUT / 'features.npz'
    if cache.exists():
        arrays = np.load(cache)
        assert str(arrays['records_sha256']) == fingerprint
        return arrays
    clip = ClipEncoder(device='cuda')
    dino = DinoEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m', device='cuda', revision=REV, local_files_only=True)
    cf, df = [], []
    start = time.perf_counter()
    # Test features are deliberately not encoded during training/model selection.
    active = [r for r in records if r['split'] != 'test']
    with ZipFile(ARCHIVE) as z:
        for offset in range(0, len(active), 64):
            images = [Image.open(io.BytesIO(z.read(r['image']))).convert('RGB') for r in active[offset:offset+64]]
            with torch.inference_mode():
                inputs = torch.stack([clip._preprocess(im) for im in images]).to('cuda')
                cf.append(normalize_rows(clip._model.encode_image(inputs).float().cpu().numpy()))
            df.append(dino.encode_batch(images))
            done = min(offset + 64, len(active))
            if offset == 0 or done % 512 == 0 or done == len(active):
                elapsed = time.perf_counter() - start
                print(f'Features {done}/{len(active)}; {elapsed:.1f}s elapsed; ETA {(len(active)-done)*elapsed/done:.1f}s', flush=True)
    signatures = sorted({r['signature'] for r in active})
    examples = {r['signature']: r for r in active}
    prompts = []
    for sig in signatures:
        r = examples[sig]
        color = ' and '.join(r['colors'])
        prompts.extend([f"a {r['size']} {color} dog", f"a photo of a {r['size']} dog with {color} fur",
                        f"a {color} dog of {r['size']} size", f"Find a {r['size']} dog whose coat is {color}."])
    text = np.concatenate([clip.encode_text_batch(prompts[i:i+64]) for i in range(0, len(prompts), 64)])
    np.savez_compressed(cache, clip=np.concatenate(cf), dino=np.concatenate(df), text=text,
                        signatures=np.array(signatures), records_sha256=np.array(fingerprint))
    write('features_manifest.json', dict(records_sha256=fingerprint, clip='ViT-B/32', dino_revision=REV,
                                        seconds=time.perf_counter()-start, prompts=prompts, images=len(active)))
    del clip, dino
    torch.cuda.empty_cache()
    return np.load(cache)


def train(records, arrays):
    import torch
    from experiments.dino_fusion.alignment import make_projection_head, project_embeddings
    from experiments.dino_fusion.train_alignment import multipositive_loss, save_checkpoint_atomic
    active = [r for r in records if r['split'] != 'test']
    signatures = list(arrays['signatures'])
    labels = torch.tensor([signatures.index(r['signature']) for r in active], device='cuda')
    tr = torch.tensor([i for i,r in enumerate(active) if r['split']=='train'], device='cuda')
    # Validation-only signatures have no training centroid: omit centroid regularization for validation.
    va = torch.tensor([i for i,r in enumerate(active) if r['split']=='validation'], device='cuda')
    texts = torch.tensor(arrays['text'], device='cuda')
    report = dict(dataset=dict(Counter(r['split'] for r in records)), runs=[],
                  validation='held-out rescuers and fourth English template; multipositive loss, no centroid penalty',
                  max_epochs=150, batch_size=512, learning_rate=.002, patience_checks=6, validation_every=5,
                  seeds=[17,29,43], encoders_frozen=True, training_language='English', test_evaluated=False)
    # A matched CLIP-space Linear adapter separates domain-training benefit from DINO benefit.
    for target, architecture in [('clip','linear'), ('dino','linear'), ('dino','mlp'), ('dino','flow')]:
        visual = torch.tensor(arrays[target], device='cuda')
        centroids = torch.zeros((len(signatures), visual.shape[1]), device='cuda')
        for i in range(len(signatures)):
            idx = tr[labels[tr] == i]
            if len(idx):
                centroids[i] = visual[idx].mean(0)
        centroids = torch.nn.functional.normalize(centroids, dim=-1)
        for seed in report['seeds']:
            torch.manual_seed(seed)
            head = make_projection_head(architecture, input_dim=512, output_dim=visual.shape[1]).cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.0001)
            best, stale, best_state, history = float('inf'), 0, None, []
            started = time.perf_counter()
            for epoch in range(1, 151):
                head.train()
                permutation = tr[torch.randperm(len(tr), device='cuda')]
                for offset in range(0, len(tr), 512):
                    idx = permutation[offset:offset+512]
                    prompt_ids = labels[idx]*4 + torch.randint(3, (len(idx),), device='cuda')
                    mapped = project_embeddings(head, texts[prompt_ids])
                    loss = multipositive_loss(torch, mapped, visual[idx], labels[idx], labels[idx], centroids,
                                             temperature=.07, centroid_weight=.25, symmetric=True)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                if epoch % 5:
                    continue
                head.eval()
                with torch.inference_mode():
                    mapped = project_embeddings(head, texts[labels[va]*4+3])
                    value = float(multipositive_loss(torch, mapped, visual[va], labels[va], labels[va], centroids,
                                                    temperature=.07, centroid_weight=0., symmetric=False))
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite validation loss')
                history.append(dict(epoch=epoch, validation_loss=value))
                if value < best - 1e-5:
                    best, stale, best_epoch = value, 0, epoch
                    best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                else:
                    stale += 1
                if epoch % 25 == 0:
                    print(f'{target}/{architecture}/{seed}: epoch {epoch}, val {value:.4f}, {time.perf_counter()-started:.1f}s', flush=True)
                if stale >= 6:
                    break
            assert best_state is not None
            checkpoint = OUT / f'{target}_{architecture}_seed{seed}.pt'
            save_checkpoint_atomic(dict(architecture=architecture, input_dim=512, output_dim=visual.shape[1],
                                         state_dict=best_state, seed=seed, target=target), checkpoint, torch)
            report['runs'].append(dict(target=target, architecture=architecture, seed=seed, best_epoch=best_epoch,
                                       epochs_completed=epoch, validation_loss=best, history=history,
                                       seconds=time.perf_counter()-started, checkpoint=str(checkpoint)))
            write('training_report.json', report)
            print('Finished', target, architecture, seed, round(time.perf_counter()-started, 1), 'seconds', flush=True)
    report['complete'] = True
    write('training_report.json', report)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA required for this run'
    started = time.perf_counter()
    records = prepare()
    train(records, features(records))
    print(f'COMPLETE in {time.perf_counter()-started:.1f}s; {OUT}', flush=True)
