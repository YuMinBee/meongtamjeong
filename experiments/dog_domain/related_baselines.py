"""dino.txt and Talk2DINO baselines under the fixed image-and-attribute protocol.

See RELATED_BASELINES_PROTOCOL.md. Stages: weights -> features -> train -> evaluate.
Existing caches, checkpoints and paper results are read only; outputs go to OUT.
"""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path, PureWindowsPath
from zipfile import ZipFile
import numpy as np
import torch
from PIL import Image
from experiments.dog_domain.petfinder_train import OUT as V1, ARCHIVE
from experiments.dog_domain.petfinder_expanded import OUT as V2, pair_loss
from experiments.dog_domain.notice_extension import OUT as KR
from experiments.dog_domain import taiwan_eval
from experiments.dog_domain import petfinder_core_eval as core
from experiments.dino_fusion.alignment import evaluation_prompts, make_projection_head, project_embeddings, projection_head_from_checkpoint
from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows
from experiments.dino_fusion.train_alignment import multipositive_loss, save_checkpoint_atomic
from experiments.composed_retrieval.metrics import cluster_interval

# On Linux hosts, a 'D:' symlink in the repo root points at the research folder (existing modules hardcode it).
RESEARCH = Path('D:/meongtamjeong_research')
OUT = RESEARCH / 'related_baselines_20260928'
DINOV3_REPO = RESEARCH / 'tools/dinov3'
PYDEPS = RESEARCH / 'tools/pydeps'
DINOTXT = RESEARCH / 'models/dinotxt'
DINOTXT_HEAD = DINOTXT / 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'
DINOTXT_BACKBONE = DINOTXT / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOTXT_BPE = DINOTXT / 'bpe_simple_vocab_16e6.txt.gz'
T2D = RESEARCH / 'models/talk2dino/model.safetensors'
DINOL = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
PROTOCOL = Path(__file__).with_name('RELATED_BASELINES_PROTOCOL.md')
RESULTS = Path(__file__).with_name('RELATED_BASELINES_RESULTS.md')
SEEDS = [17, 29, 43]
DATASETS = ['PetFinder', 'Korea', 'Taiwan']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def environment():
    return dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, platform=sys.platform)


def weights():
    """Mirror downloads must match the hash prefix in Meta's official filenames."""
    found = {}
    for path, prefix in [(DINOTXT_HEAD, 'a442d8f5'), (DINOTXT_BACKBONE, '8aa4cbdd')]:
        found[path.name] = digest(path)
        assert found[path.name].startswith(prefix), (path, found[path.name])
    found[DINOTXT_BPE.name] = digest(DINOTXT_BPE)
    found[T2D.name] = digest(T2D)
    found['dinov3_repo_commit'] = (DINOV3_REPO / '.git/HEAD').read_text().strip()
    found['environment'] = environment()
    write('weights.json', found)
    print('Weights verified', flush=True)


# ---------------------------------------------------------------- data

def views():
    """Image sources, prompts and stored-text keys in the exact order of the existing caches."""
    v1 = read(V1 / 'records.json')
    pf_test = [r for r in v1 if r['split'] == 'test']
    pf_train = [r for r in v1 if r['split'] != 'test']
    kr = read(KR / 'records.json')
    tw = read(taiwan_eval.OUT / 'records.json')
    return {
        'PetFinder_train': [('zip', r['image']) for r in pf_train],
        'PetFinder_image': [('zip', r['image']) for r in pf_test],
        'Korea_query': [('file', KR / 'images' / r['notice_id'] / 'query.png') for r in kr],
        'Korea_gallery': [('file', KR / 'images' / r['notice_id'] / 'gallery.png') for r in kr],
        'Taiwan_image': [('file', taiwan_eval.DATA.joinpath(*PureWindowsPath(r['image_path']).parts)) for r in tw],
    }, {
        'PetFinder': [f"Find a {r['size']} dog whose coat is {' and '.join(r['colors'])}." for r in pf_test],
        'Korea': [evaluation_prompts(r['attributes'])['english'] for r in kr],
        'Taiwan': [taiwan_eval.prompt(r, 'english') for r in tw],
    }


def load_images(items, archive):
    images = []
    for kind, source in items:
        if kind == 'zip':
            images.append(Image.open(io.BytesIO(archive.read(source))).convert('RGB'))
        else:
            with Image.open(source) as image:
                images.append(image.convert('RGB'))
    return images


# ---------------------------------------------------------------- encoders

class DinoTxt:
    """Official dino.txt, loaded from verified local files with strict key checks."""

    def __init__(self):
        for p in (PYDEPS, DINOV3_REPO):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
        from dinov3.data.transforms import make_classification_eval_transform
        from dinov3.eval.text.tokenizer import Tokenizer
        model, _ = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=False, bpe_path_or_url=DINOTXT_BPE.resolve().as_uri())
        backbone = torch.load(DINOTXT_BACKBONE, map_location='cpu', weights_only=True)
        model.visual_model.backbone.load_state_dict(backbone, strict=True)
        state = torch.load(DINOTXT_HEAD, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected[:5]
        assert all(k.startswith('visual_model.backbone.') for k in missing), [k for k in missing if not k.startswith('visual_model.backbone.')][:5]
        self.model = model.cuda().eval()
        self.transform = make_classification_eval_transform()
        self.tokenizer = Tokenizer(vocab_path=io.BytesIO(DINOTXT_BPE.read_bytes()))
        self.meta = dict(missing_backbone_keys_loaded_separately=len(missing), transform='resize 256, center crop 224, bicubic, ImageNet norm')

    @torch.inference_mode()
    def images(self, images):
        x = torch.stack([self.transform(im) for im in images]).cuda()
        return normalize_rows(self.model.encode_image(x).float().cpu().numpy())

    @torch.inference_mode()
    def texts(self, texts):
        return normalize_rows(self.model.encode_text(self.tokenizer.tokenize(texts).cuda()).float().cpu().numpy())


class Talk2Dino:
    """Released Talk2DINO-ViTB: DINOv2-reg + CLIP ViT-B/16 + projection, all from one safetensors file."""

    def __init__(self):
        from safetensors.torch import load_file
        import clip
        import torchvision.transforms as T
        state = load_file(str(T2D))
        part = lambda prefix: {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=False, trust_repo=True)
        self.dino.load_state_dict(part('model.'), strict=True)
        self.dino = self.dino.cuda().eval()
        self.clip = clip.model.build_model(part('clip_model.')).cuda().eval()
        p = part('proj.')
        self.w1, self.b1 = p['linear_layer.weight'].cuda(), p['linear_layer.bias'].cuda()
        self.w2, self.b2 = p['hidden_layers.0.weight'].cuda(), p['hidden_layers.0.bias'].cuda()
        self.transform = T.Compose([T.Resize((518, 518)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.feats = {}
        self.dino.blocks[-1].attn.qkv.register_forward_hook(lambda m, i, o: self.feats.__setitem__('qkv', o))
        self.heads, self.globals, self.scale = self.dino.num_heads, 5, 0.125
        self.tokenize = clip.tokenize

    @torch.inference_mode()
    def images(self, images):
        x = torch.stack([self.transform(im) for im in images]).cuda()
        out = self.dino.forward_features(x)
        patches = out['x_norm_patchtokens']
        b, n, c = patches.shape
        qkv = self.feats['qkv'].reshape(b, n + self.globals, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        # Only the CLS query row is needed: attn[:, :, 0, globals:].
        maps = ((qkv[0][:, :, :1] * self.scale) @ qkv[1].transpose(-2, -1))[:, :, 0, self.globals:]
        average = maps.mean(1).softmax(-1)
        pooled = (average.unsqueeze(-1) * patches).mean(1)
        per_head = torch.einsum('bhp,bpc->bhc', maps.softmax(-1), patches) / n
        return dict(avg=normalize_rows(pooled.float().cpu().numpy()),
                    heads=torch.nn.functional.normalize(per_head.float(), dim=-1).half().cpu().numpy(),
                    cls=normalize_rows(out['x_norm_clstoken'].float().cpu().numpy()))

    @torch.inference_mode()
    def texts(self, texts):
        t = self.clip.encode_text(self.tokenize(texts).cuda()).float()
        t = torch.tanh(t @ self.w1.T + self.b1) @ self.w2.T + self.b2
        return normalize_rows(t.cpu().numpy())


class DinoLarge:
    def __init__(self):
        self.encoder = DinoEncoder(DINOL, device='cuda', local_files_only=True)
        self.meta = dict(model=DINOL, revision=self.encoder.resolved_revision)

    def images(self, images):
        return self.encoder.encode_batch(images)


def features():
    """Encode every view once per encoder; each cache is resumable and written atomically."""
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    image_views, prompts = views()
    manifest = {}
    for name, build, views_needed in [
        ('dinoL', DinoLarge, list(image_views)),
        ('dinotxt', DinoTxt, [v for v in image_views if v != 'PetFinder_train']),
        ('t2d', Talk2Dino, [v for v in image_views if v != 'PetFinder_train']),
    ]:
        cache = OUT / f'features_{name}.npz'
        if cache.exists():
            manifest[name] = read(OUT / f'features_{name}.json')
            continue
        started = time.perf_counter()
        encoder = build()
        arrays = {}
        with ZipFile(ARCHIVE) as archive:
            for view in views_needed:
                chunks = []
                items = image_views[view]
                for start in range(0, len(items), 32):
                    chunks.append(encoder.images(load_images(items[start:start + 32], archive)))
                    if start % 1600 == 0:
                        print(name, view, min(start + 32, len(items)), '/', len(items), flush=True)
                if isinstance(chunks[0], dict):
                    for key in chunks[0]:
                        arrays[f'{view}_{key}'] = np.concatenate([c[key] for c in chunks])
                else:
                    arrays[view] = np.concatenate(chunks)
        if hasattr(encoder, 'texts'):
            for dataset, texts in prompts.items():
                arrays[f'{dataset}_text'] = np.concatenate([encoder.texts(texts[i:i + 64]) for i in range(0, len(texts), 64)])
        tmp = cache.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        tmp.replace(cache)
        manifest[name] = dict(seconds=time.perf_counter() - started, sha256=digest(cache), environment=environment(),
                              shapes={k: list(v.shape) for k, v in arrays.items()}, **getattr(encoder, 'meta', {}))
        write(f'features_{name}.json', manifest[name])
        print('Encoded', name, round(manifest[name]['seconds'], 1), 's', flush=True)
        del encoder
        torch.cuda.empty_cache()
    # Stored CLIP ViT-B/32 prompt features must be reproducible from the prompts above.
    clip = ClipEncoder(device='cuda')
    stored = {'PetFinder': np.load(V1 / 'core_evaluation/petfinder_features.npz')['text'],
              'Korea': np.load(KR / 'features.npz')['text_english'],
              'Taiwan': np.load(taiwan_eval.OUT / 'text_features.npz')['template_english']}
    check = {}
    for dataset, texts in prompts.items():
        fresh = np.concatenate([clip.encode_text_batch(texts[i:i + 64]) for i in range(0, len(texts), 64)])
        check[dataset] = float(np.abs(fresh - stored[dataset]).max())
        assert check[dataset] < 2e-3, (dataset, check[dataset])
    write('prompt_check.json', dict(max_abs_difference_vs_stored_clip_text=check))
    print('Prompt reproduction check passed', check, flush=True)


# ---------------------------------------------------------------- training

def t2d_head(input_dim, output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim), torch.nn.Tanh(), torch.nn.Linear(output_dim, output_dim))


def build_head(architecture, output_dim):
    if architecture == 't2d':
        return t2d_head(512, output_dim)
    return make_projection_head(architecture, input_dim=512, output_dim=output_dim)


def fit(visual_np, architecture, seed, target):
    """Same objective, schedule, split and early stopping as petfinder_expanded.train."""
    records = read(V2 / 'records.json')
    old = np.load(V1 / 'features.npz')
    profiles = np.load(V2 / 'profiles.npz')
    active = [r for r in records if r['split'] != 'test']
    idx = [i for i, r in enumerate(records) if r['split'] != 'test']
    assert list(profiles['ids'][idx]) == [r['pet_id'] for r in active] and len(visual_np) == len(active)
    texts = torch.tensor(old['text'], device='cuda')
    full = torch.tensor(np.stack([profiles[k][idx] for k in ['profile', 'structured', 'description']], axis=1), device='cuda')
    signatures = list(old['signatures'])
    labels = torch.tensor([signatures.index(r['signature']) for r in active], device='cuda')
    tr = torch.tensor([i for i, r in enumerate(active) if r['split'] == 'train'], device='cuda')
    va = torch.tensor([i for i, r in enumerate(active) if r['split'] == 'validation'], device='cuda')
    visual = torch.tensor(visual_np, device='cuda')
    centroids = torch.zeros((len(signatures), visual.shape[1]), device='cuda')
    for i in range(len(signatures)):
        matches = tr[labels[tr] == i]
        if len(matches):
            centroids[i] = visual[matches].mean(0)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    torch.manual_seed(seed)
    head = build_head(architecture, visual.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.0001)
    best, stale, state, history, best_epoch = float('inf'), 0, None, [], None
    start = time.perf_counter()
    for epoch in range(1, 151):
        head.train()
        perm = tr[torch.randperm(len(tr), device='cuda')]
        for offset in range(0, len(tr), 512):
            b = perm[offset:offset + 512]
            attr = project_embeddings(head, texts[labels[b] * 4 + torch.randint(3, (len(b),), device='cuda')])
            variant = torch.randint(3, (len(b),), device='cuda')
            mapped = project_embeddings(head, full[b, variant])
            loss = .5 * multipositive_loss(torch, attr, visual[b], labels[b], labels[b], centroids, temperature=.07, centroid_weight=.25, symmetric=True) + .5 * pair_loss(mapped, visual[b])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 5:
            continue
        head.eval()
        with torch.inference_mode():
            attr = project_embeddings(head, texts[labels[va] * 4 + 3])
            mapped = project_embeddings(head, full[va, 0])
            value = float(.5 * multipositive_loss(torch, attr, visual[va], labels[va], labels[va], centroids, temperature=.07, centroid_weight=0., symmetric=False) + .5 * pair_loss(mapped, visual[va]))
        assert np.isfinite(value)
        history.append(dict(epoch=epoch, validation_loss=value))
        if value < best - 1e-5:
            best, stale, best_epoch = value, 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= 6:
            break
    assert state is not None
    (OUT / 'heads').mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / 'heads' / f'{target}_{architecture}_seed{seed}.pt'
    save_checkpoint_atomic(dict(architecture=architecture, input_dim=512, output_dim=visual.shape[1], state_dict=state, seed=seed, target=target), checkpoint, torch)
    return dict(target=target, architecture=architecture, seed=seed, best_epoch=best_epoch, epochs_completed=epoch,
                validation_loss=best, seconds=time.perf_counter() - start, checkpoint=str(checkpoint), history=history)


def train():
    torch.set_num_threads(4)
    # Replication check: the copied loop must reproduce an existing v2 run before new targets are trained.
    reference = next(r for r in read(V2 / 'training_report.json')['runs'] if r['target'] == 'dino' and r['architecture'] == 'linear' and r['seed'] == 17)
    replica = fit(np.load(V1 / 'features.npz')['dino'], 'linear', 17, 'replica_dinoB')
    gap = abs(replica['validation_loss'] - reference['validation_loss'])
    print('Replication validation loss', replica['validation_loss'], 'reference', reference['validation_loss'], flush=True)
    # GPU model/kernels may differ from the original run: require a near-identical loss, record the epoch.
    assert gap < 2e-2, (replica['validation_loss'], reference['validation_loss'])
    report = dict(environment=environment(), replication=dict(reference=reference['validation_loss'], replica=replica['validation_loss'], absolute_gap=gap,
                                                             best_epoch=replica['best_epoch'], reference_best_epoch=reference['best_epoch']),
                  recipe='petfinder_expanded v2: 0.5 attribute multipositive + 0.5 full-notice instance contrast; same split, lr, batch, patience',
                  runs=[])
    dino_large = np.load(OUT / 'features_dinoL.npz')['PetFinder_train']
    dino_base = np.load(V1 / 'features.npz')['dino']
    for target, visual, architecture in [('dinoL', dino_large, 'linear'), ('dinoL', dino_large, 'mlp'), ('t2darch', dino_base, 't2d')]:
        for seed in SEEDS:
            run = fit(visual, architecture, seed, target)
            report['runs'].append(run)
            write('training_report.json', report)
            print('FINISHED', target, architecture, seed, round(run['seconds'], 1), 's', flush=True)
    report['complete'] = True
    write('training_report.json', report)


# ---------------------------------------------------------------- evaluation

def evaluate_dataset(name, data, methods):
    """nDCG@10/P@10 exactly as petfinder_core_eval.evaluate; methods map to (query, gallery) or a score callable."""
    n = len(data['ids'])
    if 'relevance' in data:
        # Caller-defined (possibly asymmetric) relevance, e.g. human colour labels in gold_labels.analyze.
        rel = np.array(data['relevance'], dtype=bool)
    else:
        attrs = data['attrs']
        rel = np.array([[a[1] == b[1] and (set(a[0]) == set(b[0]) if data['exact'] else bool(set(a[0]) & set(b[0]))) for b in attrs] for a in attrs])
    np.fill_diagonal(rel, False)
    known = rel.sum(1) > 0
    relevance = torch.tensor(rel, device='cuda')
    discount = 1 / torch.log2(torch.arange(2, 12, device='cuda', dtype=torch.float32))
    ideal = torch.tensor([float(discount[:min(10, int(k))].sum()) for k in rel.sum(1)], device='cuda')
    metrics, identity = {}, {}
    for method, spec in methods.items():
        values, identities = [], []
        if not callable(spec):
            # CPU float32 matmul, as in petfinder_core_eval, so near-tied rankings match the stored results.
            q, g = (np.asarray(x, dtype=np.float32) for x in spec)
        for start in range(0, n, 256):
            end = min(n, start + 256)
            s = spec(start, end) if callable(spec) else torch.tensor(q[start:end] @ g.T, device='cuda')
            rows = torch.arange(end - start, device='cuda')
            cols = torch.arange(start, end, device='cuda')
            if data['identity']:
                order = torch.argsort(s, descending=True, stable=True)
                rank = (order == cols[:, None]).int().argmax(1) + 1
                identities.append(torch.stack([(rank <= 1).float(), (rank <= 5).float(), (rank <= 10).float(), 1 / rank.float()], dim=1).cpu().numpy())
            s[rows, cols] = -torch.inf
            top = torch.argsort(s, descending=True, stable=True)[:, :10]
            matches = relevance[start:end].gather(1, top).float()
            ndcg = (matches * discount).sum(1) / ideal[start:end].clamp_min(1e-8)
            values.append(torch.stack([ndcg, matches.mean(1)], dim=1).cpu().numpy())
        metrics[method] = np.concatenate(values)
        if identities:
            identity[method] = np.concatenate(identities)
    return metrics, identity, known


def mix(image, text, w=.8):
    return normalize_rows(w * normalize_rows(image) + (1 - w) * normalize_rows(text))


def local_path(path):
    """Checkpoint paths recorded on Windows resolve to the repo-relative 'D:' folder on Linux."""
    if Path(path).exists():
        return Path(path)
    windows = PureWindowsPath(path)
    return Path(windows.drive, *windows.parts[1:])


def map_text(checkpoint, text):
    ck = torch.load(local_path(checkpoint), map_location='cpu', weights_only=True)
    head = (t2d_head(ck['input_dim'], ck['output_dim']) if ck['architecture'] == 't2d' else projection_head_from_checkpoint(ck)).cuda().eval()
    head.load_state_dict(ck['state_dict'])
    with torch.inference_mode():
        return project_embeddings(head, torch.tensor(text, device='cuda')).cpu().numpy()


def evaluate():
    torch.set_num_threads(4)
    base = {'PetFinder': core.petfinder(), **core.external()}
    f_l, f_x, f_t = (np.load(OUT / f'features_{k}.npz') for k in ['dinoL', 'dinotxt', 't2d'])
    v2_runs = read(V2 / 'training_report.json')['runs']
    new_runs = read(OUT / 'training_report.json')
    assert new_runs.get('complete')
    keys = {'PetFinder': ('PetFinder_image', 'PetFinder_image'), 'Korea': ('Korea_query', 'Korea_gallery'), 'Taiwan': ('Taiwan_image', 'Taiwan_image')}
    results = {}
    for name, data in base.items():
        qk, gk = keys[name]
        text = data['text']
        methods = {'CLIP_mix': (mix(data['cq'], text), data['cg']), 'DINO_image': (data['dq'], data['dg'])}
        seeded = {}

        def add_seeded(prefix, runs, query_images, gallery_images):
            for run in runs:
                mapped = map_text(run['checkpoint'], text)
                seeded.setdefault(f'{prefix}_mix', []).append(f'{prefix}_mix_seed{run["seed"]}')
                seeded.setdefault(f'{prefix}_text_only', []).append(f'{prefix}_text_only_seed{run["seed"]}')
                methods[f'{prefix}_mix_seed{run["seed"]}'] = (mix(query_images, mapped), gallery_images)
                methods[f'{prefix}_text_only_seed{run["seed"]}'] = (mapped, gallery_images)

        add_seeded('dino_linear', [r for r in v2_runs if r['target'] == 'dino' and r['architecture'] == 'linear'], data['dq'], data['dg'])
        methods['DINOL_image'] = (f_l[qk], f_l[gk])
        add_seeded('dinoL_linear', [r for r in new_runs['runs'] if r['target'] == 'dinoL' and r['architecture'] == 'linear'], f_l[qk], f_l[gk])
        add_seeded('dinoL_mlp', [r for r in new_runs['runs'] if r['target'] == 'dinoL' and r['architecture'] == 'mlp'], f_l[qk], f_l[gk])
        add_seeded('t2darch', [r for r in new_runs['runs'] if r['target'] == 't2darch'], data['dq'], data['dg'])
        xt = f_x[f'{name}_text']
        methods.update(dinotxt_image=(f_x[qk], f_x[gk]), dinotxt_text_only=(xt, f_x[gk]), dinotxt_mix=(mix(f_x[qk], xt), f_x[gk]))
        tt = f_t[f'{name}_text']
        methods.update(t2d_image=(f_t[f'{qk}_avg'], f_t[f'{gk}_avg']), t2d_cls_image=(f_t[f'{qk}_cls'], f_t[f'{gk}_cls']),
                       t2d_text_only=(tt, f_t[f'{gk}_avg']), t2d_mix=(mix(f_t[f'{qk}_avg'], tt), f_t[f'{gk}_avg']))
        heads = torch.tensor(f_t[f'{gk}_heads'].astype(np.float32), device='cuda')
        tt_gpu = torch.tensor(tt, device='cuda')
        methods['t2d_text_maxhead'] = lambda s, e: torch.einsum('qc,nhc->qnh', tt_gpu[s:e], heads).amax(-1)
        metrics, identity, known = evaluate_dataset(name, data, methods)
        for combined, parts in seeded.items():
            metrics[combined] = np.mean([metrics.pop(p) for p in parts], axis=0)
            if identity:
                identity[combined] = np.mean([identity.pop(p) for p in parts], axis=0)
        groups = np.array(data['groups'])
        summary = {k: dict(semantic=cluster_interval(v[known], groups[known]), **({'identity': cluster_interval(identity[k], groups)} if identity else {})) for k, v in metrics.items()}
        stored = read(V2 / 'evaluation' / f'{name}_results.json')['results']
        reference = {k: (summary[k]['semantic']['mean'][0], stored[k]['semantic']['mean'][0]) for k in ['CLIP_mix', 'DINO_image', 'dino_linear_mix']}
        # 1e-4 nDCG = 0.01 points; allows BLAS-level tie flips across machines. Exact values are recorded.
        assert all(abs(a - b) < 1e-4 for a, b in reference.values()), reference
        paired = {}
        for a, b in [('dinoL_linear_mix', 'dinotxt_mix'), ('dino_linear_mix', 't2d_mix'), ('dinotxt_mix', 'dinotxt_image'), ('t2d_mix', 't2d_image'),
                     ('DINOL_image', 'dinotxt_image'), ('dinoL_linear_mix', 'DINOL_image'), ('t2darch_mix', 't2d_mix'), ('dinoL_mlp_mix', 'dinotxt_mix'),
                     ('dinotxt_mix', 'CLIP_mix'), ('t2d_mix', 'CLIP_mix'), ('dinoL_linear_mix', 'dino_linear_mix'), ('t2darch_mix', 'dino_linear_mix')]:
            paired[f'{a} minus {b}'] = dict(semantic=cluster_interval((metrics[a] - metrics[b])[known], groups[known]),
                                            **({'identity': cluster_interval(identity[a] - identity[b], groups)} if identity else {}))
        results[name] = dict(gallery=len(data['ids']), evaluable_queries=int(known.sum()), results=summary, paired=paired,
                             reference_check={k: dict(recomputed=a, stored=b) for k, (a, b) in reference.items()})
        np.savez_compressed(OUT / f'{name}_per_query.npz', **metrics, **{f'identity_{k}': v for k, v in identity.items()}, known=known, groups=groups)
        for k in ['CLIP_mix', 'dino_linear_mix', 'dinoL_linear_mix', 'dinotxt_mix', 't2d_mix', 't2darch_mix']:
            print(name, k, round(summary[k]['semantic']['mean'][0] * 100, 2), flush=True)
    write('results.json', dict(protocol_sha256=digest(PROTOCOL), evaluator_sha256=digest(Path(__file__)), environment=environment(), text_weight=.2, metrics=['nDCG@10', 'P@10'],
                               identity_metrics=['R@1', 'R@5', 'R@10', 'MRR'], datasets=results))
    report(results)
    print('RELATED BASELINES COMPLETE', flush=True)


ROWS = [
    ('CLIP_mix', 'CLIP B/32 사진+글 (기존 기준선)'), ('DINO_image', 'DINOv3-B 사진만'), ('dino_linear_mix', 'DINOv3-B + Linear 정렬 (논문 주 방법)'),
    ('t2d_image', 'Talk2DINO 이미지만 (DINOv2-B reg, 어텐션 풀링)'), ('t2d_cls_image', 'Talk2DINO 백본 CLS만'), ('t2d_text_only', 'Talk2DINO 글만'),
    ('t2d_text_maxhead', 'Talk2DINO 글만 (헤드 최대, 원 학습 유사도)'), ('t2d_mix', 'Talk2DINO 사진+글 (zero-shot)'),
    ('t2darch_mix', 'Talk2DINO 구조 헤드를 우리 데이터로 재학습 (DINOv3-B)'), ('t2darch_text_only', '↳ 글만'),
    ('DINOL_image', 'DINOv3-L 사진만'), ('dinotxt_image', 'dino.txt 사진만'), ('dinotxt_text_only', 'dino.txt 글만'), ('dinotxt_mix', 'dino.txt 사진+글 (zero-shot)'),
    ('dinoL_linear_mix', 'DINOv3-L + Linear 정렬 (우리 방식)'), ('dinoL_linear_text_only', '↳ 글만'), ('dinoL_mlp_mix', 'DINOv3-L + MLP 정렬'),
]


def report(results):
    pct = lambda d: f"{d['mean'][0] * 100:.2f}"
    ci = lambda d: f"{d['mean'][0] * 100:+.2f} [{d['ci95_low'][0] * 100:+.2f}, {d['ci95_high'][0] * 100:+.2f}]"
    lines = ['# dino.txt · Talk2DINO 기준선 비교 결과', '',
             '[사전 프로토콜](RELATED_BASELINES_PROTOCOL.md)에 따라 실행. 코드: `python -m experiments.dog_domain.related_baselines --stage all`.',
             '값은 nDCG@10 × 100 (정확도 아님). 영어 템플릿 조건, 글 비중 0.20 고정, 학습 모델은 시드 17/29/43 질의별 평균.',
             '기존 참조 3개(CLIP_mix, DINO_image, dino_linear_mix)는 저장 특징으로 재계산해 논문 수치와 일치함을 확인했다.', '',
             '## 1. 속성 조건 검색', '', '| 방법 | PetFinder | 한국 | 대만 |', '|---|---:|---:|---:|']
    for key, label in ROWS:
        lines.append(f'| {label} | ' + ' | '.join(pct(results[d]['results'][key]['semantic']) for d in DATASETS) + ' |')
    lines += ['', '## 2. 사전 지정 비교 (차이 × 100, 보호소 단위 paired bootstrap 95% 구간)', '', '| 비교 | PetFinder | 한국 | 대만 |', '|---|---|---|---|']
    for key in results['PetFinder']['paired']:
        lines.append(f'| {key} | ' + ' | '.join(ci(results[d]['paired'][key]['semantic']) for d in DATASETS) + ' |')
    kr = results['Korea']
    lines += ['', '## 3. 한국: 같은 공고의 다른 사진 찾기 (R@1 / MRR × 100)', '', '| 방법 | R@1 | MRR |', '|---|---:|---:|']
    for key in ['DINO_image', 'DINOL_image', 'dinotxt_image', 't2d_image', 't2d_cls_image', 'dino_linear_mix', 'dinoL_linear_mix', 'dinotxt_mix', 't2d_mix']:
        m = kr['results'][key]['identity']['mean']
        lines.append(f'| {key} | {m[0] * 100:.2f} | {m[3] * 100:.2f} |')
    lines += ['', '| 비교 (R@1 차이) | 한국 |', '|---|---|']
    for key in ['DINOL_image minus dinotxt_image', 'dinotxt_mix minus dinotxt_image', 't2d_mix minus t2d_image']:
        d = kr['paired'][key]['identity']
        lines.append(f"| {key} | {d['mean'][0] * 100:+.2f} [{d['ci95_low'][0] * 100:+.2f}, {d['ci95_high'][0] * 100:+.2f}] |")
    lines += ['', '## 해석상 주의', '',
              '- dino.txt·Talk2DINO는 공개 가중치 그대로(zero-shot), 우리 헤드는 PetFinder로 학습했다. 방법 자체의 우열이 아니라 대규모 일반 정렬과 소규모 도메인 정렬의 비교다.',
              '- 백본(DINOv2-B/DINOv3-B/DINOv3-L), 텍스트 인코더(CLIP B/16·B/32, dino.txt 자체), 입력 해상도(518/224)가 방법마다 다르다.',
              '- 구간은 다중비교 보정 전 탐색적 구간이며 라벨 오류를 반영하지 않는다. 정답은 공고 기재 색·크기 기반 silver label이다.', '']
    RESULTS.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['weights', 'features', 'train', 'evaluate', 'all'], default='all')
    stage = parser.parse_args().stage
    assert torch.cuda.is_available()
    for name, step in [('weights', weights), ('features', features), ('train', train), ('evaluate', evaluate)]:
        if stage in (name, 'all'):
            step()
