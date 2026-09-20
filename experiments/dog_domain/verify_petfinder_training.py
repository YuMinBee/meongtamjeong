"""Check persisted split, feature and checkpoint invariants after training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.dino_fusion.alignment import projection_head_from_checkpoint, project_embeddings
from experiments.dog_domain.petfinder_train import OUT


def main():
    torch.set_num_threads(4)
    records = json.loads((OUT / 'records.json').read_text(encoding='utf-8'))
    report = json.loads((OUT / 'training_report.json').read_text(encoding='utf-8'))
    assert report.get('complete') and len(report['runs']) == 12
    assert len({r['pet_id'] for r in records}) == len(records)
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'validation']
    assert not {r['rescuer'] for r in train} & {r['rescuer'] for r in val}
    assert all(r['original_split'] == ('test' if r['split']=='test' else 'train') for r in records)
    arrays = np.load(OUT / 'features.npz')
    fingerprint = hashlib.sha256((OUT / 'records.json').read_bytes()).hexdigest()
    assert str(arrays['records_sha256']) == fingerprint
    assert len(arrays['clip']) == len(arrays['dino']) == len(train) + len(val)
    for name in ('clip', 'dino', 'text'):
        assert np.isfinite(arrays[name]).all()
        np.testing.assert_allclose(np.linalg.norm(arrays[name], axis=1), 1, atol=1e-5)
    checks = []
    for run in report['runs']:
        path = Path(run['checkpoint'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        head = projection_head_from_checkpoint(checkpoint).eval()
        head.load_state_dict(checkpoint['state_dict'], strict=True)
        with torch.inference_mode():
            output = project_embeddings(head, torch.tensor(arrays['text'][:8]))
        assert output.shape == (8, 512 if run['target']=='clip' else 768)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.norm(dim=1), torch.ones(8), atol=1e-5, rtol=1e-5)
        assert np.isclose(run['validation_loss'], min(x['validation_loss'] for x in run['history']))
        checks.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    audit = dict(passed=True, checks=checks, records_sha256=fingerprint,
                 source_sha256=hashlib.sha256(Path(__file__).with_name('petfinder_train.py').read_bytes()).hexdigest(),
                 scope='split invariants, feature finiteness/norm, all 12 checkpoint reloads and forward passes; not retrieval effectiveness')
    (OUT / 'verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('PASS: splits, features, 12 reloaded checkpoints; test not evaluated.')
    print('Training seconds:', round(sum(r['seconds'] for r in report['runs']), 2))


if __name__ == '__main__':
    main()
