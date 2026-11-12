"""Independently verify and freeze the completed, fixed five-arm R1 screen."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def review(root):
    previous = root / 'evidence/r1-v1'
    source = root / 'JET/experiments/looped_jet_tuev_v1'
    frozen = previous / 'screen-freeze-v2/screen-config.json'
    approved = 'df72fb356fb197d4e59fd25a37bb8406e5b95a74a0eb575aa20b1ad216689bd3'
    assert sha(frozen) == approved
    config = json.loads(frozen.read_text())
    for name, digest in config['source_sha256'].items():
        assert sha(source / 'r1' / name) == digest
    for name, digest in config['operator_sha256'].items():
        assert sha(source / name) == digest
    for name, digest in config['fixture_files_sha256'].items():
        assert sha(previous / 'fixtures' / name) == digest
    assert sha(root / 'baseline-seed0/checkpoint-last.pth') == config['checkpoint_sha256']
    output = previous / 'screen-results-v1'
    assert (output / 'screen.finished').exists() and not (output / 'screen.failed').exists()
    results = json.loads((output / 'five-arm-status.json').read_text())
    filters = json.loads((output / 'continuation-filter.json').read_text())
    fixture = torch.load(previous / 'fixtures/references.pt', map_location='cpu', weights_only=True)
    reference = fixture['reference'].reshape(2020, 16, 1000).numpy()
    labels = fixture['labels'].numpy()
    report = {'approved_config_sha256': approved, 'checkpoint_sha256': config['checkpoint_sha256'],
              'review_source_sha256': sha(__file__), 'arms': {},
              'scope': 'Fixed validation screen only; no publication-reproduction claim or independent-seed uncertainty.'}
    for name in ('baseline', 'loop2', 'loop3', 'flow67', 'flow83'):
        arm = results[name]
        folder = output / name
        assert arm['status'] == 'completed' and (folder / 'completed.done').exists()
        assert not (folder / 'failed.done').exists()
        archive = folder / 'eval_batch.private.npz'
        assert sha(archive) == arm['archive_sha256']
        settings = json.loads((folder / 'configuration.json').read_text())
        assert settings['approved_config_sha256'] == approved
        assert settings['initial_noise_file_sha256'] == config['fixture_files_sha256']['screen-noise.pt']
        with np.load(archive, allow_pickle=False) as data:
            assert set(data.files) == {'generated', 'ground_truth', 'labels'}
            assert np.array_equal(data['ground_truth'], reference)
            assert np.array_equal(data['labels'], labels)
            generated = data['generated']
            assert generated.shape == (2020, 16, 1000) and generated.dtype == np.float32
            assert np.isfinite(generated).all() and not (generated == 0).all(axis=(1, 2)).any()
        metric = json.loads((folder / 'metrics.json').read_text())
        audit = json.loads((folder / 'numeric-audit.json').read_text())
        assert audit['passed'] and metric['num_samples'] == 2020
        assert metric['ts_fid']['overall'] == arm['ts_fid']
        assert [metric['label_histogram'][str(c)] for c in range(6)] == config['counts']
        report['arms'][name] = {'archive_sha256': arm['archive_sha256'],
                               'references_exactly_equal_frozen_fixture': True,
                               'labels_exactly_equal_frozen_fixture': True,
                               'finite_nonzero_generated_rows': 2020,
                               'numeric_audit_passed': True, 'ts_fid': arm['ts_fid']}
    for loop, control in (('loop2', 'flow67'), ('loop3', 'flow83')):
        fid_pass = results[loop]['ts_fid'] < min(results['baseline']['ts_fid'], results[control]['ts_fid'])
        assert filters['loops'][loop]['fid_strictly_below_both'] == fid_pass
        assert not fid_pass and not filters['loops'][loop]['passes']
    assert filters['provisional_selected_arm'] is None
    report['passed'] = True
    report['selection'] = None
    report['output_files_sha256'] = {str(path.relative_to(output)): sha(path)
                                    for path in sorted(output.rglob('*')) if path.is_file()}
    destination = previous / 'final-review-v1'
    destination.mkdir(mode=0o700)
    path = destination / 'review.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (destination / 'review.sha256').write_text(sha(path) + '  review.json\n')
    for path in output.rglob('*'):
        if path.is_file():
            path.chmod(0o400)
    for path in destination.iterdir():
        path.chmod(0o400)
    print(json.dumps({'passed': True, 'arms': {name: row['ts_fid'] for name, row in report['arms'].items()},
                      'review_sha256': sha(destination / 'review.json')}, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    review(parser.parse_args().root)
