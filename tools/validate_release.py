"""Verify archived source hashes and result traceability using stdlib only."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate():
    forbidden = {'.pth', '.pt', '.ckpt', '.safetensors', '.edf', '.rec', '.pkl', '.npy', '.npz', '.pem', '.key', '.tar', '.gz', '.zip'}
    secret_rules = [re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
                    re.compile(rb'gh[pousr]_[A-Za-z0-9]{25,}'),
                    re.compile(rb'github_pat_[A-Za-z0-9_]{25,}'),
                    re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
                    re.compile(rb'(?:[A-Za-z0-9-]+\.)*cpolar\.io')]
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts
             and '__pycache__' not in p.parts]
    for path in files:
        assert path.suffix.lower() not in forbidden and '.private.' not in path.name, path.relative_to(ROOT)
        assert path.stat().st_size < 20 * 1024 * 1024, path.relative_to(ROOT)
        raw = path.read_bytes()
        if path.suffix in {'.py', '.sh', '.md', '.txt', '.json', '.log', '.csv', '.jsonl'}:
            for rule in secret_rules:
                assert not rule.search(raw), f'Secret/connection rule matched: {path.relative_to(ROOT)}'
        if path.suffix == '.py':
            ast.parse(raw.decode('utf-8-sig'), filename=str(path.relative_to(ROOT)))
        if path.suffix == '.json':
            load(path.relative_to(ROOT))
    manifest = ROOT / 'MANIFEST.sha256'
    assert manifest.is_file()
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split('  ', 1)
        assert relative not in listed and '..' not in Path(relative).parts
        assert sha(ROOT / relative) == digest, relative
        listed.add(relative)
    expected = {p.relative_to(ROOT).as_posix() for p in files if p != manifest}
    assert listed == expected, {'unlisted': sorted(expected-listed), 'missing': sorted(listed-expected)}
    module = Path('experiments/looped_jet_tuev_v1')
    aligned = load('results/provenance/paper-aligned-input-manifest.json')
    assert sha(ROOT / 'results/provenance/paper-aligned-input-manifest.json') == '4d98ade679088bcda8063772fd2210bb601005c0174950baf525fdaa73b1060c'
    for name, digest in aligned['source_sha256'].items():
        assert sha(ROOT / module / 'paper_aligned_v1' / name) == digest, name
    r1 = load('results/provenance/r1-screen-config.json')
    assert sha(ROOT / 'results/provenance/r1-screen-config.json') == 'df72fb356fb197d4e59fd25a37bb8406e5b95a74a0eb575aa20b1ad216689bd3'
    for name, digest in r1['source_sha256'].items():
        assert sha(ROOT / module / 'r1' / name) == digest, name
    for name, digest in r1['operator_sha256'].items():
        assert sha(ROOT / module / name) == digest, name
    with (ROOT / 'results/summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 11 and len({r['run_id'] for r in rows}) == 11
    for row in rows:
        evidence = load(row['evidence'])
        assert int(row['num_real']) == int(row['num_generated']) == evidence['num_samples']
        assert float(row['ts_fid']) == evidence['ts_fid']['overall'] and math.isfinite(float(row['ts_fid']))
        for c, label in enumerate(('spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg')):
            assert float(row['ts_fid_'+label]) == evidence['ts_fid']['per_class'][str(c)]
    epochs = [json.loads(line) for line in (ROOT / 'results/paper-aligned/seed0/epochs.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in epochs] == list(range(200))
    assert all(r['steps'] == 267 and math.isfinite(r['mean_loss']) for r in epochs)
    for metric in ('source1000', 'paper1024'):
        assert load(f'results/paper-aligned/test/numeric-audit-{metric}.json')['passed']
    for arm in ('baseline', 'loop2', 'loop3', 'flow67', 'flow83'):
        assert load(f'results/r1-screen-complete-v1/{arm}/numeric-audit.json')['passed']
    assert load('results/r1-screen-complete-v1/continuation-filter.json')['provisional_selected_arm'] is None
    status = load('RELEASE_STATUS.json')
    assert not status['published_value_reproduced'] and not status['aligned_train_evaluation_completed']
    print(json.dumps({'passed': True, 'files': len(files), 'verified_summary_rows': len(rows),
                      'aligned_training_epochs': len(epochs), 'gpu_used': False}))


if __name__ == '__main__':
    validate()
