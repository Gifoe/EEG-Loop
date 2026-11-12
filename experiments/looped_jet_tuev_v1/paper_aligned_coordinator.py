"""Durable execution of the reviewed view -> freeze -> own training -> evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

MODULE = 'experiments.looped_jet_tuev_v1.paper_aligned_v1.'
COUNTS = {'train': [526, 10963, 5515, 984, 10553, 39904],
          'val': [119, 291, 669, 86, 500, 13822],
          'test': [567, 4677, 1998, 329, 2204, 19646]}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def validate_view(root, output):
    view = root / 'evidence/paper-aligned-v1-data-view'
    assert (view / 'view.done').exists(), 'Materialization did not complete'
    data = json.loads((view / 'view-manifest.json').read_text())
    frozen = root / 'evidence/paper-aligned-v1-materializer-freeze/manifest.json'
    freeze = json.loads(frozen.read_text())
    assert sha(frozen) == data['materializer_freeze_sha256']
    assert sha(frozen) == frozen.with_name('manifest.sha256').read_text().split()[0]
    source = root / 'JET/experiments/looped_jet_tuev_v1/paper_aligned_v1'
    for name, digest in freeze['source_sha256'].items():
        assert sha(source / name) == digest
    synthetic = json.loads((root / 'evidence/paper-aligned-v1-newenv-synthetic/synthetic-report.json').read_text())
    assert synthetic['passed']
    for name, digest in synthetic['source_sha256'].items():
        assert sha(source / name) == digest
    ledger = view / data['mapping_ledger']
    assert sha(ledger) == data['mapping_ledger_sha256']
    labels = {split: Counter() for split in COUNTS}
    names = {split: set() for split in COUNTS}
    with ledger.open() as stream:
        for line in stream:
            row = json.loads(line)
            split = row['split']
            assert row['view_name'] == row['source_name'] and row['view_name'] not in names[split]
            names[split].add(row['view_name'])
            labels[split][int(row['label'])] += 1
    assert not any(data['cross_split_view_loader_hash_overlaps'].values())
    for split, counts in COUNTS.items():
        summary = data['splits'][split]
        assert summary['rows'] == sum(counts) == len(names[split]) and summary['one_to_one_mapping']
        assert [summary['class_counts'][str(c)] for c in range(6)] == counts
        assert [labels[split][c] for c in range(6)] == counts
        assert names[split] == {p.name for p in (Path(data['view']) / split).glob('*.pkl')}
        assert data['source_index_sha256'][split] == freeze['inputs_sha256'][f'reference-index/{split}-seed0.npz']
        for key in ('raw_max_abs_change_uv', 'raw_rms_change_uv', 'loader_max_abs_change', 'loader_rms_change'):
            assert math.isfinite(summary[key]) and summary[key] >= 0
    write(output / 'view-review.json', {'passed': True, 'manifest_sha256': sha(view / 'view-manifest.json'),
          'ledger_sha256': sha(ledger), 'counts': COUNTS, 'unique_ordered_mapping_rows': sum(map(len, names.values())),
          'cross_split_loader_hash_overlaps': data['cross_split_view_loader_hash_overlaps'],
          'scope': 'Independent aggregate/ledger/source review; per-file readback and finite waveform checks are recorded by the frozen full materializer.'})


def run(root, materializer_pid):
    output = root / 'evidence/paper-aligned-v1-run-v1'
    try:
        while not (root / 'evidence/paper-aligned-v1-data-view/view.done').exists():
            proc = Path(f'/proc/{materializer_pid}')
            assert proc.exists(), 'Materializer exited without view.done; no automatic retry'
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            assert MODULE + 'materialize_view' in command and str(root) in command
            assert (proc / 'stat').read_text().split()[2] != 'Z', 'Materializer became zombie without completion'
            write(output / 'status.json', {'phase': 'materialization', 'materializer_pid': materializer_pid})
            time.sleep(15)
        validate_view(root, output)
        write(output / 'status.json', {'phase': 'input-freeze'})
        with (output / 'input-freeze.log').open('xb') as log:
            subprocess.run([sys.executable, '-u', '-m', MODULE + 'freeze_inputs', str(root)],
                           cwd=root / 'JET', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        frozen = root / 'evidence/paper-aligned-v1-input-freeze/manifest.json'
        digest = sha(frozen)
        assert digest == frozen.with_name('manifest.sha256').read_text().split()[0]
        for phase, module, marker in (
                ('training', 'train_aligned', root / 'paper-aligned-v1-seed0/training.done'),
                ('evaluation', 'evaluate_aligned', root / 'paper-aligned-v1-eval-train-seed0/evaluation.done')):
            assert not marker.exists(), 'Fresh phase already completed; no duplicate launch'
            with (output / f'{phase}.log').open('xb') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', MODULE + module, str(root)],
                         cwd=root / 'JET', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                write(output / f'{phase}.pid.json', {'pid': child.pid, 'module': MODULE + module})
                write(output / 'status.json', {'phase': phase, 'child_pid': child.pid, 'manifest_sha256': digest})
                result = child.wait()
                assert result == 0 and marker.exists(), f'{phase} failed exit={result}; inspect log, no automatic retry'
        assert (root / 'paper-aligned-v1-eval-test-seed0/evaluation.done').exists()
        write(output / 'status.json', {'phase': 'seed0-training-and-evaluation-complete', 'manifest_sha256': digest})
        (output / 'coordinator.done').touch()
    except BaseException:
        write(output / 'failure.json', {'traceback': traceback.format_exc()})
        raise


def launch(root, materializer_pid):
    assert root.resolve() == Path('/home/luansiyu/EEG-Loop')
    prefix = root / 'conda-paper-aligned-v1'
    assert Path(sys.executable).resolve().is_relative_to(prefix.resolve())
    output = root / 'evidence/paper-aligned-v1-run-v1'
    output.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (output / 'coordinator.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.looped_jet_tuev_v1.paper_aligned_coordinator',
                   str(root), '--materializer-pid', str(materializer_pid), '--run'], cwd=root / 'JET',
                   env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': process.pid, 'materializer_pid': materializer_pid, 'source_sha256': sha(__file__),
              'scope': 'Approved single own fresh seed0 200epoch run; full test then train, fixed two metrics; no author weights/Loop search/new dataset.'}
    write(output / 'launch.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--materializer-pid', type=int, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    (run if args.run else launch)(args.root, args.materializer_pid)
