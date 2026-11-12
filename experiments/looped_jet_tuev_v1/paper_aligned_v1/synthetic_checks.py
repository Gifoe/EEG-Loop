"""CPU-only synthetic checks of the reviewed paper-aligned implementation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from data.metrics import _prepare_fid_features
from data.tuev_dataset import CustomDataset
from denoiser import Denoiser
from experiments.looped_jet_tuev_v1 import audit_ts_fid
from experiments.looped_jet_tuev_v1.paper_aligned_v1.data_view import convert_sample, loader_array, synthetic_mapping
from experiments.looped_jet_tuev_v1.paper_aligned_v1.metric import features, score
from experiments.looped_jet_tuev_v1.paper_aligned_v1.model import PaperAlignedJET, endpoint, loss_terms, patchify

torch.set_num_threads(1)
torch.manual_seed(0)
out = Path(sys.argv[1])
out.mkdir(mode=0o700)
report = {"checks": {}, "scope": "Synthetic CPU checks only; no checkpoint/data archive loading or real-signal generation."}
model = PaperAlignedJET().eval()
x = torch.randn(2, 16, 1000)
patches = patchify(x)
assert torch.equal(patches.reshape_as(x), x)
assert torch.equal(model.net.embed_tokens(x), model.net.patch_embed(patches).reshape(2, 80, 768) + model.net.pos_embed)
seen = []
handles = [block.attn.register_forward_pre_hook(lambda module, inputs: seen.append(list(inputs[0].shape))) for block in model.net.blocks]
output = model.net(x, torch.tensor([0.2, 0.8]), torch.tensor([0, 5]))
assert output.shape == (2, 16, 5, 200) and seen == [[2, 80, 768]] * 12
output.square().mean().backward()
assert model.net.pos_embed.requires_grad and model.net.pos_embed.grad.shape == (1, 80, 768)
assert torch.isfinite(model.net.pos_embed.grad).all() and model.net.pos_embed.grad.abs().sum() > 0
for handle in handles:
    handle.remove()
report["checks"]["joint_sequence"] = {"attention_shapes": seen, "position_shape": [1, 80, 768],
    "position_gradient_l1": float(model.net.pos_embed.grad.abs().sum()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "blocks": len(model.net.blocks), "heads": [b.attn.num_heads for b in model.net.blocks]}
model.zero_grad(set_to_none=True)
with torch.no_grad():
    changed = x.clone()
    changed[0, 0] += .25
    changed_output = model.net(changed, torch.tensor([0.2, 0.8]), torch.tensor([0, 5]))
    cross_channel_delta = float((changed_output[0, 1] - output.detach()[0, 1]).abs().max())
    assert cross_channel_delta > 1e-7
    null_output = model.net(x, torch.tensor([0.2, 0.8]), torch.full((2,), 6, dtype=torch.long))
    assert null_output.shape == output.shape and torch.isfinite(null_output).all()
report["checks"]["joint_sequence"]["cross_channel_perturbation_max_effect"] = cross_channel_delta
report["checks"]["joint_sequence"]["null_conditioned_forward_finite"] = True

reference = torch.zeros(1, 16, 1000)
piecewise = torch.arange(5.0).repeat_interleave(200).reshape(1, 1, 1000).expand_as(reference)
terms = loss_terms(reference, piecewise)
direct_stats = piecewise.mean(dim=-1).abs().mean() + piecewise.std(dim=-1, correction=1).mean()
assert terms["statistics"] == direct_stats
assert not torch.isclose(terms["statistics"], Denoiser._statistics_loss(patchify(reference), patchify(piecewise)))
assert torch.isclose(terms["tv"], torch.tensor(4 / 1000))
assert Denoiser._total_variation_loss(patchify(piecewise)) == 0
assert torch.equal(terms["total"], terms["reconstruction"] + terms["statistics"] + .1 * terms["tv"] + .1 * terms["pearson"])
report["checks"]["full_time_losses"] = {name: float(value) for name, value in terms.items()}

times = torch.tensor([0., .05, .5, .98, 1.])
z = torch.randn(5, 16, 1000)
velocity = torch.randn_like(z)
assert torch.equal(endpoint(z, times, velocity), z + (1 - times[:, None, None]) * velocity)
assert torch.equal(endpoint(z, times, velocity)[-1], z[-1])
assert torch.isfinite(endpoint(z, times, velocity)).all()
draws = model.sample_t(10000, "cpu")
assert draws.min() >= .05 and draws.max() <= 1
labels = torch.tensor([0, 1, 5])
model.label_drop_prob = 0
assert torch.equal(model.drop_labels(labels), labels)
model.label_drop_prob = 1
assert torch.equal(model.drop_labels(labels), torch.full_like(labels, 6))
assert torch.equal(labels, torch.tensor([0, 1, 5]))
observed = []
original_net_forward = model.net.forward
model.net.forward = lambda state, times, conditioning: (observed.append(conditioning.clone()) or torch.zeros_like(state))
model.label_drop_prob = 1
model.eval()
with torch.no_grad():
    model(torch.zeros(3,16,1000), labels)
model.train()
with torch.no_grad():
    model(torch.zeros(3,16,1000), labels)
assert torch.equal(observed[0], labels) and torch.equal(observed[1], torch.full_like(labels,6))
model.net.forward = original_net_forward
model.eval()
model.label_drop_prob = .1
calls = []
original = model._run_net
model._run_net = lambda state, t, labels: (calls.append(float(t[0])) or torch.ones_like(state))
generated = model.generate(labels, initial_noise=torch.zeros(3, 16, 1000))
assert len(calls) == 99 and torch.allclose(generated, torch.ones_like(generated), atol=1e-6)
model._run_net = original
assert PaperAlignedJET.update_ema is Denoiser.update_ema
assert PaperAlignedJET._run_net is Denoiser._run_net
report["checks"]["direct_vector"] = {"algebra_t": times.tolist(), "training_t_min_observed": float(draws.min()),
    "constant_vector_solver_nfe": len(calls), "final_constant_velocity_max_error": float((generated - 1).abs().max()),
    "ema_update_and_swap_reuse": True, "training_only_class_drop_passed": True}

t = torch.arange(1000, dtype=torch.float64) / 200
fixture = torch.stack([torch.cos(2 * torch.pi * frequency * t) for frequency in (8, 16, 24, 80)]).repeat_interleave(4, dim=0)[None]
raw = torch.fft.rfft(fixture, dim=-1).abs()
assert raw.shape[-1] == 501 and int(raw[0, -1].argmax()) == 400
spectrum = F.adaptive_avg_pool1d(raw, 256)
grouped = F.adaptive_avg_pool1d(spectrum.permute(0, 2, 1), 4).permute(0, 2, 1)
manual = torch.log1p(grouped.reshape(1, 1024))
manual = (manual - manual.mean(dim=1, keepdim=True)) / manual.std(dim=1, keepdim=True).clamp_min(1e-6)
assert np.array_equal(features(fixture), manual.numpy())
assert np.array_equal(features(fixture), _prepare_fid_features(fixture, 256, 4, None))
assert _prepare_fid_features(fixture, 256, 4, .5).shape[1] == 1000
real = torch.randn(32, 16, 1000)
perturbed = real + .4 * fixture.float()
checks = {}
for name, fake in (("identity", real), ("perturbed", perturbed)):
    native = score(fake, real, chunk_size=16)
    audit = audit_ts_fid.compare(features(fake), features(real), native)
    assert audit["passed"]
    checks[name] = audit
report["checks"]["metric"] = {"dimension": 1024, "rfft_coefficients": 501, "source_exact": True,
    "spatial_channel_groups": [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]], "numeric": checks}

sample = {"signal": np.random.default_rng(7).normal(size=(16,1000))*27.313,
          "label": np.array([3.]), "offending_channel": np.array([4.])}
view = convert_sample(sample)
assert sample["signal"].dtype == np.float64 and view["signal"].dtype == np.float32
assert np.array_equal(view["label"], sample["label"]) and np.array_equal(view["offending_channel"], sample["offending_channel"])
with tempfile.TemporaryDirectory() as temp:
    import pickle
    with (Path(temp)/"synthetic.pkl").open("wb") as stream:
        pickle.dump(view, stream)
    loaded, label = CustomDataset(temp, ["synthetic.pkl"])[0]
    assert loaded.dtype == np.float32 and np.array_equal(loaded, loader_array(view)) and label == 2
mapping = synthetic_mapping([("synthetic-a",sample),("synthetic-b",sample)])
assert len(mapping) == 2 and all(row["source_name"] == row["view_name"] for row in mapping)
assert mapping[0]["source_raw_sha256"] != mapping[0]["view_raw_sha256"]
try:
    synthetic_mapping([("same",sample),("same",sample)])
except ValueError:
    pass
else:
    raise AssertionError("Duplicate mapping accepted")
report["checks"]["data_view"] = {"synthetic_mapping": mapping, "exact_released_loader_match": True,
    "real_data_view_created": False}

report.update(passed=True, torch_version=torch.__version__, numpy_version=np.__version__,
              source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
(out/"synthetic-report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
(out/"checks.passed").touch()
print(json.dumps({"passed": True, "checks": list(report["checks"]),
                  "actual_parameter_count": report["checks"]["joint_sequence"]["trainable_parameters"]},indent=2))

