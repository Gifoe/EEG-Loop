"""Separately timed core-exit probes with diagnostic suffix decoding."""
import time

import torch

from experiments.looped_jet_tuev_v1.r1.recurrent_wrapper import RecurrentForward


def describe(hidden, batch):
    values = hidden.detach().reshape(batch, -1).double()
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite hidden state during probe")
    return {"rms": values.square().mean(dim=1).sqrt().tolist(),
            "maximum_absolute": values.abs().max(dim=1).values.tolist(), "finite": True}


def delta(actual, reference):
    a, b = actual.double().flatten(1), reference.double().flatten(1)
    difference = a - b
    return {"absolute_rms": difference.square().mean(dim=1).sqrt().tolist(),
            "relative_l2": (difference.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).tolist()}


def run_probes(model, fixture, output):
    original_forward = model.net.forward
    x, noise, labels = (fixture[key].cuda() for key in ("reference", "noise", "labels"))
    rows, predictions, failures = [], {}, []
    failed_k = set()
    decode_count = [0]
    started = time.perf_counter()
    for t in (0.0, 0.1, 0.5, 0.9, 0.98):
        z = t * x + (1 - t) * noise
        times = torch.full((12,), t, device="cuda", dtype=torch.float32)
        baseline = None
        for k in (1, 2, 3):
            if k in failed_k:
                continue
            record = {"t": t, "k": k, "labels": labels.cpu().tolist(), "core_passes": []}
            context = {"decoding": False, "previous_core": None}

            def prefix_hook(module, inputs, hidden):
                if not context["decoding"]:
                    record["prefix_exit"] = describe(hidden, 12)

            def core_hook(module, inputs, hidden):
                if context["decoding"]:
                    return
                entry = {"pass": len(record["core_passes"]) + 1, "hidden": describe(hidden, 12)}
                previous = context["previous_core"]
                if previous is not None:
                    a, b = hidden.double().reshape(12, -1), previous.double().reshape(12, -1)
                    norms = a.norm(dim=1) * b.norm(dim=1)
                    cosine = (a * b).sum(dim=1) / norms.clamp_min(1e-12)
                    entry["relative_change"] = ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).tolist()
                    entry["cosine_similarity"] = [float(value) if valid else None for value, valid in zip(cosine.cpu(), (norms > 0).cpu())]
                    entry["cosine_zero_norm_examples"] = (norms == 0).cpu().tolist()
                context["previous_core"] = hidden.detach().clone()
                context["decoding"] = True
                decode_count[0] += 1
                decoded = hidden
                for block in model.net.blocks[8:]:
                    decoded = block(decoded, inputs[1])
                entry["diagnostic_suffix_exit"] = describe(decoded, 12)
                endpoint = model.net.final_layer(decoded, inputs[1]).view(12, 16, 5, 200)
                if not torch.isfinite(endpoint).all():
                    raise ValueError("Nonfinite intermediate endpoint")
                entry["endpoint"] = endpoint.detach().cpu()
                context["decoding"] = False
                record["core_passes"].append(entry)

            def suffix_hook(module, inputs, hidden):
                if not context["decoding"]:
                    record["suffix_exit"] = describe(hidden, 12)

            hooks = [model.net.blocks[3].register_forward_hook(prefix_hook),
                     model.net.blocks[7].register_forward_hook(core_hook),
                     model.net.blocks[11].register_forward_hook(suffix_hook)]
            model.net.forward = RecurrentForward(k).bind(model.net)
            backup = [parameter.detach().clone() for parameter in model.parameters()]
            try:
                with torch.no_grad():
                    endpoint = model._run_net(z, times, labels)
            except Exception as error:
                with torch.no_grad():
                    for parameter, saved in zip(model.parameters(), backup, strict=True):
                        parameter.copy_(saved)
                failed_k.update(range(k, 4))
                failures.append({"t": t, "k": k, "error_type": type(error).__name__, "error": str(error)})
                torch.save(record, output / f"failed-t{t}-k{k}.private.pt")
                continue
            finally:
                for hook in hooks:
                    hook.remove()
                model.net.forward = original_forward
                del backup
            if baseline is None:
                baseline = endpoint.detach()
            for entry in record["core_passes"]:
                intermediate = entry.pop("endpoint").cuda()
                entry["endpoint_vs_k1"] = delta(intermediate, baseline)
                entry["vector_field_vs_k1"] = delta((intermediate - z) / max(1 - t, 0.05),
                                                    (baseline - z) / max(1 - t, 0.05))
                predictions[f"t{t}-k{k}-pass{entry['pass']}"] = intermediate.cpu()
            record["endpoint_vs_k1"] = delta(endpoint, baseline)
            rows.append(record)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    torch.save(predictions, output / "intermediate-predictions.private.pt")
    return {"rows": rows, "elapsed_s": elapsed,
            "failed_k": sorted(failed_k), "failures": failures,
            "extra_suffix_decodes": decode_count[0],
            "extra_transformer_blocks_per_example": decode_count[0] * 4,
            "scope": "Separate conditioned twelve-example probe replay. Extra diagnostic suffix forwards excluded from clean generation/profile."}


