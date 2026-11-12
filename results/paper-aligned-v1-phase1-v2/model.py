"""Joint 80-token, direct-vector JET for the reviewed paper-aligned target."""
import numpy as np
import torch
from torch import nn

from denoiser import Denoiser
from models.raw_vit import RawViTDiffusion
from util.model_util import get_1d_sincos_pos_embed_from_grid


def patchify(x):
    if x.ndim == 3 and x.shape[1:] == (16, 1000):
        return x.reshape(x.shape[0], 16, 5, 200)
    if x.ndim == 4 and x.shape[1:] == (16, 5, 200):
        return x
    raise ValueError("Expected [B,16,1000] or [B,16,5,200]")


class JointBackbone(RawViTDiffusion):
    def __init__(self):
        super().__init__("JiT-B/16", 16, 200, 1000, 6)
        positions = get_1d_sincos_pos_embed_from_grid(768, np.arange(80, dtype=np.float32))
        self.pos_embed = nn.Parameter(torch.from_numpy(positions).float().unsqueeze(0), requires_grad=True)

    def embed_tokens(self, x):
        patches = patchify(x)
        return self.patch_embed(patches).reshape(patches.shape[0], 80, 768) + self.pos_embed

    def forward(self, x, t, labels):
        batch = x.shape[0]
        condition = self.t_embedder(t) + self.y_embedder(labels)
        hidden = self.embed_tokens(x)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.final_layer(hidden, condition).reshape(batch, 16, 5, 200)


def endpoint(z, t, velocity):
    return z + (1 - t.reshape(-1, 1, 1)) * velocity


def loss_terms(reference, prediction):
    reference = patchify(reference).reshape(-1, 16, 1000)
    prediction = patchify(prediction).reshape(-1, 16, 1000)
    reconstruction = (reference - prediction).abs().mean()
    statistics = ((reference.mean(dim=-1) - prediction.mean(dim=-1)).abs().mean()
                  + (reference.std(dim=-1, correction=1) - prediction.std(dim=-1, correction=1)).abs().mean())
    variation = prediction.diff(dim=-1).abs().sum(dim=-1).div(1000).mean()
    correlation = Denoiser._pearson_corr_loss(reference, prediction)
    total = reconstruction + statistics + 0.1 * variation + 0.1 * correlation
    return {"reconstruction": reconstruction, "statistics": statistics, "tv": variation,
            "pearson": correlation, "total": total}


class PaperAlignedJET(nn.Module):
    # Preserve released EMA swap, update and class-drop semantics.
    _run_net = Denoiser._run_net
    update_ema = Denoiser.update_ema
    drop_labels = Denoiser.drop_labels

    def __init__(self):
        super().__init__()
        self.net = JointBackbone()
        self.num_classes = 6
        self.label_drop_prob = 0.1
        self.ema_decay1, self.ema_decay2 = 0.9999, 0.9996
        self.ema_params1 = None
        self.ema_params2 = None

    def initialize_ema(self):
        self.ema_params1 = [p.detach().clone() for p in self.parameters()]
        self.ema_params2 = [p.detach().clone() for p in self.parameters()]

    @staticmethod
    def sample_t(batch, device):
        return torch.sigmoid(torch.randn(batch, device=device) * 0.8 - 0.8).clamp_min(0.05)

    def forward(self, x, labels):
        x = patchify(x).reshape(-1, 16, 1000)
        t = self.sample_t(len(x), x.device)
        noise = torch.randn_like(x)
        z = t[:, None, None] * x + (1 - t[:, None, None]) * noise
        conditioning = self.drop_labels(labels) if self.training else labels
        velocity = self.net(patchify(z), t, conditioning).reshape_as(x)
        return loss_terms(x, endpoint(z, t, velocity))["total"]

    @torch.no_grad()
    def generate(self, labels, initial_noise=None, steps=50, cfg=1.0):
        if type(steps) is not int or steps < 1:
            raise ValueError("Positive integer solver steps required")
        z = torch.randn((len(labels), 16, 1000), device=labels.device) if initial_noise is None else initial_noise.clone()
        if z.shape != (len(labels), 16, 1000) or not torch.isfinite(z).all():
            raise ValueError("Invalid initial noise")
        schedule = torch.linspace(0, 1, steps + 1, device=z.device)

        def velocity(state, time):
            times = torch.full((len(labels),), time, device=z.device, dtype=torch.float32)
            conditional = self._run_net(patchify(state), times, labels).reshape_as(state)
            if cfg > 1:
                unconditional = self._run_net(patchify(state), times, torch.full_like(labels, 6)).reshape_as(state)
                return unconditional + cfg * (conditional - unconditional)
            return conditional

        for i in range(steps):
            current, following = float(schedule[i]), float(schedule[i + 1])
            dt = following - current
            first = velocity(z, current)
            if i < steps - 1:
                second = velocity(z + dt * first, following)
                z = z + 0.5 * dt * (first + second)
            else:
                z = z + dt * first
        return z

