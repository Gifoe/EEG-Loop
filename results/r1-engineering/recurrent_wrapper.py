"""Unchanged-weight prefix/core-loop/suffix forward for the frozen R1 protocol."""
from types import MethodType


class RecurrentForward:
    def __init__(self, passes):
        if type(passes) is not int or passes < 1:
            raise ValueError("Core passes must be a positive integer")
        self.passes = passes

    def bind(self, net):
        return MethodType(self.forward, net)

    def forward(self, net, x, t, y):
        if x.dim() != 4:
            raise ValueError("Expected B,C,patches,patch_size")
        bsz, channels, patch_num, patch_size = x.shape
        if channels != net.num_channels or patch_num != net.patch_num or patch_size != net.patch_size:
            raise ValueError("Input shape does not match the frozen backbone")
        cond = net.t_embedder(t) + net.y_embedder(y)
        cond = cond.repeat_interleave(channels, dim=0)
        feats = net.patch_embed(x)
        feats = feats.view(bsz * channels, patch_num, net.hidden_size)
        feats = feats + net.pos_embed
        for block in net.blocks[:4]:
            feats = block(feats, cond)
        for _ in range(self.passes):
            for block in net.blocks[4:8]:
                feats = block(feats, cond)
        for block in net.blocks[8:]:
            feats = block(feats, cond)
        feats = net.final_layer(feats, cond)
        return feats.view(bsz, channels, patch_num, patch_size)
