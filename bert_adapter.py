"""Route HF BERT self-attention through an `attnreg.AttnReg` regularizer.

Load the model with attn_implementation="eager" and call `install(model, factory)`.
Each `layer.attention.self` keeps its query/key/value submodules and weights; only its
class is swapped, so the pretrained parameters and any adapter wrappers stay intact.
The built-in attention-probability dropout is bypassed: the regularizer fully defines
attention perturbation for every arm (use kind="dropout" for the standard recipe).
Hidden-state dropout elsewhere in BERT is unchanged.
"""
import torch
from torch import nn

from attnreg import AttnReg


def _bert_classes():
    from transformers.models.bert import modeling_bert
    return modeling_bert.BertSelfAttention


class RegularizedBertSelfAttention(nn.Module):
    """Class-swap target; instances are created from existing BertSelfAttention objects."""

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, past_key_values=None, output_attentions=False,
                cache_position=None, **kwargs):
        if encoder_hidden_states is not None or past_key_values is not None:
            raise NotImplementedError("Only encoder self-attention is supported")
        if self.position_embedding_type != "absolute":
            raise NotImplementedError("Only absolute position embeddings are supported")
        b, s, _ = hidden_states.shape
        h, d = self.num_attention_heads, self.attention_head_size
        q = self.query(hidden_states).view(b, s, h, d).transpose(1, 2)
        k = self.key(hidden_states).view(b, s, h, d).transpose(1, 2)
        v = self.value(hidden_states).view(b, s, h, d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
        if attention_mask is None:
            allowed = torch.ones((b, 1, 1, s), dtype=torch.bool, device=scores.device)
        else:
            if attention_mask.dtype == torch.bool:
                allowed = attention_mask
            else:
                allowed = attention_mask > (torch.finfo(attention_mask.dtype).min / 2)
        if getattr(self.attnreg, "value_observer", None) is not None:
            with torch.no_grad():
                self.attnreg.value_observer(self.attnreg.layer, scores.detach(), allowed, v.detach())
        probs = self.attnreg(scores, allowed)
        if head_mask is not None:
            probs = probs * head_mask
        context = torch.matmul(probs.to(v.dtype), v)
        context = context.permute(0, 2, 1, 3).reshape(b, s, h * d)
        return context, (probs if output_attentions else None)


def install(model, factory):
    """Swap every BertSelfAttention in `model`; factory(layer_index, num_layers) -> AttnReg."""
    base = _bert_classes()
    targets = [m for m in model.modules() if type(m) is base or isinstance(m, RegularizedBertSelfAttention)]
    if not targets:
        raise ValueError("No eager BertSelfAttention modules found; load with attn_implementation='eager'")
    swapped = type("RegBertSelfAttention", (RegularizedBertSelfAttention, base), {})
    for i, module in enumerate(targets):
        reg = factory(i, len(targets))
        if not isinstance(reg, AttnReg):
            raise TypeError("factory must return an AttnReg")
        module.__class__ = swapped
        reg.layer = i
        module.attnreg = reg
    return [m.attnreg for m in targets]


def set_noise(model, noise):
    """Attach one attnreg.NoiseSource (or None for the global RNG) to every layer."""
    for reg in regularizers(model):
        reg.noise = noise


def regularizers(model):
    return [m.attnreg for m in model.modules() if isinstance(m, RegularizedBertSelfAttention)]


def reset_stats(model):
    for reg in regularizers(model):
        reg.reset_stats()


def collect_stats(model):
    return [reg.stats() for reg in regularizers(model)]
