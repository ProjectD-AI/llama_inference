import torch
import torch.nn as nn
import torch.nn.functional as F
from model.norm import RMSNorm
from model.rope import precompute_freqs_cis, apply_rotary_emb
import bitsandbytes as bnb
import math


class NormalLinear(nn.Linear):
    def reset_parameters(self) -> None:
        pass


class BnbInt8Linear(bnb.nn.Linear8bitLt):
    def __init__(self, *args, **kwargs):
        super().__init__(has_fp16_weights=False, threshold=6.0, *args, **kwargs)

    def reset_parameters(self) -> None:
        pass


def get_linear_layer(use_int8):
    if use_int8:
        return BnbInt8Linear
    return NormalLinear


class WordEmbedding(nn.Module):
    def __init__(self, args):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.emb_size)

    def forward(self, src):
        emb = self.embedding(src)
        return emb


class MultiHeadedAttention(nn.Module):
    def __init__(self, args, hidden_size, heads_num, attention_head_size, has_bias=True, use_int8=True):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.inner_hidden_size = heads_num * attention_head_size

        Linear = get_linear_layer(use_int8)
        self.linear_layers = nn.ModuleList(
            [Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
        )

        self.final_linear = Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

        # add cache to reduce compute source.
        self.cache_k = torch.zeros(
            (args.batch_size, args.seq_length, self.heads_num, self.per_head_size)
        )
        self.cache_v = torch.zeros(
            (args.batch_size, args.seq_length, self.heads_num, self.per_head_size)
        )

    def forward(self, key, value, query, start_pos, mask, freqs_cis):
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size
        query, key, value = [l(x).view(batch_size, -1, heads_num, per_head_size) \
                             for l, x in zip(self.linear_layers, (query, key, value))]
        query, key = apply_rotary_emb(query, key, freqs_cis=freqs_cis)
        if self.cache_k.device != key.device:
            self.cache_k = self.cache_k.to(key)
        if self.cache_v.device != value.device:
            self.cache_v = self.cache_v.to(value)

        self.cache_k[:batch_size, start_pos: start_pos + seq_length] = key
        self.cache_v[:batch_size, start_pos: start_pos + seq_length] = value

        key = self.cache_k[:batch_size, : start_pos + seq_length]
        value = self.cache_v[:batch_size, : start_pos + seq_length]

        query, key, value = [x.transpose(1, 2) for x in (query, key, value)]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        if mask is not None:
            scores += mask
        # probs = nn.Softmax(dim=-1)(scores)
        probs = F.softmax(scores.float(), dim=-1).type_as(query)
        output = torch.matmul(probs, value).transpose(1, 2).\
            contiguous().view(batch_size, seq_length, -1)
        return self.final_linear(output)


class GatedFeedForward(nn.Module):
    def __init__(self, hidden_size, feedforward_size, has_bias=True, use_int8=True):
        super(GatedFeedForward, self).__init__()
        Linear = get_linear_layer(use_int8)
        self.linear_gate = Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_1 = Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = F.silu

    def forward(self, x):
        # gate = self.act(self.linear_gate(x))
        gate = self.act(self.linear_gate(x)).type_as(x)
        inter_linear = self.linear_1(x)
        inter = gate * inter_linear
        output = self.linear_2(inter)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        # Multi-head Attention
        self.self_attn = MultiHeadedAttention(
            args, args.hidden_size, args.heads_num, attention_head_size, has_bias=has_bias,
            use_int8=args.use_int8
        )

        # FFN
        self.feed_forward = GatedFeedForward(
            args.hidden_size, args.feedforward_size, has_bias, use_int8=args.use_int8
        )

        self.layer_norm_1 = RMSNorm(args.hidden_size)
        self.layer_norm_2 = RMSNorm(args.hidden_size)

    def forward(self, hidden, start_pos, mask, freqs_cis=None):
        inter = self.layer_norm_1(hidden)
        inter = self.self_attn(inter, inter, inter, start_pos, mask, freqs_cis)
        hidden = hidden + inter
        output = self.layer_norm_2(hidden)
        output = self.feed_forward(output) + hidden
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num

        self.transformer = nn.ModuleList(
            [TransformerLayer(args) for _ in range(self.layers_num)]
        )

        self.layer_norm = RMSNorm(args.hidden_size)
        self.freqs_cis = precompute_freqs_cis(args.hidden_size // args.heads_num, args.max_seq_length * 2)

    def forward(self, emb, start_pos):
        batch_size, seq_length, _ = emb.size()
        mask = None
        if seq_length > 1:
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)

        hidden = emb
        freqs_cis = self.freqs_cis[start_pos: start_pos + seq_length].to(hidden.device)

        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, start_pos, mask, freqs_cis=freqs_cis)
        return self.layer_norm(hidden)


class LmOutput(nn.Module):
    def __init__(self, args):
        super(LmOutput, self).__init__()
        Linear = get_linear_layer(args.use_int8)
        self.lm = Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(self, x):
        return self.lm(x[:, -1, :])


class LLaMa(nn.Module):
    def __init__(self, args):
        super(LLaMa, self).__init__()
        self.embedding = WordEmbedding(args)
        self.encoder = TransformerEncoder(args)
        self.target = LmOutput(args)

    #@torch.inference_mode()
    def forward(self, src, start_pos):
        emb = self.embedding(src)
        output = self.encoder(emb, start_pos)
        output = self.target(output)
        return output
