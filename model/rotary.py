# model/rotary.py 파일 수정

import torch
from torch import nn

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# @torch.jit.script # 일단 제거
def _apply_rotary_pos_emb_torchscript(x, cos, sin):
    """Applies RoPE to a tensor x (q or k)"""
    # x shape: [Batch, SeqLen, Heads, HeadDim]
    # cos, sin shape: [1, SeqLen, 1, HeadDim]
    return (x * cos) + (rotate_half(x) * sin)

def apply_rotary_pos_emb(qkv, cos, sin):
    """
    Applies Rotary Positional Embedding to the query and key components of qkv.
    qkv shape: [Batch, SeqLen, 3, Heads, HeadDim]
    cos, sin shape: [1, SeqLen, 1, 1, HeadDim] (from Rotary class)
    """
    # cos, sin의 shape을 [1, SeqLen, 1, HeadDim] 으로 조정
    if cos.ndim == 5:
        cos = cos.squeeze(3) # shape: [1, SeqLen, 1, HeadDim]
        sin = sin.squeeze(3) # shape: [1, SeqLen, 1, HeadDim]
    elif cos.ndim == 3: # 이미 [1, SeqLen, HeadDim] 형태라면 unsqueeze 필요
        cos = cos.unsqueeze(2) # shape: [1, SeqLen, 1, HeadDim]
        sin = sin.unsqueeze(2) # shape: [1, SeqLen, 1, HeadDim]
    elif cos.ndim != 4 or cos.shape[2] != 1: # 예상과 다른 형태면 에러
         raise ValueError(f"apply_rotary_pos_emb 내 cos/sin shape 오류: {cos.shape}")

    # cos, sin 데이터 타입 맞추기 (필요시)
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)

    # q, k, v 분리
    q, k, v = qkv.chunk(3, dim=2)
    
    # q, k 에만 RoPE 적용 (이제 _apply... 함수는 q 또는 k 하나만 받음)
    q_embed = _apply_rotary_pos_emb_torchscript(q.float(), cos, sin)
    k_embed = _apply_rotary_pos_emb_torchscript(k.float(), cos, sin)
    
    # 결과를 원래 dtype으로 변환하고 v와 합침
    qkv_out = torch.cat([q_embed.to(qkv.dtype), k_embed.to(qkv.dtype), v], dim=2)
    
    return qkv_out

# Rotary 클래스는 그대로 유지
class Rotary(torch.nn.Module):
    # ... (Rotary 클래스 정의는 원본 또는 이전 상태 유지) ...
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :] # [1, S, 1, 1, D]
            self.sin_cached = emb.sin()[None, :, None, None, :] # [1, S, 1, 1, D]
        return self.cos_cached, self.sin_cached
