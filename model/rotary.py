# model/rotary.py 파일 내용 전체 교체

import torch
from torch import nn

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
# model/rotary.py 수정 제안 (_apply_rotary_pos_emb_torchscript 함수만)

# @torch.jit.script # 일단 제거
def _apply_rotary_pos_emb_torchscript(q, k, cos, sin):
    """Applies Rotary Positional Embedding to the query and key tensors."""
    # q, k shape: [Batch, SeqLen, Heads, HeadDim]
    # cos, sin shape: [1, SeqLen, 1, 1, HeadDim] (Rotary 클래스에서 생성된 형태)
    
    # --- 여기가 수정된 부분 ---
    # cos, sin의 불필요한 중간 차원(dim 2, 3)을 제거하고 q, k와 브로드캐스팅 가능하도록 함
    # 최종 목표 shape: [1, SeqLen, 1, HeadDim] 또는 [SeqLen, HeadDim]
    if cos.ndim == 5:
        cos = cos.squeeze(3).squeeze(2) # shape: [1, SeqLen, HeadDim]
        sin = sin.squeeze(3).squeeze(2) # shape: [1, SeqLen, HeadDim]
    # --- 수정 끝 ---

    # cos, sin 텐서의 마지막 차원이 q, k와 같은지 확인
    if cos.shape[-1] != q.shape[-1]:
        # (이전의 HeadDim/2 처리 로직은 남겨둘 수 있음 - 만약을 위해)
        if cos.shape[-1] * 2 == q.shape[-1]: 
             cos = torch.cat((cos, cos), dim=-1)
             sin = torch.cat((sin, sin), dim=-1)
        else:
             raise ValueError(f"RoPE 적용 전 q/k와 cos/sin의 마지막 차원 크기가 맞지 않습니다. q: {q.shape}, cos: {cos.shape}")

    # q, k에 Rotary Embedding 적용
    # q: [B, S, H, D], cos: [1, S, 1, D] -> 브로드캐스팅 가능
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def apply_rotary_pos_emb(qkv, cos, sin):
    """
    Applies Rotary Positional Embedding to the query and key components of qkv.
    qkv shape: [Batch, SeqLen, 3, Heads, HeadDim]
    cos, sin shape: [1, SeqLen, 1, 1, HeadDim] or [1, SeqLen, 1, 1, HeadDim/2]
    """
    # q, k, v 분리
    q, k, v = qkv.chunk(3, dim=2)
    
    # q, k 에만 RoPE 적용
    # apply_rotary_pos_emb_torchscript 함수는 이제 q, k를 받음
    q_embed, k_embed = _apply_rotary_pos_emb_torchscript(q.float(), k.float(), cos.to(qkv.dtype), sin.to(qkv.dtype)) # float32로 연산 시도
    
    # 결과를 원래 dtype으로 변환하고 v와 합침
    qkv_out = torch.cat([q_embed.to(qkv.dtype), k_embed.to(qkv.dtype), v], dim=2)
    
    return qkv_out


# Rotary 클래스는 원본 그대로 유지
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        # x shape: [Batch, SeqLen, ...]
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # 최종 shape: [1, SeqLen, 1, 1, HeadDim]
            self.cos_cached = emb.cos()[None, :, None, None, :]
            self.sin_cached = emb.sin()[None, :, None, None, :]

        # cos_cached, sin_cached의 shape이 예상과 다른 경우 여기서 조정 가능
        # 예: return self.cos_cached.to(x.dtype), self.sin_cached.to(x.dtype)

        return self.cos_cached, self.sin_cached
