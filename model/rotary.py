# model/rotary.py 파일 내용 전체 교체

import torch
from torch import nn

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# @torch.jit.script # 일단 제거
def _apply_rotary_pos_emb_torchscript(q, k, cos, sin):
    """Applies Rotary Positional Embedding to the query and key tensors."""
    # q, k shape: [Batch, SeqLen, Heads, HeadDim]
    # cos, sin shape: [Batch, SeqLen, 1, 1, HeadDim] 또는 [1, SeqLen, 1, 1, HeadDim]
    
    # cos, sin 텐서의 마지막 차원이 q, k와 같은지 확인 및 조정
    if cos.shape[-1] != q.shape[-1]:
        if cos.shape[-1] * 2 == q.shape[-1]: # HeadDim/2 로 생성된 경우
             # print("Adjusting cos/sin shape for RoPE.")
             cos = torch.cat((cos, cos), dim=-1)
             sin = torch.cat((sin, sin), dim=-1)
        else:
             raise ValueError(f"q/k와 cos/sin의 마지막 차원 크기가 맞지 않습니다. q: {q.shape}, cos: {cos.shape}")
             
    # 불필요한 차원 제거 (q/k 와 shape 맞추기 위해)
    # cos, sin shape: [1, SeqLen, 1, HeadDim]
    if cos.ndim == 5 and cos.shape[2] == 1 and cos.shape[3] == 1:
         cos = cos.squeeze(2).squeeze(2) # shape: [1, SeqLen, HeadDim]
         sin = sin.squeeze(2).squeeze(2) # shape: [1, SeqLen, HeadDim]
    elif cos.ndim == 4 and cos.shape[2] == 1: # 예: [Batch, SeqLen, 1, HeadDim] 형태일 경우 대비
         cos = cos.squeeze(2)
         sin = sin.squeeze(2)
    elif cos.ndim != 3 or cos.shape[0] != 1 or cos.shape[1] != q.shape[1] or cos.shape[2] != q.shape[-1]:
        # 예상과 다른 shape일 경우 에러 발생시켜 확인 유도
        # print(f"Debug: q.shape={q.shape}, cos.shape={cos.shape}, sin.shape={sin.shape}")
        # raise ValueError("cos/sin 텐서 shape이 예상과 다릅니다. [1, SeqLen, HeadDim] 형태여야 합니다.")
        # 또는 브로드캐스팅 가능한 형태로 reshape 시도
        try:
             cos = cos.view(1, q.shape[1], -1)[..., :q.shape[-1]]
             sin = sin.view(1, q.shape[1], -1)[..., :q.shape[-1]]
        except Exception as e:
             raise ValueError(f"cos/sin shape 조정 실패: {e}. q:{q.shape}, cos:{cos.shape}")


    # q, k에 Rotary Embedding 적용
    # (q * cos) shape: [Batch, SeqLen, Heads, HeadDim]
    # rotate_half(q) shape: [Batch, SeqLen, Heads, HeadDim]
    # (rotate_half(q) * sin) shape: [Batch, SeqLen, Heads, HeadDim]
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
