import torch


def build_compress_aware_causal_mask(total_len, compress_indices, device="cuda"):
    """
    构建 Compress-Aware 因果注意力 mask（适配 scaled_dot_product_attention）
    返回一个 [T+N, T+N] 的 mask，dtype=float32，禁止 attend 位置为 -inf，允许 attend 为 0.0
    """
    # 初始 causal mask（上三角）
    mask = torch.triu(torch.ones(total_len, total_len, device=device), diagonal=1).bool()  # [total_len, total_len]

    # 构建 compress-token 索引的 bool mask
    compress_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
    compress_mask[compress_indices] = True  # shape: [total_len]

    # Expand 成 [T+N, T+N]
    # input-token 禁止 attend compress-token（行非compress，列compress）
    mask |= (~compress_mask).unsqueeze(1) & compress_mask.unsqueeze(0)

    # compress-token 禁止 attend 其他 compress-token（行compress，列compress，排除自己）
    compress_matrix = compress_mask.unsqueeze(0) & compress_mask.unsqueeze(1)
    compress_matrix.fill_diagonal_(False)
    mask |= compress_matrix

    # 转换成 float mask
    float_mask = torch.where(mask, torch.tensor(float('-inf'), device=device), torch.tensor(0.0, device=device))
    return float_mask

def Bidirectional_build_compress_aware_causal_mask(total_len, compress_indices, device="cuda"):
    """
    构建一个 float 型 attention mask：
    - input-token 遵循 causal 结构
    - compress-token 可以看到所有 token
    - 返回: [total_len, total_len] 的 float32 tensor，值为 0.0 或 -inf
    """
    mask = torch.zeros((total_len, total_len), dtype=torch.float32, device=device)

    # causal mask
    causal = torch.tril(torch.ones((total_len, total_len), dtype=torch.float32, device=device))

    # compress token index mask
    compress_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
    compress_mask[compress_indices] = True

    for i in range(total_len):
        if compress_mask[i]:
            mask[i] = 1.0  # compress-token 行全为 1（全可见）
        else:
            mask[i] = causal[i]  # input-token 遵循 causal

    # 转换成 float mask：不可见（0）→ -inf，可见（1）→ 0.0
    float_mask = torch.where(mask == 0, torch.tensor(float('-inf'), device=device), torch.tensor(0.0, device=device))
    return float_mask



if __name__ == '__main__':
    compress_indices = [3, 7, 11]
    seq_len = 12
    attention_mask = Bidirectional_build_compress_aware_causal_mask(seq_len, compress_indices)
    print(attention_mask)

