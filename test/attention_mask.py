import torch


def build_compress_aware_causal_mask(total_len, compress_indices, device="cuda"):
    """
    构建 Compress-Aware 因果注意力 mask（适配 scaled_dot_product_attention）
    返回一个 [T+N, T+N] 的 mask，dtype=float32，禁止 attend 位置为 -inf，允许 attend 为 0.0
    """
    mask = torch.triu(torch.ones(total_len, total_len, device=device), diagonal=1).bool()

    compress_set = set(compress_indices)
    for i in range(total_len):
        for j in range(total_len):
            if i in compress_set:
                # 当前是 compress-token
                if j in compress_set and j != i:
                    mask[i, j] = True  # 禁止看其他 compress
            else:
                if j in compress_set:
                    mask[i, j] = True  # input-token 禁止看 compress

    # 将bool mask转float mask：True->-inf，False->0.0
    float_mask = torch.where(mask, torch.tensor(float('-inf'), device=device), torch.tensor(0.0, device=device))

    return float_mask



if __name__ == '__main__':
    compress_indices = [3, 7, 11]
    seq_len = 12
    attention_mask = build_compress_aware_causal_mask(seq_len, compress_indices)
    print(attention_mask)

