import torch


def build_compress_aware_causal_mask(total_len, input_len, compress_rate=3, device="cuda"):
    """
    返回一个 [total_len, total_len] 的 0/1 可见矩阵
    - 前 input_len 行是 input-token causal
    - 后面是 compress-token 按压缩率看前若干 input-token 以及自己
    """
    compress_len = total_len - input_len
    visible = torch.zeros((total_len, total_len), dtype=torch.int8, device=device)

    # 1. input-token: causal mask
    visible[:input_len, :input_len] = torch.tril(torch.ones((input_len, input_len), dtype=torch.int8, device=device))

    # 2. compress-token: 每个看前 k_i 个 input-token，以及自己
    for i in range(compress_len):
        row = input_len + i
        k = min(input_len, compress_rate * (i + 1))
        visible[row, :k] = 1              # 看前 k 个 input-token
        visible[row, row] = 1             # 只看自己（不看其他 compress）

    # True 表示不能看，因此 0 -> -inf, 1 -> 0.0
    float_mask = torch.where(visible == 1, torch.tensor(0.0, device=visible.device),torch.tensor(float('-inf'), device=visible.device))

    return float_mask


if __name__ == '__main__':
    total_len = 12
    input_len = 9
    attention_mask = build_compress_aware_causal_mask(total_len, input_len)
    print(attention_mask)