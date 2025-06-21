

1 0 0 0 0 0 1 0 0

1 1 0 0 0 0 1 0 0

1 1 1 0 0 0 0 1 0

1 1 1 1 0 0 0 1 0

1 1 1 1 1 0 0 0 1

1 1 1 1 1 1 0 0 1

# mask矩阵
- 第1个compress-token只能看到压缩率的1倍token（5个）
- 第2个compress-token只能看到压缩率的2倍token（10个）
- 第3个compress-token只能看到压缩率的3倍token（15个）
- compress-token之间互不可见：这样可以防止局部归纳偏置问题