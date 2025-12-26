# BS-RoFormer 模型参数说明

## 当前使用的参数

```python
model = BSRoformer(
    dim=512,                      # 模型维度
    depth=12,                     # Transformer 层数
    time_transformer_depth=1,      # 时间维度 Transformer 深度
    freq_transformer_depth=1,      # 频率维度 Transformer 深度
    num_stems=6,                  # 输出音轨数量（6音轨：vocals, bass, drums, guitar, piano, other）
    stereo=True                    # 是否使用立体声
)
```

## 所有可用参数

### 基础参数（必需）

- **`dim`** (int): 模型维度，默认 512
  - 控制模型的容量和表达能力
  - 越大模型越强，但内存占用也越大

- **`depth`** (int): Transformer 层数，默认 12
  - 控制模型的深度
  - 越多分离效果可能越好，但计算量越大

### 音轨和声道参数

- **`num_stems`** (int): 输出音轨数量，默认 1
  - 4音轨：vocals, drums, bass, other
  - 6音轨：vocals, bass, drums, guitar, piano, other
  - 可以自定义任意数量

- **`stereo`** (bool): 是否使用立体声，默认 False
  - True: 处理立体声音频（2声道）
  - False: 处理单声道音频（1声道）

### Transformer 深度参数

- **`time_transformer_depth`** (int): 时间维度 Transformer 深度，默认 2
  - 控制时间维度的注意力层数
  - 当前使用 1（更快的推理速度）

- **`freq_transformer_depth`** (int): 频率维度 Transformer 深度，默认 2
  - 控制频率维度的注意力层数
  - 当前使用 1（更快的推理速度）

### 注意力机制参数

- **`dim_head`** (int): 每个注意力头的维度，默认 64
  - 控制注意力的表达能力

- **`heads`** (int): 注意力头数量，默认 8
  - 多头注意力的头数
  - 越多模型越强，但计算量越大

- **`attn_dropout`** (float): 注意力层的 dropout，默认 0.0
  - 训练时防止过拟合
  - 推理时通常设为 0

- **`ff_dropout`** (float): 前馈网络的 dropout，默认 0.0
  - 训练时防止过拟合
  - 推理时通常设为 0

- **`flash_attn`** (bool): 是否使用 Flash Attention，默认 True
  - True: 使用 Flash Attention（更快，更省内存）
  - False: 使用标准注意力（兼容性更好）

### 频率分割参数

- **`freqs_per_bands`** (tuple[int, ...]): 每个频带的频率数量，默认见下方
  - 控制如何将频谱分割成不同的频带
  - 默认值：`(2, 2, ..., 2, 4, 4, ..., 4, 12, 12, ..., 12, 24, ..., 48, ..., 128, 129)`
  - 约 60 个频带，低频用更细的分割，高频用更粗的分割

- **`freq_range`** (tuple[int, int] | None): 频率范围，默认 None
  - 指定处理的频率范围 `(min_freq, max_freq)`
  - None: 处理全部频率
  - `(-1, -1)`: 处理全部频率（默认）
  - 例如 `(0, 1000)`: 只处理 0-1000 Hz

### 残差连接参数

- **`num_residual_streams`** (int): 残差流数量，默认 4
  - 控制 Hyper Connections 的流数量
  - 设为 1 可以禁用 Hyper Connections（节省内存）

- **`num_residual_fracs`** (int): 残差分数数量，默认 1
  - 作为残差流的替代方案，更省内存
  - 可以保留 Hyper Connections 的好处但内存占用更小

### STFT 参数（短时傅里叶变换）

- **`dim_freqs_in`** (int): 输入频率维度，默认 1025
  - STFT 输出的频率维度

- **`stft_n_fft`** (int): FFT 窗口大小，默认 2048
  - 控制频率分辨率
  - 越大频率分辨率越高，但时间分辨率越低

- **`stft_hop_length`** (int): 跳跃长度，默认 512
  - 在 44100 Hz 采样率下约为 10ms
  - 论文推荐使用 `// 2` 或 `// 4` 以获得更好的重建质量

- **`stft_win_length`** (int): 窗口长度，默认 2048
  - 通常等于 `stft_n_fft`

- **`stft_normalized`** (bool): 是否归一化 STFT，默认 False
  - True: 归一化 STFT 输出

- **`stft_window_fn`** (Callable | None): 窗口函数，默认 None（使用 hann_window）
  - 可以自定义窗口函数
  - 例如：`torch.hann_window`, `torch.hamming_window`

- **`zero_dc`** (bool): 是否将 DC 分量置零，默认 False
  - True: 将直流分量（0 Hz）置零

### 掩码估计参数

- **`mask_estimator_depth`** (int): 掩码估计器深度，默认 2
  - 控制掩码估计网络的层数

### 多尺度 STFT 损失参数（训练时使用）

- **`multi_stft_resolution_loss_weight`** (float): 多尺度损失权重，默认 1.0
  - 训练时多尺度 STFT 损失的权重

- **`multi_stft_resolutions_window_sizes`** (tuple[int, ...]): 多尺度窗口大小，默认 (4096, 2048, 1024, 512, 256)
  - 不同尺度的 FFT 窗口大小

- **`multi_stft_hop_size`** (int): 多尺度跳跃大小，默认 147
  - 多尺度 STFT 的跳跃长度

- **`multi_stft_normalized`** (bool): 多尺度是否归一化，默认 False

- **`multi_stft_window_fn`** (Callable): 多尺度窗口函数，默认 `torch.hann_window`

## 推荐配置

### 快速推理（当前使用）

```python
model = BSRoformer(
    dim=512,
    depth=12,
    time_transformer_depth=1,      # 减少深度以加快推理
    freq_transformer_depth=1,      # 减少深度以加快推理
    num_stems=6,
    stereo=True,
    flash_attn=True                # 使用 Flash Attention 加速
)
```

### 高质量分离

```python
model = BSRoformer(
    dim=512,
    depth=12,
    time_transformer_depth=2,      # 增加深度提高质量
    freq_transformer_depth=2,      # 增加深度提高质量
    num_stems=6,
    stereo=True,
    flash_attn=True,
    stft_hop_length=256            # 更小的跳跃长度，更好的重建质量
)
```

### 节省内存

```python
model = BSRoformer(
    dim=256,                       # 减小模型维度
    depth=8,                      # 减少层数
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_stems=6,
    stereo=True,
    num_residual_streams=1,       # 禁用 Hyper Connections
    flash_attn=True
)
```

## 注意事项

1. **预训练权重**: 当前脚本使用的是随机初始化的模型，分离效果可能不理想。需要加载预训练权重才能获得好的分离效果。

2. **内存占用**: 
   - `dim` 和 `depth` 越大，内存占用越大
   - `num_residual_streams=1` 可以显著减少内存占用
   - 使用 `flash_attn=True` 可以节省内存

3. **推理速度**:
   - `time_transformer_depth` 和 `freq_transformer_depth` 越小，推理越快
   - `flash_attn=True` 可以加速推理

4. **分离质量**:
   - 需要预训练权重才能有好的分离效果
   - 增加 `depth`、`time_transformer_depth`、`freq_transformer_depth` 可能提高质量
   - 减小 `stft_hop_length` 可以提高重建质量

