#!/usr/bin/env python3
"""
BS-RoFormer 测试脚本
支持批量处理 input 目录下的音频文件，输出到 output 目录
"""

# 修复 OpenBLAS 警告
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import librosa
import soundfile as sf
from bs_roformer import BSRoformer
import numpy as np
import os
import sys
from pathlib import Path
import glob

print("=" * 60)
print("BS-RoFormer 音频分离测试脚本")
print("=" * 60)

# 配置 - 基于脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = str(SCRIPT_DIR / "input")
OUTPUT_DIR = str(SCRIPT_DIR / "output")
SAMPLE_RATE = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEMS = 6  # 6音轨：vocals, bass, drums, guitar, piano, other
STEM_NAMES = ['vocals', 'bass', 'drums', 'guitar', 'piano', 'other']

print(f"使用设备: {DEVICE}")
print(f"采样率: {SAMPLE_RATE} Hz")
print(f"音轨数量: {NUM_STEMS}")
print(f"音轨名称: {', '.join(STEM_NAMES)}")
print()

# 检查设备
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()

# 1. 创建模型
print("[1/4] 创建模型...")
model = BSRoformer(
    dim=512,
    depth=12,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_stems=NUM_STEMS,  # 6个音轨
    stereo=True
)
model.to(DEVICE)
model.eval()
print(f"✓ 模型创建成功！")
print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"  注意: 如果没有预训练权重，分离效果可能不理想")
print()

# 2. 检查输入目录
print(f"[2/4] 检查输入目录: {INPUT_DIR}")
if not os.path.exists(INPUT_DIR):
    print(f"✗ 错误: 输入目录不存在: {INPUT_DIR}")
    sys.exit(1)

# 查找所有音频文件
audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']
audio_files = []
for ext in audio_extensions:
    audio_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    audio_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))

if not audio_files:
    print(f"✗ 错误: 在 {INPUT_DIR} 目录下未找到音频文件")
    sys.exit(1)

print(f"✓ 找到 {len(audio_files)} 个音频文件")
for f in audio_files:
    print(f"  - {os.path.basename(f)}")
print()

# 3. 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[3/4] 输出目录: {OUTPUT_DIR}")
print()

# 4. 处理每个文件
print("[4/4] 开始处理音频文件...")
print("=" * 60)

for idx, audio_file in enumerate(audio_files, 1):
    print(f"\n[{idx}/{len(audio_files)}] 处理: {os.path.basename(audio_file)}")
    print("-" * 60)
    
    try:
        # 加载音频
        print("  加载音频...")
        mix, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=False)
        duration = len(mix[0]) / sr if len(mix.shape) > 1 else len(mix) / sr
        print(f"  音频信息: {mix.shape}, 采样率: {sr} Hz, 时长: {duration:.2f} 秒")
        
        # 转换为张量格式 (batch, channels, samples)
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix])  # 单声道转立体声
        mix_tensor = torch.from_numpy(mix).float().unsqueeze(0).to(DEVICE)
        print(f"  输入张量形状: {mix_tensor.shape}")
        
        # 进行分离
        print("  执行音频分离...")
        with torch.no_grad():
            separated = model(mix_tensor)  # 输出形状: (batch, stems, channels, samples)
        print(f"  分离结果形状: {separated.shape}")
        
        # 保存结果
        track_name = Path(audio_file).stem
        track_output_dir = os.path.join(OUTPUT_DIR, track_name)
        os.makedirs(track_output_dir, exist_ok=True)
        
        print(f"  保存结果到: {track_output_dir}/")
        for i, stem_name in enumerate(STEM_NAMES):
            stem_audio = separated[0, i].cpu().numpy()  # 移除batch维度并转到CPU
            output_path = os.path.join(track_output_dir, f"{stem_name}.wav")
            sf.write(output_path, stem_audio.T, sr)
            print(f"    ✓ {stem_name}.wav")
        
        print(f"  ✓ 完成: {os.path.basename(audio_file)}")
        
    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        continue

print()
print("=" * 60)
print("所有文件处理完成！")
print("=" * 60)
print(f"\n结果保存在: {os.path.abspath(OUTPUT_DIR)}")
print("\n注意: BS-RoFormer 需要预训练权重才能有好的分离效果。")
print("如果没有权重，这只是演示模型结构，分离效果可能不理想。")
print("建议寻找预训练的权重文件以获得更好的分离效果。")

