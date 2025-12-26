#!/usr/bin/env python3
"""
BS-RoFormer 高质量音频分离测试脚本
使用更深的 Transformer 和更精细的 STFT 参数，分离质量更好但速度较慢
"""

# 修复 OpenBLAS 警告和 GPU 内存管理
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import librosa
import soundfile as sf
from bs_roformer import BSRoformer
import numpy as np
import os
import sys
from pathlib import Path
import glob
import torchaudio

print("=" * 60)
print("BS-RoFormer 高质量音频分离测试脚本")
print("=" * 60)

# 配置 - 基于脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
INPUT_DIR = str(SCRIPT_DIR / "input")
OUTPUT_DIR = str(SCRIPT_DIR / "output")
SAMPLE_RATE = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEMS = 6  # 6音轨：vocals, bass, drums, guitar, piano, other
STEM_NAMES = ['vocals', 'bass', 'drums', 'guitar', 'piano', 'other']
SEGMENT_LENGTH = 20  # 高质量模式使用更短的分段（20秒），避免内存不足
OVERLAP = 2  # 分段重叠（秒）

print(f"使用设备: {DEVICE}")
print(f"采样率: {SAMPLE_RATE} Hz")
print(f"音轨数量: {NUM_STEMS}")
print(f"音轨名称: {', '.join(STEM_NAMES)}")
print(f"模式: 高质量（更慢但质量更好）")
print()

# 检查设备
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    print(f"已清理 GPU 缓存")
print()

# 1. 创建模型（高质量配置）
print("[1/4] 创建高质量模型...")
print("  配置: dim=512, depth=12, time_depth=2, freq_depth=2, hop_length=256")
model = BSRoformer(
    dim=512,                      # 模型维度
    depth=12,                     # Transformer 层数
    time_transformer_depth=2,     # 时间维度深度（增加以提高质量）
    freq_transformer_depth=2,      # 频率维度深度（增加以提高质量）
    num_stems=NUM_STEMS,          # 6个音轨
    stereo=True,                   # 立体声
    stft_hop_length=256,           # 更小的跳跃长度，更好的重建质量
    flash_attn=True               # 使用 Flash Attention 加速
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
        # 加载音频（尝试多种方式）
        print("  加载音频...")
        mix = None
        sr = None
        
        # 方法1: 尝试使用 torchaudio（更可靠）
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            # 转换为 numpy 数组
            mix = waveform.numpy()
            sr = sample_rate
            
            # 如果是单声道，转换为立体声
            if len(mix.shape) == 1:
                mix = np.stack([mix, mix])
            elif mix.shape[0] == 1:
                mix = np.stack([mix[0], mix[0]])
            
            # 重采样到目标采样率
            if sr != SAMPLE_RATE:
                print(f"    重采样: {sr} Hz -> {SAMPLE_RATE} Hz")
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform_resampled = resampler(torch.from_numpy(mix))
                mix = waveform_resampled.numpy()
                sr = SAMPLE_RATE
            
            # 确保格式为 (channels, samples)
            if mix.shape[0] > mix.shape[1]:
                mix = mix.T
            print(f"    ✓ 使用 torchaudio 加载成功")
            
        except Exception as e1:
            print(f"    torchaudio 加载失败: {e1}")
            # 方法2: 尝试使用 librosa
            try:
                mix, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=False)
                print(f"    ✓ 使用 librosa 加载成功")
            except Exception as e2:
                print(f"  ✗ 所有音频加载方法都失败")
                print(f"    torchaudio 错误: {e1}")
                print(f"    librosa 错误: {e2}")
                print(f"  提示: 文件可能已损坏或格式不支持")
                print(f"  跳过此文件，继续处理下一个...")
                continue
        
        duration = len(mix[0]) / sr if len(mix.shape) > 1 else len(mix) / sr
        print(f"  音频信息: {mix.shape}, 采样率: {sr} Hz, 时长: {duration:.2f} 秒")
        
        # 转换为张量格式 (batch, channels, samples)
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix])  # 单声道转立体声
        
        # 分段处理（避免 GPU 内存不足）
        total_samples = mix.shape[1]
        segment_samples = int(SEGMENT_LENGTH * SAMPLE_RATE)
        overlap_samples = int(OVERLAP * SAMPLE_RATE)
        
        if total_samples > segment_samples:
            print(f"  音频较长 ({duration:.1f}秒)，将分段处理（每段 {SEGMENT_LENGTH} 秒）...")
            all_separated = []
            
            for start_idx in range(0, total_samples, segment_samples - overlap_samples):
                end_idx = min(start_idx + segment_samples, total_samples)
                segment = mix[:, start_idx:end_idx]
                
                # 如果最后一段太短，从后往前取
                if end_idx - start_idx < segment_samples // 2 and start_idx > 0:
                    start_idx = max(0, total_samples - segment_samples)
                    end_idx = total_samples
                    segment = mix[:, start_idx:end_idx]
                
                segment_tensor = torch.from_numpy(segment).float().unsqueeze(0).to(DEVICE)
                
                # 清理 GPU 缓存
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                
                print(f"    处理段: {start_idx/SAMPLE_RATE:.1f}s - {end_idx/SAMPLE_RATE:.1f}s")
                with torch.no_grad():
                    segment_separated = model(segment_tensor)
                
                # 转换到 CPU 并保存
                segment_separated = segment_separated.cpu()
                all_separated.append(segment_separated)
            
            # 拼接所有分段
            print(f"  拼接 {len(all_separated)} 个分段...")
            separated = torch.cat(all_separated, dim=-1)
            # 裁剪到原始长度（去除重叠部分）
            if separated.shape[-1] > total_samples:
                separated = separated[..., :total_samples]
        else:
            # 短音频直接处理
            mix_tensor = torch.from_numpy(mix).float().unsqueeze(0).to(DEVICE)
            print(f"  输入张量形状: {mix_tensor.shape}")
            print("  执行音频分离...")
            
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                separated = model(mix_tensor)  # 输出形状: (batch, stems, channels, samples)
                separated = separated.cpu()
        
        print(f"  分离结果形状: {separated.shape}")
        
        # 保存结果
        track_name = Path(audio_file).stem
        track_output_dir = os.path.join(OUTPUT_DIR, track_name)
        os.makedirs(track_output_dir, exist_ok=True)
        
        print(f"  保存结果到: {track_output_dir}/")
        for i, stem_name in enumerate(STEM_NAMES):
            stem_audio = separated[0, i].numpy()  # 移除batch维度（已在CPU）
            output_path = os.path.join(track_output_dir, f"{stem_name}.wav")
            sf.write(output_path, stem_audio.T, sr)
            print(f"    ✓ {stem_name}.wav")
        
        # 清理 GPU 缓存
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
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

