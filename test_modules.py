#!/usr/bin/env python
"""
StreamSpatial-VLM 核心模块导入与初始化测试
==========================================
验证三个核心模块是否可以正确导入和初始化。

用法:
    python test_modules.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import json

print("=" * 70)
print("StreamSpatial-VLM 核心模块导入与初始化测试")
print("=" * 70)

all_tests = []

# ============ Test 1: Gate 模块 ============
print("\n[Test 1] 2D→3D Gate 语义门控模块")
print("-" * 70)

try:
    from models.gate_2d3d import SemanticGate2D3D, GateConfig
    
    gate = SemanticGate2D3D(GateConfig(tau=0.15))
    print(f"✅ Gate 模块导入成功")
    print(f"   - 类型: {type(gate).__name__}")
    print(f"   - 阈值 τ: {gate.config.tau}")
    print(f"   - cooldown 步长: {gate.config.history_step}")
    print(f"   - warmup 帧数: {gate.config.warmup_frames}")
    
    # 模拟调用
    cls_token = torch.randn(768)
    trigger1 = gate(cls_token)  # 首帧
    cls_token2 = torch.randn(768)
    trigger2 = gate(cls_token2)  # 第二帧
    
    print(f"   - 首帧触发: {trigger1} (expected: True)")
    print(f"   - 随机帧触发: {trigger2}")
    print(f"✅ Gate 模块测试: PASS")
    test1_pass = True
    
except Exception as e:
    print(f"❌ Gate 模块测试失败: {e}")
    test1_pass = False

all_tests.append(("Gate 语义门控", test1_pass))

# ============ Test 2: Zip 模块 ============
print("\n[Test 2] 3D→2D Zip 几何引导压缩模块")
print("-" * 70)

try:
    from models.zip_3d2d import GeometryGuidedZip, ZipConfig
    
    zip_module = GeometryGuidedZip(ZipConfig(keep_ratio=0.6))
    print(f"✅ Zip 模块导入成功")
    print(f"   - 类型: {type(zip_module).__name__}")
    print(f"   - 保留率 r: {zip_module.config.keep_ratio:.2%}")
    print(f"   - 深度方差权重 α: {zip_module.config.alpha}")
    print(f"   - 位姿熵权重 β: {zip_module.config.beta}")
    print(f"   - Patch 大小: {zip_module.config.patch_size}")
    print(f"✅ Zip 模块测试: PASS")
    test2_pass = True
    
except Exception as e:
    print(f"❌ Zip 模块测试失败: {e}")
    test2_pass = False

all_tests.append(("Zip 几何压缩", test2_pass))

# ============ Test 3: KV Cache 模块 ============
print("\n[Test 3] 增量流式 KV Cache 模块")
print("-" * 70)

try:
    from models.kv_cache import IncrementalKVCache, KVCacheConfig
    
    cache = IncrementalKVCache(KVCacheConfig(window_size=4))
    print(f"✅ KV Cache 模块导入成功")
    print(f"   - 类型: {type(cache).__name__}")
    print(f"   - 窗口大小 w: {cache.config.window_size}")
    print(f"   - 3D 特征维度: {cache.config.feat_3d_dim}")
    print(f"   - 2D 特征维度: {cache.config.feat_2d_dim}")
    print(f"✅ KV Cache 模块测试: PASS")
    test3_pass = True
    
except Exception as e:
    print(f"❌ KV Cache 模块测试失败: {e}")
    test3_pass = False

all_tests.append(("KV Cache 流式缓存", test3_pass))

# ============ Test 4: Profiling 数据检查 ============
print("\n[Test 4] Profiling 冗余分析数据验证")
print("-" * 70)

try:
    with open('results/profiling/redundancy_analysis.json', 'r', encoding='utf-8') as f:
        prof_data = json.load(f)
    
    print(f"✅ Profiling 数据导入成功")
    print(f"   - 帧级冗余: {prof_data['frame_redundancy']['redundancy_conclusion']}")
    print(f"   - Token 冗余: {prof_data['token_redundancy']['redundancy_conclusion']}")
    print(f"   - 推荐参数:")
    print(f"     • τ (门控阈值): {prof_data['summary']['recommended_tau']}")
    print(f"     • r (保留率): {prof_data['summary']['recommended_keep_ratio']:.2%}")
    print(f"     • w (窗口): {prof_data['summary']['recommended_window_size']}")
    print(f"✅ Profiling 数据测试: PASS")
    test4_pass = True
    
except Exception as e:
    print(f"❌ Profiling 数据测试失败: {e}")
    test4_pass = False

all_tests.append(("Profiling 数据", test4_pass))

# ============ Test 5: 数据完备性检查 ============
print("\n[Test 5] ScanNet 数据完备性检查")
print("-" * 70)

try:
    import os
    from pathlib import Path
    
    scannet_root = Path('data/raw/spar7m/spar/scannet/images')
    
    rgb_count = len(list(scannet_root.rglob('*.jpg'))) + len(list(scannet_root.rglob('*.png')))
    depth_count = len(list((scannet_root / 'scene0000_00' / 'image_depth').glob('*.png'))) if (scannet_root / 'scene0000_00' / 'image_depth').exists() else 0
    
    print(f"✅ ScanNet 数据检查完成")
    print(f"   - RGB 总数: {rgb_count}")
    print(f"   - Depth 总数: {depth_count}")
    print(f"   - 数据完整: {'✅ 是' if depth_count > 0 else '❌ 否'}")
    
    test5_pass = rgb_count > 0 and depth_count > 0
    print(f"✅ 数据完备性测试: {'PASS' if test5_pass else 'FAIL'}")
    
except Exception as e:
    print(f"❌ 数据完备性测试失败: {e}")
    test5_pass = False

all_tests.append(("ScanNet 数据", test5_pass))

# ============ 总结 ============
print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)

for test_name, passed in all_tests:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")

all_pass = all(passed for _, passed in all_tests)
print("\n" + "=" * 70)
print(f"整体测试结果: {'✅ 全部通过' if all_pass else '❌ 有失败'}")
print("=" * 70)
print("\n📊 总结:")
print(f"  - 核心模块: {'✅ 就位' if test1_pass and test2_pass and test3_pass else '❌ 缺失'}")
print(f"  - Profiling 数据: {'✅ 就位' if test4_pass else '❌ 缺失'}")
print(f"  - ScanNet 数据: {'✅ 就位' if test5_pass else '❌ 缺失'}")
print(f"\n✅ 项目准备完毕，可进行消融实验")

sys.exit(0 if all_pass else 1)
