# Checkpoint管理系统使用说明

## 功能概述

训练系统现在具有完整的checkpoint管理功能：
- ✅ 每10个episode自动保存checkpoint
- ✅ 启动时自动加载最新checkpoint
- ✅ 最多保留100个checkpoint，自动清理旧的
- ✅ 保存完整训练状态（模型、优化器、统计数据）

## 快速开始

### 1. 开始新训练
```bash
python train.py
```

输出示例：
```
No checkpoint found. Starting from scratch.
Training on device: cuda
Action space: 9
Starting from episode: 0
Global step: 0
```

### 2. 中断后恢复训练
如果训练中断（Ctrl+C或意外关闭），只需再次运行：
```bash
python train.py
```

输出示例：
```
Loading checkpoint: checkpoints/checkpoint_230.pth
Resumed from episode 230, global step 115000
Loaded 230 episode records
Training on device: cuda
Starting from episode: 230
Global step: 115000
```

### 3. 测试checkpoint功能
```bash
python test_checkpoint.py
```

这会显示：
- 当前所有checkpoint文件
- 最新checkpoint信息
- 自动加载测试
- 清理机制状态

## Checkpoint文件结构

每个checkpoint包含：
```python
{
    'policy_state_dict': ...,      # 模型参数
    'optimizer_state_dict': ...,   # 优化器状态
    'episode': 230,                # 当前episode
    'global_step': 115000,         # 全局步数
    'training_stats': {            # 训练统计
        'episode_rewards': [...],
        'episode_lengths': [...],
        'losses': [...]
    }
}
```

## 自定义配置

在 `train.py` 的 `main()` 函数中修改：

```python
trainer = Trainer(
    game='Jackal-Nes',
    render=True,
    save_interval=10,      # 每10个episode保存一次
    max_checkpoints=100    # 最多保留100个checkpoint
)
```

### 常用配置示例

**快速测试（频繁保存）：**
```python
save_interval=5,        # 每5个episode保存
max_checkpoints=20      # 只保留20个
```

**长期训练（节省空间）：**
```python
save_interval=50,       # 每50个episode保存
max_checkpoints=50      # 只保留50个
```

**完整记录（保留所有）：**
```python
save_interval=10,
max_checkpoints=10000   # 实际上不会删除
```

## 文件管理

### Checkpoint文件位置
```
checkpoints/
├── checkpoint_10.pth
├── checkpoint_20.pth
├── checkpoint_30.pth
...
└── checkpoint_1000.pth
```

### 手动管理

**查看所有checkpoint：**
```bash
ls -lh checkpoints/
```

**删除特定checkpoint：**
```bash
rm checkpoints/checkpoint_100.pth
```

**从头开始训练：**
```bash
rm -rf checkpoints/
python train.py
```

**从特定checkpoint恢复：**
```bash
# 删除比它新的checkpoint，系统会自动加载最新的
rm checkpoints/checkpoint_[2-9]*.pth
python train.py
```

## 最佳实践

### 1. 定期备份重要checkpoint
```bash
# 备份表现好的checkpoint
cp checkpoints/checkpoint_500.pth backups/best_model_500.pth
```

### 2. 监控磁盘空间
每个checkpoint约10-20MB，100个约1-2GB

### 3. 训练策略
- 初期训练：`save_interval=10`，快速迭代
- 稳定后：`save_interval=50`，节省空间
- 接近收敛：`save_interval=100`，保留关键点

### 4. 多次实验
```bash
# 为不同实验创建不同目录
mkdir -p experiments/exp1/checkpoints
mkdir -p experiments/exp2/checkpoints

# 修改train.py中的checkpoint_dir
self.checkpoint_dir = 'experiments/exp1/checkpoints'
```

## 故障排除

### 问题1：加载checkpoint失败
```
Failed to load checkpoint: ...
Starting from scratch.
```

**解决方案：**
- 检查checkpoint文件是否损坏
- 删除损坏的文件，系统会加载上一个
- 或删除所有checkpoint从头开始

### 问题2：磁盘空间不足
```
OSError: [Errno 28] No space left on device
```

**解决方案：**
- 减少 `max_checkpoints` 数量
- 增加 `save_interval` 间隔
- 手动删除旧的checkpoint

### 问题3：想从特定episode重新开始
**解决方案：**
```bash
# 保留到episode 200的checkpoint，删除之后的
rm checkpoints/checkpoint_[2-9][1-9]*.pth
rm checkpoints/checkpoint_[3-9]*.pth
python train.py  # 会从checkpoint_200.pth恢复
```

## 高级用法

### 1. 编程方式加载checkpoint
```python
from ppo_agent import PPOAgent

# 创建agent
agent = PPOAgent(input_shape=(4, 84, 84), n_actions=9)

# 加载特定checkpoint
checkpoint = agent.load('checkpoints/checkpoint_500.pth')

print(f"Loaded episode: {checkpoint['episode']}")
print(f"Global step: {checkpoint['global_step']}")
```

### 2. 导出最佳模型
```python
import torch

# 加载checkpoint
checkpoint = torch.load('checkpoints/checkpoint_500.pth')

# 只保存模型参数（更小的文件）
torch.save(
    checkpoint['policy_state_dict'],
    'best_model.pth'
)
```

### 3. 分析训练历史
```python
import torch
import matplotlib.pyplot as plt

checkpoint = torch.load('checkpoints/checkpoint_500.pth')
stats = checkpoint['training_stats']

plt.plot(stats['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()
```

## 总结

Checkpoint管理系统让你可以：
- 🔄 随时中断和恢复训练
- 💾 自动管理存储空间
- 📊 保留完整训练历史
- 🎯 回退到任意训练点
- 🚀 无缝继续长期训练

现在就开始训练吧！🎮
