# Auto Jackal - 强化学习自动玩NES游戏

基于PyTorch和PPO算法的强化学习系统，用于训练AI自动玩NES游戏（Jackal）。

## 功能特点

- 🎮 使用OpenAI Retro环境运行NES游戏
- 🧠 基于PyTorch实现PPO（Proximal Policy Optimization）算法
- 📊 实时可视化训练过程（奖励、损失、趋势等）
- 🎨 **彩色游戏画面显示（RGB真彩色，非灰度图）**
- 🖼️ **独立实时模拟窗口（大画面观看训练过程）**
- 🎯 **Sigmoid多标签输出（允许同时按多个按钮，512种组合）**
- 💾 智能checkpoint管理系统（自动保存/加载/清理）
- 🔄 支持训练中断后自动恢复
- ⚡ **多线程并行训练（8个模拟器同时运行，速度提升8倍）**
- 🎲 每个模拟器使用不同的随机种子，增加探索多样性
- 🚀 **跳帧处理（每4帧决策一次，训练速度再提升4倍）**
- 🎮 支持训练好的模型进行游戏演示

## 项目结构

```
.
├── model.py               # Actor-Critic神经网络模型
├── ppo_agent.py           # PPO算法实现
├── env_wrapper.py         # 游戏环境预处理包装器
├── train.py               # 多进程并行训练+单窗口渲染（推荐）
├── train_parallel_visual.py # 多线程训练+8画面实时显示
├── play.py                # 使用训练好的模型玩游戏
├── test.py                # 原始测试脚本
├── test_checkpoint.py     # Checkpoint功能测试
├── test_parallel.py       # 并行训练测试
├── test_frame_skip.py     # 跳帧功能测试
├── test_color_display.py  # 彩色显示功能测试
├── test_sigmoid.py        # Sigmoid多标签输出测试
├── test_reward_penalty.py # 奖励惩罚机制测试
├── COLOR_DISPLAY_GUIDE.md # 彩色显示功能详解
├── SIGMOID_VS_SOFTMAX.md  # Sigmoid vs Softmax对比说明
└── requirements.txt       # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

#### 方式A：多进程并行训练+单窗口渲染（推荐！🔥）

使用多个worker并行收集经验，同时保留一个窗口显示游戏画面：

```bash
python train.py
```

**新功能：**
- ⚡ **多进程并行**：使用config.N_WORKERS个worker并行收集经验（默认8个）
- 🎮 **单窗口渲染**：主进程保留一个环境用于显示游戏画面
- 📊 实时训练统计图表
- 💾 自动checkpoint管理

**优势：**
- ⚡ 训练速度提升约8倍（8个worker）
- 🎮 可以实时观看游戏画面（不影响训练速度）
- 🎲 每个worker使用不同的随机种子，增加探索多样性
- 📊 实时显示训练统计（奖励、步数、损失等）
- 💾 自动checkpoint管理，支持中断恢复

**配置：**
在`config.py`中调整参数：
```python
N_WORKERS = 8           # worker数量（建议等于CPU核心数）
RENDER_INTERVAL = 5     # 每5个episode渲染一次
FRAME_SKIP = 4          # 跳帧数
```

**注意：**
- 需要较多CPU资源（建议8核以上）
- 内存占用约2-4GB
- 渲染在主进程运行，不会阻塞worker进程

#### 方式B：多线程并行训练+8画面实时显示

同时显示8个worker的**彩色**游戏画面，实时观看训练过程：

```bash
python train_parallel_visual.py
```

**新功能：**
- ✨ **彩色RGB画面**：显示真实的游戏彩色画面，而非灰度图
- 🖼️ **独立实时窗口**：额外的大窗口显示Worker 0的实时游戏画面
- 📊 8个小窗口 + 4个训练图表 + 1个大窗口实时模拟
- 🎮 完整的训练可视化体验

**优势：**
- 🎮 **同时显示8个游戏画面**，实时观看所有worker的游戏过程
- ⚡ 8个模拟器并行运行，训练速度提升约8倍
- 🎲 每个worker使用不同的随机种子，增加探索多样性
- 📊 实时显示训练统计（奖励、步数、损失等）
- 💾 自动checkpoint管理，支持中断恢复

**画面布局：**
```
┌─────────────────────────────────────────┐
│  Worker0  Worker1  Worker2  Worker3    │  ← 上排4个游戏画面
│  Worker4  Worker5  Worker6  Worker7    │  ← 下排4个游戏画面
├─────────────────────────────────────────┤
│  奖励曲线  │  Episode长度              │  ← 训练统计图表
│  训练损失  │  奖励趋势                 │
└─────────────────────────────────────────┘
```

**注意：**
- 需要较多CPU资源（建议8核以上）
- 内存占用约2-4GB
- 可视化会略微降低速度（约10-15%），但能直观看到训练过程

#### 方式B：多线程并行训练+8画面实时显示

同时显示8个worker的**彩色**游戏画面，实时观看训练过程：

```bash
python train_parallel_visual.py
```

**新功能：**
- ✨ **彩色RGB画面**：显示真实的游戏彩色画面，而非灰度图
- 🖼️ **独立实时窗口**：额外的大窗口显示Worker 0的实时游戏画面
- 📊 8个小窗口 + 4个训练图表 + 1个大窗口实时模拟
- 🎮 完整的训练可视化体验

**优势：**
- 🎮 **同时显示8个游戏画面**，实时观看所有worker的游戏过程
- ⚡ 8个模拟器并行运行，训练速度提升约8倍
- 🎲 每个worker使用不同的随机种子，增加探索多样性
- 📊 实时显示训练统计（奖励、步数、损失等）
- 💾 自动checkpoint管理，支持中断恢复

**画面布局：**
```
┌─────────────────────────────────────────┐
│  Worker0  Worker1  Worker2  Worker3    │  ← 上排4个游戏画面
│  Worker4  Worker5  Worker6  Worker7    │  ← 下排4个游戏画面
├─────────────────────────────────────────┤
│  奖励曲线  │  Episode长度              │  ← 训练统计图表
│  训练损失  │  奖励趋势                 │
└─────────────────────────────────────────┘
```

**注意：**
- 需要较多CPU资源（建议8核以上）
- 内存占用约2-4GB
- 可视化会略微降低速度（约10-15%），但能直观看到训练过程

#### 方式C：纯速度优化（无渲染）

不显示游戏画面，只显示训练统计，速度最快。

**注意：** 如果你的项目中有`train_parallel.py`文件，可以使用它。否则，可以在`config.py`中设置`RENDER=False`来禁用渲染。

#### Checkpoint管理

训练系统具有智能checkpoint管理功能：

- **自动保存**：每10个episode保存一次完整的训练状态
- **自动加载**：启动训练时自动检测并加载最新的checkpoint
- **自动清理**：保留最新的100个checkpoint，自动删除旧的
- **完整状态**：保存模型参数、优化器状态、训练统计等

Checkpoint文件命名格式：`checkpoints/checkpoint_<episode>.pth`

测试checkpoint功能：
```bash
python test_checkpoint.py
```

### 2. 中断后恢复训练

如果训练中断，只需再次运行 `python train.py`，系统会：
1. 自动检测最新的checkpoint
2. 加载模型参数和优化器状态
3. 恢复episode计数和训练统计
4. 从中断点继续训练

```bash
# 示例输出
Loading checkpoint: checkpoints/checkpoint_230.pth
Resumed from episode 230, global step 115000
Loaded 230 episode records
Training on device: cuda
Starting from episode: 230
```

### 3. 使用训练好的模型玩游戏

```bash
python play.py checkpoints/checkpoint_100.pth
```

或者直接运行（默认加载最新的checkpoint）：

```bash
python play.py
```

## 算法说明

### PPO (Proximal Policy Optimization)

- **优势**：稳定、高效、易于调参
- **核心思想**：限制策略更新幅度，避免训练崩溃
- **关键技术**：
  - Clipped surrogate objective
  - Generalized Advantage Estimation (GAE)
  - Actor-Critic架构

### 网络架构

- **输入**：4帧堆叠的灰度图像 (4, 84, 84)
- **卷积层**：3层CNN提取视觉特征
- **Actor头**：输出动作概率分布
- **Critic头**：输出状态价值估计

### 环境预处理

- 灰度化：RGB → 灰度
- 缩放：原始分辨率 → 84x84
- 帧堆叠：连续4帧作为输入
- **跳帧处理**：每4帧执行一次动作（可配置）
- **动作空间转换**：将Retro的MultiBinary动作空间转换为离散动作空间，包含常用的按钮组合

### 奖励塑形机制

为了鼓励AI更积极地探索和获取奖励，添加了以下机制：

- **无奖励惩罚**：每个没有获得正奖励的step，给予小额惩罚（默认-0.01）
- **超时机制**：如果连续450步（约30秒）没有获得奖励，强制结束episode并给予额外惩罚（-1.0）
- **奖励跟踪**：自动跟踪无奖励步数，获得正奖励时重置计数器

这些机制可以有效避免AI陷入无效的行为模式，加速训练收敛。

## 超参数配置

可以在`train.py`中调整以下参数：

```python
# Trainer参数
save_interval = 10      # checkpoint保存间隔（episode）
max_checkpoints = 100   # 最多保留的checkpoint数量
frame_skip = 4          # 跳帧数（每N帧决策一次）
max_checkpoints = 100   # 最多保留的checkpoint数量

# PPO参数
lr = 3e-4              # 学习率
gamma = 0.99           # 折扣因子
gae_lambda = 0.95      # GAE lambda
clip_epsilon = 0.2     # PPO裁剪参数
c1 = 0.5               # 价值损失系数
c2 = 0.01              # 熵奖励系数

# 训练参数
max_episodes = 1000    # 最大训练episode数
max_steps = 10000      # 每个episode最大步数
update_interval = 2048 # 更新策略的步数间隔

# 奖励惩罚参数
no_reward_penalty = -0.01      # 无奖励惩罚
no_reward_timeout_steps = 450  # 超时步数
```

## 训练技巧

1. **初期训练**：前几百个episode可能奖励很低，这是正常的
2. **调整更新频率**：如果训练不稳定，可以增加`update_interval`
3. **学习率调整**：如果收敛太慢，可以适当增加学习率
4. **奖励工程**：可以在`env_wrapper.py`中修改奖励函数

## 性能优化

### 单线程优化
- 自动使用GPU（如果可用）
- 批量处理训练数据
- 梯度裁剪防止梯度爆炸
- 优势归一化提高稳定性

### 多线程并行优化
- **8个worker并行采样**：同时运行8个游戏环境，数据收集速度提升8倍
- **独立随机种子**：每个worker使用不同的随机种子（`worker_id * 1000 + timestamp`），确保探索多样性
- **CPU并行**：worker在CPU上运行环境，避免GPU内存瓶颈
- **GPU集中更新**：所有worker的数据汇总后，在GPU上统一更新策略
- **异步通信**：使用multiprocessing.Queue实现高效的进程间通信

### 性能对比

| 训练方式 | 速度 | CPU占用 | 内存占用 | 游戏画面 | 适用场景 |
|---------|------|---------|----------|---------|----------|
| train.py (8 workers) | ~8x | 高 | ~2-4GB | 单窗口渲染 | **推荐：快速训练+观察** |
| train.py (8 workers + 跳帧4) | ~32x | 高 | ~2-4GB | 单窗口渲染 | **最快+可观察** |
| train_parallel_visual.py | ~7x | 高 | ~2-4GB | 8个实时画面 | 全面监控所有worker |
| 单worker (旧版) | 1x | 低 | ~256MB | 无 | 资源受限、调试 |

**注意**：
- 跳帧处理（`FRAME_SKIP=4`）可以显著提升训练速度，但可能略微影响控制精度
- 对于大多数NES游戏，`FRAME_SKIP=4`是速度和精度的最佳平衡点
- `train.py`的渲染在主进程运行，不会阻塞worker进程，几乎不影响训练速度

## 常见问题

**Q: 训练很慢怎么办？**
A: 使用`train_parallel.py`进行多线程并行训练，速度可提升约8倍。确保安装了CUDA版本的PyTorch，策略更新会自动使用GPU加速。

**Q: 如何调整可视化频率？**
A: 在`train.py`中修改`episode % 5 == 0`中的数字。

**Q: Checkpoint保存在哪里？**
A: Checkpoint保存在`checkpoints/`目录下，文件名格式为`checkpoint_<episode>.pth`。

**Q: 如何修改checkpoint保存频率？**
A: 在创建Trainer时修改`save_interval`参数，例如`save_interval=20`表示每20个episode保存一次。

**Q: 训练中断后如何恢复？**
A: 直接运行`python train.py`，系统会自动检测并加载最新的checkpoint。

**Q: 如何从头开始训练？**
A: 删除或重命名`checkpoints/`目录，然后运行`python train.py`或`python train_parallel.py`。

**Q: 多线程训练需要什么配置？**
A: 建议8核以上CPU，2-4GB内存。如果资源不足，可以在`train_parallel.py`中修改`n_workers`参数（例如改为4）。

**Q: 如何测试并行训练是否正常工作？**
A: 运行`python test_parallel.py`，会测试8个worker是否使用了不同的随机种子。

**Q: 多线程训练时如何调整worker数量？**
A: 在`train_parallel.py`的`main()`函数中修改`n_workers`参数，例如`n_workers=4`表示使用4个worker。

## 扩展建议

- 尝试其他强化学习算法（A3C, SAC, DQN等）
- 添加课程学习（Curriculum Learning）
- 实现分布式训练
- 添加TensorBoard日志
- 尝试其他NES游戏

## 许可证

MIT License
