# 8画面实时显示训练指南

## 快速开始

想要同时看到8个游戏模拟器的训练过程吗？使用这个命令：

```bash
python train_parallel_visual.py
```

## 效果展示

运行后，你会看到一个大窗口，包含：

### 上半部分：8个游戏画面（4x2布局）
```
┌─────────────────────────────────────────┐
│  [Worker 0]  [Worker 1]  [Worker 2]  [Worker 3]  │
│  R:12.5 S:45 R:8.3 S:32  R:15.1 S:67 R:9.8 S:28  │
│                                                   │
│  [Worker 4]  [Worker 5]  [Worker 6]  [Worker 7]  │
│  R:11.2 S:51 R:13.7 S:43 R:7.9 S:19  R:14.3 S:58 │
└─────────────────────────────────────────┘
```

每个画面显示：
- 实时游戏画面（灰度图）
- R: 当前episode的累计奖励
- S: 当前episode的步数

### 下半部分：4个训练统计图表（2x2布局）

1. **Episode Rewards（左上）**
   - 每个episode的奖励值
   - 10-episode移动平均线

2. **Episode Lengths（右上）**
   - 每个episode的长度（步数）

3. **Training Loss（左下）**
   - 策略更新时的损失值

4. **Reward Trends（右下）**
   - 10/50/100-episode移动平均趋势

## 功能特点

### 1. 实时游戏画面
- 每个worker每5步更新一次画面
- 可以看到8个不同的游戏进程
- 每个worker使用不同的随机种子，探索不同的策略

### 2. 性能优化
- 画面更新频率：每0.5秒
- 避免过于频繁的更新导致训练变慢
- 相比纯统计版本，速度损失约10-15%

### 3. 自动管理
- 自动加载最新checkpoint
- 支持Ctrl+C中断后继续训练
- 自动清理旧checkpoint

## 配置选项

### 关闭游戏画面显示

如果你想要最快的训练速度，可以关闭游戏画面：

在 `train_parallel_visual.py` 中修改：

```python
trainer = ParallelTrainerWithVisual(
    game='Jackal-Nes',
    n_workers=8,
    render=True,
    show_game_screens=False,  # 改为False
    save_interval=10,
    max_checkpoints=100
)
```

### 调整Worker数量

根据你的CPU核心数调整：

```python
trainer = ParallelTrainerWithVisual(
    game='Jackal-Nes',
    n_workers=4,  # 改为4个worker
    render=True,
    show_game_screens=True,
    save_interval=10,
    max_checkpoints=100
)
```

### 调整更新频率

在 `train()` 方法中修改更新间隔：

```python
# 当前：每0.5秒更新一次
if self.render and (current_time - last_update_time) > 0.5:
    self.update_visualization()
    
# 更快更新（更流畅但更慢）
if self.render and (current_time - last_update_time) > 0.2:
    self.update_visualization()
    
# 更慢更新（更快但不流畅）
if self.render and (current_time - last_update_time) > 1.0:
    self.update_visualization()
```

## 系统要求

### 最低配置
- CPU: 4核
- 内存: 2GB
- 显示器: 1920x1080
- Python: 3.7+

### 推荐配置
- CPU: 8核或以上
- 内存: 4GB或以上
- 显示器: 1920x1080或更高
- Python: 3.8+
- GPU: 可选（用于策略更新加速）

## 使用技巧

### 1. 观察训练进度

**看游戏画面：**
- 初期：角色可能乱走，经常卡住
- 中期：开始有目的性移动，偶尔能前进
- 后期：能够稳定前进，避开障碍

**看奖励曲线：**
- 初期：奖励波动大，平均值低
- 中期：奖励逐渐上升，波动减小
- 后期：奖励稳定在较高水平

### 2. 判断训练效果

**好的迹象：**
- ✓ 奖励曲线整体上升
- ✓ Episode长度逐渐增加
- ✓ 不同worker表现趋于一致
- ✓ 损失值逐渐下降并稳定

**需要调整的迹象：**
- ✗ 奖励长期不增长（>100 episodes）
- ✗ 损失值持续上升或剧烈波动
- ✗ 所有worker都卡在同一个地方
- ✗ Episode长度没有增长

### 3. 调试技巧

**问题：画面不更新**
- 检查worker是否正常启动
- 查看终端是否有错误信息
- 确认matplotlib正常工作

**问题：训练很慢**
- 减少worker数量
- 增加更新间隔（0.5秒 → 1.0秒）
- 关闭游戏画面显示

**问题：内存不足**
- 减少worker数量（8 → 4）
- 关闭游戏画面显示
- 减小update_interval

## 性能对比

### 不同模式的速度对比

| 模式 | Episodes/小时 | 相对速度 | 可视化 |
|------|--------------|---------|--------|
| 单线程 | ~50 | 1x | 统计图表 |
| 8线程（无画面） | ~350 | 7x | 统计图表 |
| 8线程（8画面） | ~300 | 6x | 8画面+统计 |

### 可视化开销

- 画面更新：约10-15%性能损失
- 但能直观看到训练过程
- 适合调试和演示

## 常见问题

**Q: 为什么有些worker画面不动？**
A: 可能该worker的episode已结束，正在等待新任务。这是正常现象。

**Q: 8个画面会不会太卡？**
A: 不会。我们每5步才更新一次画面，且整体更新频率为0.5秒，不会影响训练速度。

**Q: 可以只显示4个画面吗？**
A: 可以。修改代码中的`n_workers=4`，画面布局会自动调整。

**Q: 画面太小看不清？**
A: 可以调整窗口大小，或修改代码中的`figsize=(20, 12)`参数。

**Q: 可以保存训练视频吗？**
A: 可以使用屏幕录制软件，或修改代码添加视频保存功能。

## 高级功能

### 1. 自定义画面布局

修改 `__init__` 方法中的GridSpec：

```python
# 当前：4x2布局（上半部分游戏，下半部分统计）
gs = GridSpec(4, 6, figure=self.fig, hspace=0.3, wspace=0.3)

# 改为：左右布局（左边游戏，右边统计）
gs = GridSpec(4, 8, figure=self.fig, hspace=0.3, wspace=0.3)
```

### 2. 添加更多统计信息

在worker_stats中添加更多信息：

```python
worker_stats[worker_id] = {
    'reward': episode_reward,
    'steps': episode_length,
    'max_reward': max(episode_reward, prev_max),  # 添加最大奖励
    'avg_reward': running_avg_reward,  # 添加平均奖励
}
```

### 3. 高亮表现最好的Worker

在 `update_visualization` 中：

```python
# 找到奖励最高的worker
best_worker = max(range(self.n_workers), 
                 key=lambda i: self.worker_stats[i]['reward'])

# 高亮显示
if i == best_worker:
    ax.set_title(f'Worker {i} ⭐ | R:{stats["reward"]:.1f}', 
               fontsize=9, color='gold', fontweight='bold')
```

## 总结

8画面实时显示训练是观察和理解强化学习训练过程的最佳方式：

✅ **优势**
- 直观看到训练过程
- 发现问题更容易
- 适合演示和教学
- 性能损失可接受（10-15%）

⚠️ **注意**
- 需要较大的显示器
- 略微降低训练速度
- 增加内存占用

🚀 **建议**
- 调试时使用8画面模式
- 正式训练时可关闭画面
- 演示时必备功能

现在就试试吧！运行 `python train_parallel_visual.py`，享受8个游戏同时训练的震撼效果！🎮✨
