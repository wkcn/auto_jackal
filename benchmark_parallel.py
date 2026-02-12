import time
import torch.multiprocessing as mp
import numpy as np
from env_wrapper import RetroWrapper


def single_thread_benchmark(n_episodes=10):
    """单线程基准测试"""
    print("=" * 70)
    print("单线程训练基准测试")
    print("=" * 70)
    
    env = RetroWrapper(game='Jackal-Nes')
    
    start_time = time.time()
    total_steps = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_steps = 0
        
        for step in range(1000):  # 限制每个episode最多1000步
            action = np.random.randint(0, env.n_actions)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        print(f"Episode {episode + 1}/{n_episodes} completed, steps: {episode_steps}")
    
    elapsed_time = time.time() - start_time
    env.close()
    
    print(f"\n总耗时: {elapsed_time:.2f}秒")
    print(f"总步数: {total_steps}")
    print(f"平均每episode: {elapsed_time / n_episodes:.2f}秒")
    print(f"步数/秒: {total_steps / elapsed_time:.2f}")
    
    return elapsed_time, total_steps


def worker_benchmark(worker_id, n_episodes, result_queue):
    """Worker基准测试"""
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
    
    env = RetroWrapper(game='Jackal-Nes')
    
    total_steps = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_steps = 0
        
        for step in range(1000):
            action = np.random.randint(0, env.n_actions)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
    
    env.close()
    result_queue.put((worker_id, total_steps))


def parallel_benchmark(n_workers=8, episodes_per_worker=10):
    """多线程基准测试"""
    print("\n" + "=" * 70)
    print(f"多线程训练基准测试 ({n_workers} workers)")
    print("=" * 70)
    
    mp.set_start_method('spawn', force=True)
    
    result_queue = mp.Queue()
    workers = []
    
    start_time = time.time()
    
    # 启动workers
    for worker_id in range(n_workers):
        p = mp.Process(
            target=worker_benchmark,
            args=(worker_id, episodes_per_worker, result_queue)
        )
        p.start()
        workers.append(p)
        print(f"Worker {worker_id} started")
    
    # 收集结果
    results = []
    for _ in range(n_workers):
        result = result_queue.get()
        results.append(result)
    
    # 等待所有workers完成
    for p in workers:
        p.join()
    
    elapsed_time = time.time() - start_time
    
    total_steps = sum(steps for _, steps in results)
    total_episodes = n_workers * episodes_per_worker
    
    print(f"\n总耗时: {elapsed_time:.2f}秒")
    print(f"总步数: {total_steps}")
    print(f"总episodes: {total_episodes}")
    print(f"平均每episode: {elapsed_time / total_episodes:.2f}秒")
    print(f"步数/秒: {total_steps / elapsed_time:.2f}")
    
    return elapsed_time, total_steps


def main():
    """性能对比测试"""
    print("\n" + "🎮" * 35)
    print("强化学习训练性能对比测试")
    print("🎮" * 35 + "\n")
    
    # 测试参数
    n_episodes_single = 10
    n_workers = 8
    episodes_per_worker = 10
    
    print(f"测试配置:")
    print(f"  - 单线程episodes: {n_episodes_single}")
    print(f"  - 多线程workers: {n_workers}")
    print(f"  - 每个worker的episodes: {episodes_per_worker}")
    print(f"  - 多线程总episodes: {n_workers * episodes_per_worker}")
    print()
    
    # 单线程测试
    single_time, single_steps = single_thread_benchmark(n_episodes_single)
    
    # 多线程测试
    parallel_time, parallel_steps = parallel_benchmark(n_workers, episodes_per_worker)
    
    # 性能对比
    print("\n" + "=" * 70)
    print("性能对比结果")
    print("=" * 70)
    
    # 计算加速比（基于相同episode数量）
    single_time_per_episode = single_time / n_episodes_single
    parallel_time_per_episode = parallel_time / (n_workers * episodes_per_worker)
    
    speedup = single_time_per_episode / parallel_time_per_episode
    
    print(f"\n单线程:")
    print(f"  - 每episode耗时: {single_time_per_episode:.2f}秒")
    print(f"  - 步数/秒: {single_steps / single_time:.2f}")
    
    print(f"\n多线程 ({n_workers} workers):")
    print(f"  - 每episode耗时: {parallel_time_per_episode:.2f}秒")
    print(f"  - 步数/秒: {parallel_steps / parallel_time:.2f}")
    
    print(f"\n加速比: {speedup:.2f}x")
    print(f"效率: {(speedup / n_workers) * 100:.1f}%")
    
    # 可视化对比
    print("\n" + "=" * 70)
    print("速度对比可视化")
    print("=" * 70)
    
    bar_length = 50
    single_bar = "█" * bar_length
    parallel_bar = "█" * int(bar_length * speedup)
    
    print(f"\n单线程: {single_bar} 1.00x")
    print(f"多线程: {parallel_bar} {speedup:.2f}x")
    
    # 建议
    print("\n" + "=" * 70)
    print("建议")
    print("=" * 70)
    
    if speedup >= 6:
        print("✓ 多线程性能优秀！建议使用train_parallel.py进行训练")
    elif speedup >= 4:
        print("✓ 多线程性能良好！建议使用train_parallel.py进行训练")
    elif speedup >= 2:
        print("⚠ 多线程有一定提升，但可能受到CPU限制")
        print("  建议检查CPU核心数和系统负载")
    else:
        print("⚠ 多线程提升不明显，可能存在以下问题：")
        print("  - CPU核心数不足")
        print("  - 系统负载过高")
        print("  - 进程间通信开销过大")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
