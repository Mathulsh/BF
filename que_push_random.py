"""
本机推送数据到 Redis（支持断点续传 + 批处理 + 内存安全）
适用于大规模组合生成（1e8级别）
"""

import random
import signal
import sys
import os
from itertools import islice
from rd import push_to_redis  # type: ignore

# =========================
# 全局控制开关
# =========================
should_stop = False

def signal_handler(signum, frame):
    """处理中断信号（Ctrl+C / kill）"""
    global should_stop
    print("\n收到中断信号，准备安全退出...")
    should_stop = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =========================
# 特征生成器（核心优化点）
# =========================
def generate_task(seed: int):
    """
    用 seed 控制随机空间（完全可复现 + 无状态）
    """
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(1, 100), 7)))


# =========================
# seed 流式生成（关键优化）
# =========================
def seed_stream(start: int, end: int):
    for seed in range(start, end):
        yield seed


# =========================
# batch 切分（流式，不占内存）
# =========================
def batch_generator(iterable, batch_size: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


# =========================
# 断点读取
# =========================
def get_last_processed_batch():
    path = "push_progress.txt"
    if os.path.exists(path):
        try:
            return int(open(path).read().strip())
        except:
            return 0
    return 0


def save_progress(batch_idx: int):
    with open("push_progress.txt", "w") as f:
        f.write(str(batch_idx))


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    TOTAL_SEEDS = 100_000_000
    BATCH_SIZE = 100_000

    # 从断点恢复
    start_batch_index = get_last_processed_batch()
    start_seed = start_batch_index * BATCH_SIZE

    print(f"🚀 从 batch {start_batch_index} 开始恢复")
    print(f"🚀 seed 起点: {start_seed}")

    # seed 流
    seed_iter = seed_stream(start_seed, TOTAL_SEEDS)

    for batch_idx, seed_batch in enumerate(
        batch_generator(seed_iter, BATCH_SIZE),
        start=start_batch_index
    ):

        if should_stop:
            print(f"⛔ 中断于 batch {batch_idx}")
            save_progress(batch_idx)
            sys.exit(0)

        # =========================
        # 关键：这里才生成 task
        # =========================
        task_batch = [
            generate_task(seed)
            for seed in seed_batch
        ]

        # 推送 Redis（建议内部用 pipeline）
        push_to_redis(task_batch)

        print(f"✅ Batch {batch_idx} 完成，size={len(task_batch)}")

        # 保存进度
        save_progress(batch_idx + 1)

    # 清理进度文件
    if os.path.exists("push_progress.txt"):
        os.remove("push_progress.txt")

    print("🎉 全部任务生成并推送完成")