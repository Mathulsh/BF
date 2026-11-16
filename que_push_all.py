"""全量推送特征索引组合到 Redis 队列。

按 0..feature_num-1 生成全部长度为 k 的特征索引组合（``itertools.combinations``），
分批通过 pipeline 写入 Redis 的 ``mylist``；每完成一批就把批次号写入
``push_progress.txt``，中断后重新运行会从下一批继续，不会重复推送已完成的批次。

默认规模是 95 个特征中选 6 个，共 ``C(95, 6) = 869,107,785`` 个组合，
全量推送会占用大量 Redis 内存。正式运行前请先估算规模
（例如 ``python main.py --dry-run``），或改用更小的 k / 特征数。

用法：
    python que_push_all.py

注意：
    - 特征索引从 0 开始，需与 train.py 中 ``.iloc`` 的位置索引一致。
    - 收到 SIGINT / SIGTERM 时保存进度并退出，再次运行可断点续传。
"""
from __future__ import annotations

import itertools
import math
import os
import signal
import sys

from rd import push_to_redis

# 默认参数：95 个候选特征、组合长度 6、每批 10 万条
FEATURE_NUM = 95
K = 6
BATCH_SIZE = 100_000

# 断点续传进度文件：内容为「已完成的批次数」
PROGRESS_FILE = "push_progress.txt"

# 中断标志：由 signal_handler 置位，推送主循环检测后保存进度并退出
should_stop = False


def signal_handler(signum, frame):
    """处理中断信号，置位全局 should_stop。

    Args:
        signum: 信号编号（SIGINT / SIGTERM）。
        frame: 当前栈帧，由 signal 模块传入，此处不使用。
    """
    global should_stop
    print("\n收到中断信号，正在保存进度...")
    should_stop = True


def batch_generator(iterable, batch_size, start_index=0):
    """把可迭代对象切分为固定大小的批次，并支持跳过开头若干批次。

    惰性生成，任意时刻内存中只保留一个批次，适合上亿级别的组合空间。

    Args:
        iterable: 待切分的数据源（通常是 combinations 迭代器）。
        batch_size: 每个批次最多包含的元素个数。
        start_index: 跳过的批次数，用于断点续传。

    Yields:
        list: 一个批次的数据；数据源耗尽时结束。
    """
    iterator = iter(iterable)

    # 跳过已处理的批次
    for _ in range(start_index):
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break

    while True:
        batch = list(itertools.islice(iterator, batch_size))
        # 批次为空说明数据源已耗尽
        if not batch:
            break
        yield batch


def get_last_processed_batch(progress_file=PROGRESS_FILE):
    """读取进度文件中记录的已完成批次数。

    Args:
        progress_file: 进度文件路径。

    Returns:
        int: 已完成的批次数；文件不存在或内容非法时返回 0。
    """
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_progress(batch_index, progress_file=PROGRESS_FILE):
    """把已完成的批次数写入进度文件。

    Args:
        batch_index: 已完成的批次数（从 0 开始计数）。
        progress_file: 进度文件路径。
    """
    with open(progress_file, "w", encoding="utf-8") as f:
        f.write(str(batch_index))


def push_all_combinations(
    feature_num=FEATURE_NUM,
    k=K,
    batch_size=BATCH_SIZE,
    progress_file=PROGRESS_FILE,
):
    """生成全部特征组合并分批推送到 Redis。

    需在主线程中调用：内部会临时注册 SIGINT / SIGTERM 处理器，结束后恢复。

    Args:
        feature_num: 候选特征个数，索引范围为 0..feature_num-1。
        k: 每个组合包含的特征个数。
        batch_size: 每批推送的组合数。
        progress_file: 断点续传使用的进度文件路径。

    Returns:
        tuple[int, bool]: (本次推送的组合数, 是否全部推送完成)。
        被中断时第二项为 False，此时进度已写入 progress_file。

    Raises:
        ValueError: k 不在 1..feature_num 范围内。
    """
    if not 1 <= k <= feature_num:
        raise ValueError(f"k 必须位于 1..{feature_num} 之间，当前为 {k}")

    global should_stop
    should_stop = False

    numbers = list(range(feature_num))
    total = math.comb(feature_num, k)
    total_batches = math.ceil(total / batch_size)
    print(f"特征数={feature_num}, 组合长度={k}, 组合总数={total:,}（约 {total_batches:,} 批）")

    start_batch = get_last_processed_batch(progress_file)
    if start_batch > 0:
        print(f"检测到进度：从第 {start_batch + 1} 批开始（已完成 {start_batch:,} 批）")

    original_handlers = (
        signal.signal(signal.SIGINT, signal_handler),
        signal.signal(signal.SIGTERM, signal_handler),
    )
    pushed = 0
    completed = False
    try:
        combos = itertools.combinations(numbers, k)
        for i, batch in enumerate(batch_generator(combos, batch_size, start_batch), start=start_batch):
            if should_stop:
                print(f"程序被中断，批次 {i + 1} 未完成，进度已保存")
                save_progress(i, progress_file)
                return pushed, False

            push_to_redis(batch)
            pushed += len(batch)
            save_progress(i + 1, progress_file)
            done = min(total, (i + 1) * batch_size)
            print(f"Batch {i + 1}/{total_batches} completed（累计 {done:,}/{total:,}）")

        completed = True
    finally:
        signal.signal(signal.SIGINT, original_handlers[0])
        signal.signal(signal.SIGTERM, original_handlers[1])
        # 全部完成后删除进度文件，避免下次误以为还有未完成批次
        if completed and os.path.exists(progress_file):
            os.remove(progress_file)

    return pushed, completed


def main():
    """脚本入口：按默认规模（95 选 6）全量推送组合。

    Returns:
        int: 0 表示全部推送完成，1 表示被中断（进度已保存，可重新运行续传）。
    """
    print("开始推送全部特征组合到 Redis ...")
    pushed, completed = push_all_combinations()
    if completed:
        print(f"全部推送完成，本次共推送 {pushed:,} 个组合")
        return 0
    print(f"推送未完成，本次已推送 {pushed:,} 个组合；重新运行可从断点继续")
    return 1


if __name__ == "__main__":
    sys.exit(main())
