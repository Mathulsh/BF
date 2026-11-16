"""主运行脚本：一键串联「推送 → 训练 → 收集」三个阶段。

模块对应关系：
    ① 推送：que_push_all.push_all_combinations（生成并推送特征组合）
    ② 训练：train.run_workers（多进程训练，仅 RF / GB）
    ③ 收集：collector.collect_results（Redis 结果 → DuckDB）

三个阶段通过 Redis 队列解耦，因此也可以分别单独运行：
    ``python que_push_all.py``、``python train.py 4``、``python collector.py``。

用法：
    python main.py                            # 1 个 worker，RF，95 选 6
    python main.py 4                          # 4 个 worker，RF
    python main.py 4 --model gb               # 4 个 worker，GB
    python main.py --dry-run                  # 只估算组合总数，不连接 Redis
    python main.py --features 20 --k 3        # 小规模验证. 20 个特征，3 特征组合
    python main.py --skip-push 4              # 队列中已有任务，直接训练 4 个 worker
    python main.py --skip-push --no-collect   # 只训练，不收集

注意：
    - 默认 C(95, 6) = 869,107,785 个组合，正式运行前请先 ``--dry-run`` 估算；
      单条组合在本机约 0.17 秒，全量穷举需要成规模的算力。
    - 推送完成后会写入 ``push_completed.txt``，避免重复推送；需要重推时删除
      该文件与 ``push_progress.txt``，并清空 Redis 中的三个队列键。
    - 训练阶段在队列空闲 ``--idle-exit`` 秒（默认 60）后自动结束并进入收集，
      也可以提前按 Ctrl+C 结束训练。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# 与 train.py 保持一致；此处单独定义是为了让 --dry-run 不必导入联网模块
MODEL_CHOICES = ("rf", "gb")

# 推送完成标记文件：存在则跳过推送阶段
PUSH_MARKER = "push_completed.txt"


def parse_args(argv=None):
    """解析命令行参数。

    Args:
        argv: 参数列表；None 表示使用 ``sys.argv[1:]``。

    Returns:
        argparse.Namespace: 解析后的参数对象。

    Raises:
        SystemExit: 参数非法时由 argparse 退出。
    """
    parser = argparse.ArgumentParser(
        description="BF：高熵合金特征组合暴力穷举流水线（推送 → 训练 → 收集）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("num_workers", nargs="?", type=int, default=1, help="训练 worker 进程数")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="rf", help="基学习器：rf=随机森林，gb=梯度提升")
    parser.add_argument("--features", type=int, default=95, help="候选特征数，索引范围 0..features-1")
    parser.add_argument("--k", type=int, default=6, help="每个组合包含的特征个数")
    parser.add_argument("--batch-size", type=int, default=100_000, help="推送阶段的每批组合数")
    parser.add_argument("--idle-exit", type=float, default=60.0, help="队列空闲多少秒后结束训练，0 表示一直等待")
    parser.add_argument("--duckdb", default="auto", help="结果库路径；auto 表示 results_{model}.duckdb")
    parser.add_argument("--skip-push", action="store_true", help="跳过推送阶段（任务已在 Redis 中）")
    parser.add_argument("--no-collect", action="store_true", help="跳过结果收集阶段")
    parser.add_argument("--dry-run", action="store_true", help="只打印组合总数，不连接 Redis")

    args = parser.parse_args(argv)

    if args.num_workers < 1:
        parser.error("num_workers 必须 >= 1")
    if args.features < 1:
        parser.error("--features 必须 >= 1")
    if not 1 <= args.k <= args.features:
        parser.error(f"--k 必须位于 1..{args.features} 之间")
    if args.batch_size < 1:
        parser.error("--batch-size 必须 >= 1")
    if args.idle_exit < 0:
        parser.error("--idle-exit 不能为负数")
    return args


def _read_push_marker(path=PUSH_MARKER):
    """读取推送完成标记文件。

    Args:
        path: 标记文件路径。

    Returns:
        dict | None: 文件不存在返回 None；存在时返回解析出的
        ``{"features": int, "k": int}``，内容无法解析时返回空字典。
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            parts = dict(item.split("=", 1) for item in f.read().split() if "=" in item)
        return {"features": int(parts["features"]), "k": int(parts["k"])}
    except (OSError, KeyError, ValueError):
        return {}


def _write_push_marker(features, k, total, path=PUSH_MARKER):
    """写入推送完成标记文件，便于下次运行时跳过推送。

    Args:
        features: 本次推送使用的特征数。
        k: 本次推送使用的组合长度。
        total: 组合总数。
        path: 标记文件路径。
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"features={features} k={k} total={total}\n")


def run_pipeline(args):
    """依次执行推送、训练、收集三个阶段。

    Args:
        args: :func:`parse_args` 返回的参数对象。

    Returns:
        int: 进程退出码，0 表示正常结束，1 表示前置检查或推送阶段失败。
    """
    start = time.time()
    total = math.comb(args.features, args.k)

    print("=" * 64)
    print(
        f"BF 流水线 | {args.features} 选 {args.k} = {total:,} 个组合 | "
        f"模型={args.model.upper()} | worker={args.num_workers}"
    )
    print("=" * 64)

    if args.dry_run:
        print("[dry-run] 仅估算组合总数，未连接 Redis，也未写入任何任务。")
        print("          去掉 --dry-run 即可开始正式运行。")
        return 0

    # 延迟导入：避免 --dry-run 时因导入 rd 而按 REDIS_CONFIGS 逐个连接（失败时会有长超时）
    from collector import collect_results
    from que_push_all import push_all_combinations
    from rd import get_healthy_redis_clients, redis_list
    from train import run_workers

    # ---------- 阶段 1：推送 ----------
    print("\n=== 推送阶段 ===")
    healthy = len(get_healthy_redis_clients())
    if healthy == 0:
        print("❌ 错误：所有 Redis 实例均不可用，请检查网络连接与 Redis 服务器状态")
        return 1
    print(f"✅ Redis 健康检查通过：{healthy}/{len(redis_list)} 个实例可用")

    marker = _read_push_marker()
    if args.skip_push:
        print("⏭  已指定 --skip-push，跳过推送")
    elif marker is not None:
        if marker and (marker["features"], marker["k"]) != (args.features, args.k):
            print(f"⚠️  已有推送标记（features={marker['features']}, k={marker['k']}），"
                  f"与本次（features={args.features}, k={args.k}）不一致")
            print(f"   如需重推：删除 {PUSH_MARKER} 与 push_progress.txt 并清空 Redis 队列后重试；")
            print("   如需训练现有队列：请显式加上 --skip-push。")
            return 1
        print(f"⏭  检测到 {PUSH_MARKER}，跳过推送（如需重推请删除该文件与 push_progress.txt）")
    else:
        try:
            pushed, completed = push_all_combinations(
                feature_num=args.features,
                k=args.k,
                batch_size=args.batch_size,
            )
        except Exception as e:  # 例如推送过程中所有 Redis 实例掉线
            print(f"❌ 推送失败：{e}")
            return 1
        if not completed:
            print("⚠️  推送被中断，进度已保存；未进入训练阶段。重新运行可从断点继续。")
            return 1
        _write_push_marker(args.features, args.k, total)
        print(f"✅ 推送完成，本次共推送 {pushed:,} 个组合")

    # ---------- 阶段 2：训练 ----------
    print("\n=== 训练阶段 ===")
    print(f"模型={args.model.upper()}，worker={args.num_workers}，空闲退出={args.idle_exit:.0f}s")
    run_workers(
        num_workers=args.num_workers,
        model_name=args.model,
        idle_exit=args.idle_exit,
    )
    print("训练阶段结束")

    # ---------- 阶段 3：收集 ----------
    if args.no_collect:
        print("\n⏭  已指定 --no-collect，跳过收集阶段（结果仍留在 Redis 中）")
    else:
        print("\n=== 收集阶段 ===")
        duckdb_path = args.duckdb if args.duckdb != "auto" else f"results_{args.model}.duckdb"
        print(f"结果库：{duckdb_path}")
        collect_results(duckdb_path=duckdb_path)

    print(f"\n🎉 全部流程结束，总耗时 {time.time() - start:.1f}s")
    return 0


def main(argv=None):
    """脚本入口：解析参数并执行完整流水线。

    Args:
        argv: 参数列表；None 表示使用 ``sys.argv[1:]``。

    Returns:
        int: 进程退出码。
    """
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
