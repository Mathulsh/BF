"""结果收集脚本：把 Redis 中的训练结果批量导入 DuckDB。

对所有健康实例的 ``results`` 队列做批量 RPOP，攒够一批后写入 DuckDB 的
``results`` 表（表不存在时自动创建）。可一边训练一边运行，队列连续多轮
为空时自动结束；运行中按 Ctrl+C 可优雅退出且不丢已处理的数据。

用法：
    python collector.py                  # 写入 results.duckdb
    python collector.py results_gb.duckdb
    python collector.py --batch-size 50000
"""
from __future__ import annotations

import argparse
import sys

from rd import collect_redis_results_to_duckdb, redis_list

# 默认结果库路径与单实例单批拉取条数
DEFAULT_DUCKDB_PATH = "results.duckdb"
DEFAULT_BATCH_SIZE = 50_000


def collect_results(duckdb_path=DEFAULT_DUCKDB_PATH, batch_size=DEFAULT_BATCH_SIZE):
    """把 Redis 中积压的训练结果收集到 DuckDB。

    Args:
        duckdb_path: 结果库文件路径，不存在时自动创建。
        batch_size: 每个实例每次从 results 队列拉取的条数。

    Returns:
        None
    """
    return collect_redis_results_to_duckdb(
        redis_list=redis_list,
        duckdb_path=duckdb_path,
        batch_size=batch_size,
    )


def main(argv=None):
    """解析命令行参数并执行一次结果收集。

    Args:
        argv: 参数列表；None 表示使用 ``sys.argv[1:]``。

    Returns:
        int: 进程退出码，0 表示正常结束。
    """
    parser = argparse.ArgumentParser(
        description="BF 结果收集：Redis results 队列 → DuckDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("duckdb_path", nargs="?", default=DEFAULT_DUCKDB_PATH, help="结果库文件路径")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每个实例每次拉取的条数")
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size 必须 >= 1")

    collect_results(args.duckdb_path, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
