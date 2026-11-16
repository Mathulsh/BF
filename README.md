# BF —— 高熵合金性能分类的特征组合暴力穷举

**BF** 为 **Brute Force** 的缩写，即中文的「暴力穷举」。

本项目面向高熵合金（High-Entropy Alloy, HEA）的小样本性能数据集，对候选描述符做**特征子集穷举搜索**：枚举所有大小为 *k* 的特征组合，对每一个组合执行同一套机器学习评估流程（分层 5 折交叉验证 + 训练集/测试集指标），再把海量结果汇总进 DuckDB，用于检索「哪些特征组合最有效」。

核心思路是用空间换时间——把**组合爆炸**交给 Redis 队列，把**逐个组合训练**摊到多进程与多台机器上。

评估器只保留两种：**随机森林（RF）** 与 **梯度提升（GB）**；三个环节（推送 / 训练 / 收集）分别对应 `que_push_all.py`、`train.py`、`collector.py`，由 `main.py` 一键串联，也可单独运行。

## 特性速览

- **组合枚举 + 断点续传**：中断后从上次完成的批次继续，不重复推送
- **多 Redis 实例分片**：`hash` 分片写入，带健康检查、自动重连、故障转移
- **任务可追溯、不静默丢失**：`LMOVE` 原子领取 + `mylist:processing` 在途队列 + 事务 ACK，异常任务可查漏补算
- **流式生成**：批次生成器按需产出组合，不会把上亿条组合一次性读进内存
- **边算边收**：worker 结果回流 Redis，采集器批量落盘 DuckDB，可与训练并行运行
- **只需两种基学习器**：随机森林（RF）与梯度提升（GB），降低依赖与维护成本
- **一条命令跑全程**：`main.py` 串联推送、训练、收集，队列空闲后自动收尾

## 目录

- [1. 研究背景](#1-研究背景)
- [2. 系统架构](#2-系统架构)
- [3. 目录结构](#3-目录结构)
- [4. 环境准备](#4-环境准备)
- [5. 数据说明](#5-数据说明)
- [6. 配置 Redis](#6-配置-redis)
- [7. 快速开始](#7-快速开始)
- [8. 结果库与常用查询](#8-结果库与常用查询)
- [9. 核心机制](#9-核心机制)
- [10. 组合规模参考](#10-组合规模参考)
- [11. 已知问题与限制](#11-已知问题与限制)
- [12. 常见问题](#12-常见问题)
- [13. 后续改进方向](#13-后续改进方向)
- [14. 许可证](#14-许可证)

## 1. 研究背景

高熵合金的候选特征（描述符）成百上千，而实验样本通常只有几十到一百余条。此时「特征越多越好」并不成立：冗余特征会显著抬高过拟合风险，也会让模型难以解释。因此需要从大量候选描述符中挑出一个**规模小、泛化好**的特征子集。

穷举搜索的好处是没有任何启发式假设——只要算力足够，就能给出该组合规模下的全局最优解。难点在于组合数量随特征数与 *k* 呈组合级爆炸（见[第 10 节](#10-组合规模参考)），必须借助分布式队列才能落地。

本仓库承担的工作即这条流水线：**生成组合 → 分发 → 训练评估 → 汇总入库**。

## 2. 系统架构

```
① 组合生成    que_push_all.py（main.py 默认调用）/ que_push_fixed.py / que_push_random.py
      │       按 itertools.combinations 或随机种子流式产出特征索引组合
      │       RPUSH 批量写入（pipeline），按 hash(组合) 分片到多个实例
      ▼
② Redis 队列  mylist             待领取任务
              mylist:processing  已领取、未确认（在途）
              results            训练结果
      │       LMOVE LEFT → RIGHT 原子领取
      ▼
③ 训练评估    train.py（仅 RF / GB，多进程 / 多机）
              取列 → 分层 5 折 CV → 全量重训 → 训练集/测试集指标
      │       MULTI/EXEC：RPUSH results + LREM mylist:processing
      ▼
④ 结果收集    collector.py / rd.collect_redis_results_to_duckdb
              RPOP 批量拉取 → results.duckdb（表 results）
```

## 3. 目录结构

| 文件 | 作用 |
| --- | --- |
| `rd.py` | **核心库**：Redis 连接池、健康检查、任务推送/领取、结果回写、结果采集入 DuckDB |
| `train.py` | 训练 worker：循环领取任务，做 5 折 CV 与训练/测试评估，结果写回 Redis；仅 RF / GB，可多进程 |
| `que_push_all.py` | 推送脚本：全量顺序枚举（默认 95 选 6），支持断点续传 |
| `que_push_fixed.py` | 推送脚本：固定一个关键特征（索引 0），枚举其余特征组合 |
| `que_push_random.py` | 推送脚本：按 `seed` 随机采样，可复现、支持断点、适合 1e8 级任务量 |
| `collector.py` | 采集脚本：把 Redis 中的结果批量导入 DuckDB（默认 `results.duckdb`） |
| `main.py` | 一键流水线：复用 `que_push_all` + `train` + `collector`，可选模型、组合规模与结果库 |
| `data_reg_to_cls.py` | 数据预处理：连续标签按阈值离散化为 3/4/5 类，分层划分并导出 CSV |
| `duckdb_check.py` | 结果检查：统计条数、取 Top-N、合并多个 DuckDB、查漏补算（部分代码已注释） |
| `pipipickle.py` | 小工具：CSV ↔ pickle 互转（当前仅保留读取、打印 pickle 内容） |
| `yasuo-*.pkl`、`lashen-*.pkl` | 压缩 / 拉伸数据集（训练集与测试集） |
| `requirements-3.10.txt` | 依赖清单（锁定版本） |
| `results/` | 历史实验结果库（已被 `.gitignore` 忽略） |

## 4. 环境准备

```bash
# 建议使用 Python 3.10（与 requirements-3.10.txt 的锁定版本匹配）
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-3.10.txt
```

说明：

- `pyproject.toml` 声明 `requires-python = ">=3.9"`。依赖清单锁定的版本面向 **Python 3.10**，3.9 / 3.11 一般也可用，更高版本需适当放宽版本锁定。仓库内 `.python-version` 为 `3.13`，而既有 `.venv` 是 Python 3.9，建议按需**重建虚拟环境**，避免依赖版本冲突。
- 所有模块均以 `from __future__ import annotations` 开头，因此在 Python 3.9 下也能正常导入（否则 `X | None` 形式的注解会在 3.9 报 `TypeError`）。
- 核心依赖：`pandas`、`numpy`、`scikit-learn`、`redis`、`duckdb`。RF 与 GB 均由 scikit-learn 提供，无需额外安装算法库。

## 5. 数据说明

仓库自带两套数据，均为 `pandas.DataFrame` 的 pickle：

| 数据集 | 文件 | 样本数 | 特征数 | 标签列 | 标签取值 |
| --- | --- | --- | --- | --- | --- |
| 压缩（yasuo） | `yasuo-train.pkl` / `yasuo-test.pkl` | 78 / 14 | 95 | `Label` | `low` / `moderate` / `high` |
| 拉伸（lashen） | `lashen-train-101tz.pkl` / `lashen-test-101tz.pkl` | 107 / 20 | 101 | `label` | `0` / `1` / `2` |

**约定：最后一列为标签，其余列依次为特征。** `train.py` 使用 `.iloc[:, -1]` 取标签、`.iloc[:, idx]` 取特征，因此**列顺序即特征编号**（0 起始）。

连续标签的离散化规则（`data_reg_to_cls.py`）：

| 类别数 | 分箱 | 标签映射 |
| --- | --- | --- |
| 3 类 | `(-inf, 5)`、`[5, 20)`、`[20, inf)` | 0 / 1 / 2 |
| 4 类 | 在 3 类基础上再拆分 `[10, 20)` | 0 / 1 / 2 / 3 |
| 5 类 | 进一步拆分 `[20, 30)`、`[30, inf)` | 0 / 1 / 2 / 3 / 4 |

> **数据小贴士**：`yasuo-*.pkl` 的列名是乱码——原始 UTF-8 文本（首列含 BOM）被按 GBK 解码所致，例如 `锘縆/G` 实为 `K/G`、`蟽E1` 为 `σE1`、`惟^-1` 为 `ν^-1`。按位置取特征不受影响，但按列名检索/展示时需注意；`lashen-*.pkl` 列名正常。另外两套数据的标签列名大小写不一致（`Label` vs `label`），代码里均按位置访问，故无影响。

## 6. 配置 Redis

所有 Redis 相关逻辑集中在 `rd.py` 顶部：

```python
# Redis配置——bj 8个（当前生效）
REDIS_CONFIGS = [
    {"host": "172.19.123.221", "port": 40042},
    # ... 共 8 个实例
]
```

文件中已预设三套配置（`hz` 4 实例、`bj` 8 实例、本地 4 实例），按需注释/启用即可。要点：

- **导入即连接**：`import rd` 时会立即创建客户端并 `ping()`，因此离线环境下导入会打印连接失败日志，被标记为 `None` 的实例会在后续 `get_healthy_redis_clients()` 中尝试重连。
- **必须修改**：仓库中的主机地址为实验内网地址，运行前请替换为自己的 Redis 地址与端口。
- **无认证**：当前未配置密码。跨机部署时请限定在可信内网，或为 Redis 开启 `requirepass` 并相应修改连接参数。
- **队列名**（硬编码于 `rd.py`）：`mylist`（待处理）、`mylist:processing`（在途）、`results`（结果）。
- 连接参数：连接超时 10s、读写超时 30s、`health_check_interval=30`、`max_connections=50`、`retry_on_timeout=True`。

## 7. 快速开始

### 7.1 分步运行（推荐）

```bash
# 0) 可选：先确认 Redis 可用数量
python -c "from rd import get_healthy_redis_clients as g, redis_list; print(len(g()), '/', len(redis_list))"

# 1) 推送任务（三选一，按实验设计决定组合空间）
python que_push_all.py         # 全量 95 选 6 → 869,107,785 个组合（务必先估算规模）
# python que_push_fixed.py     # 固定索引 0，其余 97 选 4 → 3,464,840 个组合
# python que_push_random.py    # 1e8 次随机采样（可复现）

# 2) 启动训练 worker：进程数 [模型]
python train.py 4              # 4 个 worker，RF
python train.py 4 gb           # 4 个 worker，GB

# 3) 另开终端收集结果（可与训练并行；空闲约 2 秒后自动退出）
python collector.py                    # 写入 results.duckdb
python collector.py results_gb.duckdb  # 指定结果库
```

> 独立运行 `train.py` 时 worker 默认一直等待（等价于旧的 `while True`），训练完成后用 `Ctrl+C` 结束，再运行采集器；加上 `--idle-exit 60` 可以让 worker 在队列空闲 60 秒后自行退出。
>
> 多 worker 之间互不协调，也不会自动重投在途任务；结束训练前建议先确认 `mylist` 为空。

### 7.2 一键运行

`main.py` 复用上述三个模块，按「推送 → 训练 → 收集」顺序执行；训练在队列空闲 `--idle-exit` 秒（默认 60）后自动结束，然后进入收集阶段。

```bash
python main.py                                   # 1 个 worker，RF，默认 95 选 6
python main.py 4                                 # 4 个 worker，RF
python main.py 4 --model gb                      # 4 个 worker，GB
python main.py --dry-run                         # 只估算组合总数（不连接 Redis）
python main.py --features 20 --k 3               # 小规模验证
python main.py --skip-push 4                     # 队列中已有任务，直接训练
python main.py --skip-push --no-collect          # 只训练，不收集
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `num_workers` | `1` | 训练 worker 进程数 |
| `--model {rf,gb}` | `rf` | 基学习器：随机森林 / 梯度提升 |
| `--features` | `95` | 候选特征数，索引 `0..features-1` |
| `--k` | `6` | 每个组合的特征个数 |
| `--batch-size` | `100000` | 推送阶段每批组合数 |
| `--idle-exit` | `60` | 队列空闲多少秒后结束训练，`0` 表示一直等待 |
| `--duckdb` | `auto` | 结果库路径；`auto` 表示 `results_{model}.duckdb` |
| `--skip-push` | 关 | 跳过推送阶段 |
| `--no-collect` | 关 | 跳过收集阶段 |
| `--dry-run` | 关 | 只打印组合总数，不连接 Redis |

编排细节：

- **推送完成标记**：推送全部结束后写入 `push_completed.txt`（内容含 `features` / `k` / `total`）。再次运行时若标记的参数与本次一致就跳过推送；不一致则报错退出，避免用错误的任务集训练。需要重推时删除该文件与 `push_progress.txt`，并清空 Redis 三个队列键。
- **按模型分库**：RF 与 GB 默认分别写入 `results_rf.duckdb` / `results_gb.duckdb`，避免两种算法的结果混在一张表里；用 `--duckdb` 可覆盖。
- **中断行为**：推送阶段按 `Ctrl+C` 会保存进度并以退出码 1 结束（不进入训练）；训练阶段按 `Ctrl+C` 会终止所有 worker 并继续进入收集阶段。

### 7.3 数据预处理（可选）

若手头是连续标签的 CSV：

```bash
# 修改 data_reg_to_cls.py 末尾的 df 路径与 output_prefix 后运行
python data_reg_to_cls.py
```

脚本会按阈值做 3 类（或 4/5 类，改一处函数名即可）离散化，以 `stratify` 做 8:2 划分，`MinMaxScaler` **仅在训练集上 fit**，并导出 `*_train.csv`、`*_test.csv` 与未归一化的完整数据集。

### 7.4 冒烟测试（首次部署建议先做）

用 `--features 8 --k 2` 把组合数压到 28 个，并让 worker 在队列空闲 3 秒后自动退出：

```bash
python main.py --features 8 --k 2 --idle-exit 3 --duckdb smoke_rf.duckdb
python -c "import duckdb; print(duckdb.connect('smoke_rf.duckdb').execute('select count(*) from results').fetchone())"
```

期望输出 `28`（对应 `C(8,2)`）。校验通过后：

```bash
rm -f push_completed.txt push_progress.txt smoke_rf.duckdb
```

并清空 Redis 中的 `mylist`、`mylist:processing`、`results` 三个键，再按正式规模运行。

## 8. 结果库与常用查询

`rd.py` 会自动建表（`CREATE TABLE IF NOT EXISTS results`），字段如下：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `features` | `VARCHAR[]` | 特征名列表（按列名写入；`yasuo` 数据为乱码列名，见[第 5 节](#5-数据说明)） |
| `cv_f1_macro` | `DOUBLE` | 5 折分层交叉验证的宏平均 F1 均值 |
| `cv_accuracy` | `DOUBLE` | 5 折分层交叉验证的准确率均值 |
| `train_f1_macro` | `DOUBLE` | 全量训练后，训练集宏平均 F1 |
| `train_accuracy` | `DOUBLE` | 训练集准确率 |
| `test_f1_macro` | `DOUBLE` | 测试集宏平均 F1（留出集，最能反映泛化） |
| `test_accuracy` | `DOUBLE` | 测试集准确率 |

> 早期实验（如 `duckdb_check.py` 中的 `mean_f1_macro` / `mean_accuracy` 两列版本）表结构与上表不同，查询前请用 `DESCRIBE results` 确认。
>
> `main.py` 默认按模型分库（`results_rf.duckdb` / `results_gb.duckdb`）；若自行把两种算法的结果写进同一个库，表中没有 `model` 列可供区分，需要在文件名或实验记录里标注。

常用查询：

```sql
-- 1. 按交叉验证得分取 Top-20
SELECT features, cv_f1_macro, cv_accuracy, test_f1_macro, test_accuracy
FROM results
ORDER BY cv_f1_macro DESC
LIMIT 20;

-- 2. 去重后取每个组合的最佳成绩（重试可能产生重复行）
SELECT features, max(cv_f1_macro) AS best_cv_f1
FROM results
GROUP BY features
ORDER BY best_cv_f1 DESC
LIMIT 20;

-- 3. 找到「交叉验证高但测试集明显下降」的过拟合组合
SELECT features, cv_f1_macro, test_f1_macro, cv_f1_macro - test_f1_macro AS gap
FROM results
ORDER BY gap DESC
LIMIT 20;

-- 4. 只看包含某个特征（如 Hmix）的组合
SELECT * FROM results
WHERE list_contains(features, 'Hmix')
ORDER BY cv_f1_macro DESC
LIMIT 10;

-- 5. 统计进度
SELECT count(*) AS n FROM results;
```

合并多个结果库：

```python
import duckdb

con = duckdb.connect("all_results.duckdb")

# 第一个库：建表并导入
con.execute("ATTACH 'results/result-GB-3-yasuo-MD.duckdb' AS db3 (READ_ONLY)")
con.execute("CREATE TABLE IF NOT EXISTS results AS SELECT * FROM db3.results")
con.execute("DETACH db3")

# 后续库：追加
con.execute("ATTACH 'results/result-GB-4-yasuo-MD.duckdb' AS db4 (READ_ONLY)")
con.execute("INSERT INTO results SELECT * FROM db4.results")
con.execute("DETACH db4")

con.close()
```

> 注意：DuckDB 不能直接把 `.duckdb` 文件当作表名查询（replacement scan 仅支持 CSV/Parquet/JSON），合并必须用 `ATTACH`。

## 9. 核心机制

**1) 分片写入**
`push_to_redis()` 用 `hash(组合) % 健康实例数` 决定写入哪个实例，并通过 `pipeline` 一次性提交整批 `RPUSH`，减少往返次数。分片只影响写入分布，读取端采用轮询，不依赖该 hash。

**2) 领取与 ACK**
`read_one_from_redis()` 用 `LMOVE mylist → mylist:processing`（`LEFT` → `RIGHT`）原子地把任务从待处理队列搬到在途队列：任务被领取后即从 `mylist` 消失，即使 worker 崩溃也不会被其他 worker 重复领取——代价是**不会自动重投**，需要人工补算。

训练完成后 `push_result_to_redis()` 在同一个 `MULTI/EXEC` 事务中执行 `RPUSH results` 与 `LREM mylist:processing`，因此不会出现「结果写了但任务没确认」的不一致；网络异常时的重试可能造成**少量重复结果**，分析时按 `features` 去重即可。

**3) 断点续传**
`que_push_all.py` 每完成一批就写入 `push_progress.txt`（记录已完成批次数），重启时从下一批继续；全部完成后删除进度文件。`main.py` 另外用 `push_completed.txt` 记录本次推送的 `features` / `k` / `total`，避免重复推送以及参数不一致时误用旧队列。

**4) 健康检查与故障转移**
`get_healthy_redis_clients()` 会 `ping` 现有连接、尝试重连此前失败的实例，并剔除不健康节点；结果回写与 ACK 在首选实例失败时会依次尝试其余健康实例。

**5) 内存控制**
`push_all_combinations()` 内部用 `batch_generator` + `islice` 惰性切批，组合空间再大也只在内存中保留一个批次（默认 10 万条）。采集端按批 `RPOP`，并每累计 50 万行落盘一次后再清空缓冲。

**6) 评估口径**
所有组合共用同一组 `StratifiedKFold` 划分（`train.py` 中 `random_state=0`、`shuffle=True`），保证不同组合之间的分数可比。指标同时记录 CV、训练集、测试集三组，便于识别「CV 很高但测试集崩掉」的过拟合组合。

**7) 空闲退出**
`read_one_from_redis(timeout=...)` 支持带超时的领取：`main.py` 让 worker 以 1 秒为粒度轮询，队列持续空闲 `--idle-exit` 秒后返回，随后进入收集阶段。独立运行 `train.py` 时该机制默认关闭（`--idle-exit 0`），行为与原来一致。

## 10. 组合规模参考

| 入口 | 特征索引范围 | 组合长度 | 组合总数 | 批次大小 |
| --- | --- | --- | --- | --- |
| `main.py` / `que_push_all.py`（默认） | 0 – 94（95 个） | 6 | 869,107,785 | 100,000 |
| `main.py --k 4` | 0 – 94（95 个） | 4 | 3,183,545 | 100,000 |
| `main.py --features 20 --k 3` | 0 – 19（20 个） | 3 | 1,140 | 100,000 |
| `que_push_fixed.py` | 固定 0，其余 97 选 4 | 5 | 3,464,840 | 100,000 |
| `que_push_random.py` | 1 – 99，`random.sample` | 6 | 100,000,000 次采样（非穷举） | 100,000 |

按实测：在本机（Apple Silicon，单进程、默认参数）评估**一个组合**（78 样本 × 5 折 CV）RF 约需 **0.16 s**、GB 约需 **0.27 s**，折合单核每秒约 6 个组合（RF）或 4 个组合（GB）。下表按 RF 估算：

| 组合空间 | 单核耗时（估算） |
| --- | --- |
| `C(95,4) = 3,183,545` | 约 143 CPU·小时（≈ 6.0 CPU·天） |
| `C(97,4) = 3,464,840` | 约 156 CPU·小时（≈ 6.5 CPU·天） |
| `C(95,6) = 869,107,785` | 约 1,630 CPU·天（≈ 4.5 CPU·年） |

也就是说，`k = 4` 量级用几台多核机器可以接受，`k = 6` 的全量穷举必须依靠成规模的集群；换成 GB 时上述耗时再乘约 1.6 倍（`C(95,6)` 约 2,676 CPU·天）。同时 Redis 还会堆积大量任务：单条任务的 pickle 载荷约 42 字节，8.69 亿条仅载荷就有约 34 GiB，算上列表节点开销通常要到数十 GB 量级。**建议先用 `main.py --dry-run` 估算规模、用 `--features 20 --k 3` 验证流程，再逐步放大。** 若改用更轻的模型配置（减少树数、限制 `max_depth`、或先用单折快速粗筛），单条任务还能再降一个数量级。

## 11. 已知问题与限制

1. **进度文件不记录参数**：`push_progress.txt` 只保存「已完成批次数」，不保存 `features` / `k` / `batch_size`。中断后若改用不同参数续传，会跳过错误的组合——请务必用相同参数续传，或删除进度文件与队列后重推。（`main.py` 的 `push_completed.txt` 记录了参数，可拦住「整轮跑完之后改参数」的情况。）
2. **索引基准不统一**：`que_push_all.py`（含 `main.py`）与 `que_push_fixed.py` 使用 0 起始索引（配合 `train.py` 的 `.iloc` 位置索引），而 `que_push_random.py` 仍是 1–99。推送前请确认与数据集列数一致，否则会越界或漏掉第 0 列。
3. **路径与地址硬编码**：`rd.py` 的 Redis 地址、`train.py` 的 `TRAIN_PATH` / `TEST_PATH`、`data_reg_to_cls.py` 的 `data/77.csv`、`duckdb_check.py` 的 DuckDB 绝对路径都写在代码里（`data/` 目录未随仓库提供），迁移环境时需逐一修改。
4. **结果表没有 `model` 列**：RF 与 GB 若写入同一个库就无法区分（`main.py` 默认按模型分库规避）；同时 `results` 没有主键或唯一索引，重复实验、重试都会追加行，需要按 `features` 去重（见[第 8 节](#8-结果库与常用查询)）。
5. **在途任务不会自动重投**：进程被杀后，任务会滞留在 `mylist:processing`，需要人工处理；`duckdb_check.py` 里有一段「查漏并重新推送」的 SQL（已注释，需按实际特征范围与组合长度修改）。
6. **worker 退出方式**：独立运行 `train.py` 时默认一直等待（需 `Ctrl+C`）；`main.py` 依靠 `--idle-exit` 自动收尾，判定粒度为 1 秒，队列刚好在边界附近波动时可能提前退出——正式跑大批量时建议把 `--idle-exit` 调大。
7. **两个推送脚本仍是独立脚本**：`que_push_fixed.py` / `que_push_random.py` 未抽成可复用函数，`main.py` 只编排 `que_push_all`；如需纳入流水线，先按 `push_all_combinations` 的形式重构。
8. **安全**：Redis 未启用认证，且任务与结果均以 `pickle` 序列化传输/存储——只应在可信内网使用，不要连接不可信的 Redis 实例（`pickle.loads` 存在反序列化风险）。
9. **环境声明不一致**：`.python-version` 为 `3.13`，依赖清单名为 `requirements-3.10.txt`，`pyproject.toml` 又声明 `>=3.9`；建议统一到一个版本并重建虚拟环境。
10. **无测试与 CI**：仓库没有单元测试和持续集成，改动后建议先用 `main.py --features 8 --k 2 --idle-exit 3` 跑通全链路（见[第 7.4 节](#74-冒烟测试首次部署建议先做)）。

## 12. 常见问题

**Q：想重新从头推送，该怎么做？**
删除 `push_progress.txt` 与 `push_completed.txt`，并清空 Redis 中的三个键。可逐实例精确删除：

```bash
for port in 40042 40043 40044 40045 40057 40067 40068 40069; do
  redis-cli -h 172.19.123.221 -p "$port" DEL mylist mylist:processing results
done
```

（`FLUSHDB` 能一次清空，但会连带删掉同库的其他数据，慎用。）

**Q：worker 一直打印 Redis 错误或空转？**
`read_one_from_redis()` 在队列全空时会以 10ms 间隔轮询，属正常等待；若提示所有实例不可用，检查 `rd.py` 中的地址/端口、网络连通性与防火墙。想让它自己停下来，用 `--idle-exit <秒>`。

**Q：结果少了很多 / 出现重复？**
少：worker 被强杀导致任务滞留 `mylist:processing`，或结果写入三次重试均失败；补算方式见 `duckdb_check.py` 的查漏 SQL。重复：`push_result_to_redis` 在超时后重试，事务可能已执行成功但响应丢失；用 `GROUP BY features` 去重。

**Q：如何切换 RF 与 GB？**
流水线用 `python main.py 4 --model gb`；单独训练用 `python train.py 4 gb`。两者都在 `train.py` 的 `build_model()` 中构建，新增算法只需在那里加一个分支。分布式场景下建议保持每进程单线程（`n_jobs=1`），把并行度留给 worker 进程；实测同数据下 GB 单条约 0.27 s，约为 RF（0.16 s）的 1.6 倍。

**Q：想换评估指标？**
改 `train.py` 的 `SCORING` 列表（任意 sklearn 评分字符串），并同步调整 `evaluate_combination()` 里的结果字段与 `rd.py` 建表/写库的字段名。

**Q：DuckDB 库越来越大怎么办？**
结果表是追加写入的，单库到 GB 级后查询会变慢；可按实验（算法 / 特征数 / 数据集）分库命名（`results/result-GB-5-yasuo-MD.duckdb` 即此类命名，`main.py` 默认按模型命名），需要时用 `ATTACH` 合并查询。`.gitignore` 已忽略 `results/` 与 `*.duckdb`。

## 13. 后续改进方向

- 把评分（`SCORING`）、CV 折数与随机种子也做成命令行参数
- 让 `push_progress.txt` 记录 `features` / `k` / `batch_size`，从源头杜绝「换参数续传」
- 把 `que_push_fixed.py` / `que_push_random.py` 抽成可复用函数，统一由 `main.py` 编排
- 任务超时重投：用 `ZSET` 记录领取时间，配合巡检脚本把超时任务搬回 `mylist`；或改用 Redis Streams 的消费组
- 结果表增加 `model` / `dataset` / `k` / `created_at` 等元数据列与去重视图
- 用 `BRPOP`/阻塞式读取替代 10ms 轮询，降低空转开销
- 把特征组合搜索拆成「粗筛（小 k）→ 细筛（固定关键特征）」两阶段，降低总量
- 增加端到端冒烟测试（推送 10 条 → 训练 → 入库 → 断言行数）
- 补一份英文 README，并整理实验结论（RF / GB 在不同特征规模下的最优组合）

## 14. 许可证

本项目未声明开源许可证。如用于论文或公开项目，请先与作者确认授权与引用方式。

---

依赖与致谢：[scikit-learn](https://scikit-learn.org/)、[Redis](https://redis.io/)、[DuckDB](https://duckdb.org/)。
