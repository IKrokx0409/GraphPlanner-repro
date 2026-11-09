# 复现记录（交接用）

课程作业：复现 GraphPlanner（ICLR 2026，arXiv 2604.23626）的 Phase 2（动态路由）。目标是认真复现论文的结论和趋势，不追求和论文数字一致；最终要把它接入 LycheeMAS（https://github.com/ecoli-hit/LycheeMAS），课程报告的重点是适配逻辑，测试部分和论文的数字做量级对比。

最后更新：2026-11-03

实验设置、工程措施、各项修复和目前结论的完整说明在 [`results/report/README.md`](../results/report/README.md)（写报告的素材都在 `results/report/`）。本文件只记交接需要的内容：进度、下一步、改动清单和坑。

## 当前状态

| 步骤 | 状态 |
|---|---|
| 模型池（DeepInfra，4 个模型）、API 预算和缓存 | ✅ 完成 |
| 单模型 ×4、random 基线（556 题完整测试集） | ✅ 完成 |
| 训练集单次调用标签（1000 题 × 4 模型） | ✅ 完成 |
| Router-KNN、GraphRouter（官方 GNN，3 个种子） | ✅ 完成 |
| GraphPlanner seed 0、seed 1，w/o History（seed 0）：训练 1000 episode + transductive/inductive 评测 | ✅ 完成 |
| GraphPlanner seed 2：训练 + transductive/inductive 评测 | ✅ 完成（训练 11-05 01:52 结束，评测 12:49 结束） |
| 3 个种子的汇总表（mean±std）和论文 Table 3 对比 | ✅ 报告 README 第 5 节 |
| 训练曲线图 + Pareto 图（`repro/plot_figures.py` → `results/report/figs/`） | ✅ 2026-11-09 |
| 接入 LycheeMAS、框架内功能测试 | ⏳ 未开始 |
| RouterDC（3 个种子，离线，`repro/routerdc_baseline.py`） | ✅ 0.626±0.001（2026-11-07） |
| Router-MLP（3 个种子）、Router-SVM（`repro/router_mlp_svm.py`，离线） | ✅ 0.501±0.003 / 0.502（2026-11-07） |
| Prompt LLM（`repro/prompt_llm_baseline.py`，Router-R1 的 prompt，路由器是 8B） | ✅ 0.385（2026-11-07，花费约 $0.09） |

目前的结果（556 题测试集，准确率 / 平均成本）：

| 方法 | Acc | Cost |
|---|---|---|
| 单模型 Llama-3.1-8B | 0.496 | 54.1 |
| 单模型 Mistral-Nemo-12B | 0.448 | 90.6 |
| 单模型 Llama-3.3-70B（训练集上最好的单模型） | 0.631 | 321.7 |
| 单模型 Mistral-Small-24B（测试集上最好的单模型） | 0.646 | 205.6 |
| random | 0.570 | 1013.0 |
| Router-KNN（多数标签，CV 选出 k=1） | 0.511 | 97.6 |
| GraphRouter（3 个种子） | 0.498±0.003 | 56.9 |
| Router-MLP（3 个种子） / Router-SVM | 0.501±0.003 / 0.502 | 69.6 / 73.2 |
| RouterDC（3 个种子） | 0.626±0.001 | 270.4 |
| Prompt LLM（多轮，路由器 8B） | 0.385 | 1432.9 |
| Router-KNN (score)（加做的变体，k=50） | 0.676 | 244.1 |
| GraphPlanner seed 0（trans / ind） | 0.640 / 0.644 | 300.3 / 361.3 |
| GraphPlanner seed 1（trans / ind） | 0.646 / 0.638 | 902.8 / 797.6 |
| GraphPlanner seed 2（trans / ind） | 0.619 / 0.619 | 4356.1 / 4356.1 |
| **GraphPlanner 3 种子均值（trans / ind）** | **0.635±0.014 / 0.634±0.013**（论文 0.636） | 1853 / 1838 |
| w/o History seed 0（trans / ind） | 0.619 / 0.619 | 362.0 / 362.0 |
| oracle（上界） | 0.761 | 87.3 |

要点：GraphPlanner 比论文式的单轮路由器高约 13 个点，但没有超过最佳单模型和 Router-KNN (score)；历史记忆的效果在噪声范围内（556 题的标准误约 ±0.02）。解读见报告 README。

API 总花费 $1.793（截至 2026-11-07，含 Prompt LLM），硬上限 $8。

## 下一步（按顺序）

1. ~~画训练曲线图~~ 已完成：`python -m repro.plot_figures` 生成 `results/report/figs/training_curves.*`（running reward / 熵 / episode length，4 个 run 叠图）和 `pareto.*`（论文 Figure 4 的复现，全部方法的 Acc-Cost 散点）。
2. 给 `run_phase2.sh` 的 eval 阶段加"输出已存在就跳过"（**只能在脚本没在跑的时候改**，见下方"坑"）。
3. 接入 LycheeMAS。

## 模型池（DeepInfra，`repro/build_config.py` 生成 `config/llm_repro_phase2.json`）

| key | DeepInfra model id | 名义价格（router 看到的成本信号） | 计费价格（输入/输出，每百万 token） |
|---|---|---|---|
| `llama-3.1-8b-instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | 0.2 | $0.02 / $0.04 |
| `mistral-nemo-12b-instruct` | `mistralai/Mistral-Nemo-Instruct-2407` | 0.3 | $0.029 / $0.03 |
| `llama-3.3-70b-instruct` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 0.9 | $0.10 / $0.32 |
| `mistral-small-24b-instruct` | `mistralai/Mistral-Small-24B-Instruct-2501` | 0.6 | $0.05 / $0.08 |

- 名义价格按论文表 10 的同规模模型设定；计费价格只用于预算账本。
- `mistral-small-24b-instruct` 占的是论文里 Mixtral-8x7B 的位置（Mistral 官方文档把 mistral-small-latest 列为 Mixtral 的替代）。原本这个位置是 Llama-4-Scout，试点时持续返回 429 engine_overloaded（最长等了 6547 秒），所以换掉。
- 以前用的 NVIDIA NIM 模型池（nemotron 等）已经弃用，原因是免费账号对大部分模型返回 404、连接不稳定。`results/fixed_nemotron-3.5-bolt-e4b.json` 是那时的结果，只作历史记录。`results/fixed_llama-3.2-90b-instruct.json` 和 `results/fixed_nemotron-3.5-lightning-30b-a3b.json` 是通过 GitHub 网页上传的，内容只有 `[object Object]`，是坏文件，可以删掉。

## 对官方代码的修改

| 修改 | 位置 | 原因 |
|---|---|---|
| API 地址和 key 可配置（`LLM_BASE_URL`，默认 DeepInfra；`DEEPINFRA_API_KEY` 或 `LLM_API_KEYS`） | `shared/utils.py` | 换平台 |
| **预算账本和回答缓存**（SQLite，多进程共享，WAL 模式）。超过 `LLM_BUDGET_USD`（默认 8）后拒绝调用；相同请求直接返回缓存 | `shared/api_guard.py`、`shared/utils.py` | 额度只有 $10 且不能充值；同一种子同一代码重跑时完全从缓存回放，不花钱 |
| **断网时无限等待重试**，单次调用 90 秒硬超时（线程实现）；API 明确拒绝的请求（如 400）重试 3 次后按空答案计分 | `shared/utils.py` | 笔记本会断网、休眠，训练不能因此中断 |
| 增加 `MOCK_LLM=1` 模拟模式 | `shared/utils.py` | 离线测试流程，不花 API 额度 |
| 去掉 `codegemma-7b → gemma-3-27b-it` 的替换；支持在配置中给单个模型加 `max_tokens`、`extra_body` | `shared/utils.py`、`dynamic_router/route_env.py` | 直接调用配置里的模型 |
| **MMLU/GPQA 改回用 `em_mc` 评分** | `shared/response_eval.py` | 见"数据和评测问题"第 1 条 |
| **代码题提取时去掉 markdown 代码块标记** | `shared/response_eval.py` | 见"数据和评测问题"第 5 条 |
| **HumanEval/MBPP 改在独立子进程（`python -I`）里执行** | `shared/evaluation/isolated.py`、`human_eval/execution.py`、`mbpp/mbpp_eval.py` | 原来的 `multiprocessing.Manager` 在多线程进程里 fork 会死锁，评测卡死 |
| **PPO 更新改为逐样本反向传播、累积梯度** | `dynamic_router/network.py`（`RouteAgent.update`） | 原来一个 mini-batch 的计算图同时挂着整张历史记忆图，内存随训练线性增长（每个进程 2.5~3 GB 还在涨）；改后约 1.3 GB 且不再增长。数学上等价（SGD 对比误差 ≤3e-6） |
| **完整续跑存档**：每 25 个 episode 写 `{name}_resume.pt`（网络、优化器、PPO buffer、历史记忆、RNG 等），`--resume` 从中继续 | `dynamic_router/train.py` | 关机不浪费进度；已验证续跑后参数和不中断完全一致 |
| 测试记录里加 `query` 字段 | `dynamic_router/train.py` | 批内结果乱序返回，需要按查询对齐（KNN/GraphRouter 离线评测要用） |
| 可选的成本惩罚 `--cost_alpha`（默认 0，即公开版本的行为） | `dynamic_router/route_env.py` | 论文公式 (4) 有成本项 |
| 缓存解析后的查询嵌入 | `dynamic_router/route_env.py` | 原来每次 reset 都要重新解析约 5000 条字符串（11.5 秒） |
| 测试时按任务/领域统计 accuracy、调用次数和角色使用，并保存每题的回答和标准答案 | `dynamic_router/train.py` | 原来只输出平均 reward |
| 增加 `--no_history`、`--inductive_test`、`--seed` | `dynamic_router/train.py` | 消融和多种子 |

用 `git diff 56010b1` 可以看到相对官方最后一次提交的全部改动。

## `repro/` 脚本

| 脚本 | 作用 |
|---|---|
| `build_config.py` | 生成模型配置 |
| `prepare_data.py` | 去掉训练/测试重合，生成测试子集 |
| `check_api.py` | 每个模型发一条请求 |
| `evaluate.py` | 评测 fixed / random / graphplanner |
| `run_phase2.sh {train,labels,eval}` | 正式实验全流程。已完成的训练和评测会跳过，未完成的训练自动 `--resume`，中断后用同一条命令接着跑 |
| `router_baselines.py`、`graphrouter_baseline.py`、`run_router_baselines.sh` | 用单次调用标签离线评测 Router-KNN 和 GraphRouter（官方 GNN 代码在 `third_party/graphrouter/`，MIT） |
| `router_mlp_svm.py` | Router-MLP / Router-SVM，离线，几分钟 |
| `prompt_llm_baseline.py` | Prompt LLM（多轮），prompt 和流程照 Router-R1 官方代码（收录在 `third_party/router_r1/`，Apache 2.0）；要调 API |
| `routerdc_baseline.py` | RouterDC，按官方代码（shuhao02/RouterDC，commit 168c026，仓库没有 license，所以重写而不是收录）的损失和超参数重新实现；适配点写在文件开头。编码器 `microsoft/mdeberta-v3-base` 已下载到 HF 缓存 |
| `resume_from_latest.py` | 从官方代码每 50 个 episode 保存的 latest checkpoint 加 wandb 日志，拼出续跑存档（用于支持续跑之前就中断的 run） |
| `export_curves.py` | 从 wandb 离线日志导出训练曲线到 `results/report/curves/` |
| `progress_log.py` | 每 5 分钟往 `logs/progress.log` 写一行内存、花费、各进程调用情况 |
| `backup_data.sh` | 把 git 不管的数据（wandb、日志、checkpoint、缓存等）备份到 Windows 盘 |
| `summarize.py` | 汇总结果 JSON |

## 数据和评测问题

1. **MMLU/GPQA 评分失效（已修复）。** 数据文件里这两个任务标注的评分方式是 `em_mc`，但公开代码把它们强制改成了 `cem`（标准答案是否出现在回答里）。对单个字母的选项，这种方式会失效："a" 会被当作英文冠词删掉，于是标准答案为 A 时任何回答（包括空回答）都判对；B/C/D 只要回答里有单词含有这个字母就判对。修复后，从回答中取最后一个 "(X)" 和标准答案比对。
2. **训练集和测试集重合（已处理）。** 公开的训练集包含 GPQA 和 HumanEval 的全部测试题，以及 MBPP 的 20 道测试题。`repro/prepare_data.py` 会从训练集中去掉这 121 条，生成 `data/router_data_train_clean.csv`。
3. **TriviaQA 有标注错误（未处理）。** 比如某道题的标准答案写成 "stillwell"（正确拼法 Stilwell），所有模型都会被判错。对各方法影响相同，未修改。
4. **MATH 要求 `\boxed{}` 格式。** 模型算对了但没写成这个格式，会被判错。这是论文评测本身的设定，未修改。
5. **代码题回答带 markdown 代码块（已修复）。** Mistral-Small 常把代码包在 ```` ```python ```` 里，原来的提取逻辑会把这些标记一起拿去执行，导致语法错误。修复后 Mistral-Small 的 MBPP 从 0.06 变为 0.68、HumanEval 从 0.58 变为 0.82，其他模型不变。
6. **二值效用导致标签大量并列。** 单次调用标签里，大部分题目要么所有模型都对、要么都错，"效用最高、并列选最便宜"的标签有 69% 是 8B。论文式的 Router-KNN 和 GraphRouter 因此几乎全选 8B，这是它们只有约 0.50 的主要原因。

## 运行环境和坑

- **机器：** WSL2 笔记本，15 GB 内存、20 核。4 个训练进程并行时每个约 1.3 GB。
- **休眠：** 只是暂停进程，醒来后自动继续，断网期间 API 调用会一直等待重试。
- **关机：** 会杀掉进程，`/tmp` 也会清空（包括 Claude Code 的 scratchpad，不要把要长期保留的脚本放那里）。开机后重新运行 `bash repro/run_phase2.sh train` 即可从最近的续跑存档（最多丢 25 个 episode）继续。
- **临时暂停：** 用 `kill -STOP <pid>` / `kill -CONT <pid>`。
- **不要在 bash 脚本运行时修改它。** bash 是边读边执行的，改了会让正在跑的脚本执行错位。
- **不要用 `pkill -f` 匹配命令行。** 模式很容易匹配到执行命令的 shell 本身，把自己杀掉；按 PID 杀。
- **API key 读取。** Ubuntu 默认的 `~/.bashrc` 开头有一段判断，非交互式 shell（包括 Claude Code 执行的命令）读到那里就返回，写在文件末尾的 export 不会生效。在命令前加 `eval "$(grep '^export DEEPINFRA_API_KEY=' ~/.bashrc)"`。
- **RouterDC 的内存。** batch 64 × 512 token 直接训练会超出 15 GB 内存被系统杀掉（exit 137），所以改成先无梯度编码整批、再按长度分成 8 条一组反传（gradient cache，已验证与整批梯度一致，相对误差 4e-5）。单进程峰值约 13 GB，跑它时不要同时跑别的训练。tokenizer 需要 `sentencepiece`（已加进 requirements）。
- **seed 2 为什么慢。** seed 0/1 在第 300~400 个 episode 学会了"一步调用强模型"（每个 episode 1~1.5 步，8~25 秒）；seed 2 收敛到"规划 → 执行子任务 → 汇总"的完整工作流（每个 episode 约 10 步，80~100 秒）。数值正常（reward 和 seed 0/1 同水平，熵平滑下降，无 NaN），属于 RL 的种子方差：`cost_alpha=0` 时工作流长短不影响 reward。瓶颈是 API 等待和每步一次的 Longformer 嵌入，不是 CPU 抢占，加线程无效（历史图前向 3 线程和 12 线程都是 0.06 秒）。
