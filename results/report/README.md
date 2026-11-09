# 报告素材（2026-11-02 记录）

本目录存放写课程报告用的数据，以及实验过程中的关键记录。每个数字都注明了出处，方便在报告里引用。

## 1. 模型池与选择理由

LLM 全部通过 DeepInfra 的 OpenAI 兼容 API 调用，配置由 `repro/build_config.py` 生成到 `config/llm_repro_phase2.json`。

| 本组模型（DeepInfra id） | 对应论文模型 / 档位 | 名义价格（路由器成本信号，Table 10） | 实际计费（$/1M token，输入/输出） |
|---|---|---|---|
| meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo | Llama-3.1-8B（论文原模型） | 0.2 | 0.02 / 0.04 |
| mistralai/Mistral-Nemo-Instruct-2407 | Mistral-Nemo-12B（论文原模型） | 0.3 | 0.029 / 0.03 |
| meta-llama/Llama-3.3-70B-Instruct-Turbo | 70B 档（Nemotron-Super-49B / Nemotron-51B / ChatQA-1.5-70B，均源自 Llama-3.x-70B） | 0.9 | 0.10 / 0.32 |
| mistralai/Mistral-Small-24B-Instruct-2501 | Mixtral-8x7B 的位置 | 0.6 | 0.05 / 0.08 |

- 两个论文原模型的描述文本和 embedding 直接沿用官方配置。新增模型的描述按官方格式编写，用同一个 Longformer 做 embedding。用这个流程重算官方描述，与官方存储的 embedding 余弦相似度约为 1.0。
- 带 "Turbo" 的模型是 FP8 量化版本，DeepInfra 只提供这一种。
- 论文 Phase 2 有 5 个模型（官方代码 `items[:3]+items[4:-8]`）。本组用 4 个：Qwen2.5-7B 在 DeepInfra 上不可用；Mixtral-8x22B 在官方代码里属于 held-out 模型，不参与 Phase 2。
- **Mistral-Small-24B 的选择理由：** Mistral 官方文档把 mistral-small-latest 列为 Mixtral-8x7B 的替代模型（Mixtral 于 2024-11-30 弃用，2025-03-30 退役），当时 mistral-small-latest 对应 Small 3（2501）。名义价格沿用 Mixtral 档的 0.6。它不是 MoE 架构，报告里要说明。
- **为什么放弃 Llama-4-Scout（最初的候选）：** 第一次试点中，DeepInfra 对它累计返回 91 次 429 `engine_overloaded`（"Model busy, retry later"）和 45 次超时，单次调用最长等了 6547 秒，导致试点停滞约 3 小时。服务端间歇性过载会让长时间训练卡住，所以换掉。

### 候选模型稳定性探测（`pilot/model_probe.py`）

每个模型 12 次请求，并发 4，`max_tokens` 256，同一道算术题：

| 模型 | 成功 | 中位延迟 | 最大延迟 | 输出 token（中位） |
|---|---|---|---|---|
| Llama-4-Scout-17B-16E | 12/12 | 5.1s | 5.6s | 81 |
| Mistral-Small-24B-2501 | 12/12 | 3.6s | 3.9s | 180 |
| Mistral-Small-3.2-24B-2506 | 12/12 | 6.5s | 13.3s | 208 |
| gemma-4-26B-A4B | 12/12 | 3.2s | 5.3s | 166 |
| Qwen3-235B-A22B-2507 | 12/12 | 13.2s | 18.4s | 132 |

Scout 在探测时已经恢复，但吞吐只有约 16 token/s，说明仍处于高负载状态。

## 2. 工程措施：API 预算、缓存与断网处理

详见 `router_planner/shared/api_guard.py` 和 `shared/utils.py`。

- **花费账本**（SQLite，`cache/api_spend.sqlite`）：根据 API 返回的实际 token 用量和计费单价记账，多个进程共享，花费达到 `LLM_BUDGET_USD`（默认 $8）后拒绝继续调用。
- **回答缓存**（`cache/llm_responses.sqlite`）：完全相同的请求（模型、提示词和采样参数都相同）直接返回缓存结果。不同种子之间也共享这个缓存，同一段提示词不会重新采样。温度是 0.1，影响很小，但需要披露。
- **断网等待：** 遇到连接错误、429、5xx 或单次调用超过 90 秒时，按指数退避一直重试，直到网络恢复，不会把这次调用当成答错。这样网络问题不会污染奖励信号。
- **其他错误：** 400 类错误最多重试 3 次，仍失败的调用计为空答案并记入失败数。401/403/404 和余额不足（402）会直接终止运行。

- **代码题执行改用独立解释器**（`router_planner/shared/evaluation/isolated.py`）：官方 HumanEval/MBPP 评测通过 `multiprocessing.Manager` 加 fork 运行生成的代码。评测进程本身是多线程的（批量 API 调用），fork 时如果其他线程正持有锁，子进程会死锁，父进程也跟着永久卡住。第一次正式运行中，单模型基线评测就因此卡了 15 分钟。现在改为 `subprocess` 启动新的解释器（fork+exec，线程安全），超时直接杀掉子进程。用 8 个线程同时活跃的情况验证过：164 道 HumanEval 标准答案和 100 道 MBPP 参考答案全部通过，死循环、`input()`、`os._exit` 都能正确判为失败或超时。所有正式结果都在这次修复后从头重跑（第一次运行的日志在 `logs/aborted_run1/`）。
- **代码答案提取时去掉 Markdown 代码块标记**（`shared/response_eval.py` 中的 `_strip_markdown_fence`）：
  - **问题：** 提示词要求模型按 `[BEGIN] <code> [Done]` 格式输出代码。Mistral-Small-24B 经常在标签里面再包一层 ```python 代码块，官方提取逻辑会把这一行标记当成代码执行，直接语法错误。
  - **修复：** 提取出代码后，如果其中有代码块标记，只取代码块内部的内容；没有标签但有代码块的回答也按同样方式处理。
  - **影响：** 在完整测试集的单模型结果上重新评分，Mistral-Small 的 MBPP 从 0.06 变为 0.68，HumanEval 从 0.58 变为 0.82（50 道 MBPP 中有 45 道带代码块标记，50 道 HumanEval 中有 16 道）。其他 3 个模型的代码分数完全不变。
  - **处理：** 这个问题会污染训练奖励，让路由器误以为 Mistral-Small 不会写代码，所以第二次运行在约 1.5 小时后中止，修复后从头重跑（日志和结果在 `logs/aborted_run2/`、`results/aborted_run2/`）。这是继 MMLU/GPQA 评分问题之后，又一个"评测流程误判模型能力"的例子，可以写进报告的适配难点部分。

- **PPO 更新改为逐样本反向传播**（`dynamic_router/network.py` 中的 `RouteAgent.update`）：
  - **问题：** 官方实现每个 mini-batch（32 个样本）的 policy 和 value 都要做一次前向，每次前向都在**整个**历史记忆上建图。这些计算图会一直保留到最后统一做 backward，所以峰值内存约等于 64 × 历史图大小，而历史图随训练线性增长。
  - **实测：** 第三次运行中，GraphPlanner 进程的内存每小时增长约 0.3 GB。w/o History 进程的记忆为空，内存不涨，这正好印证了上面的原因。按此推算，到第 1000 个 episode 时每个进程约需 6 GB，4 个进程并行会超出 15 GB 内存。
  - **修复：** 每个样本单独 backward，梯度累加，最后按原来的方式调用一次 optimizer.step。损失本来就是 batch 内的均值，所以把每个样本的损失除以 bs 再累加，得到的梯度和原来完全相同。
  - **验证：** 在模拟训练里截取第一次 PPO 更新，用同一份 buffer 分别跑新旧两种实现。换成 SGD 时，两者每一步参数更新的相对差异最大为 3e-6，属于浮点舍入误差。用 Adam 时差异会被放大，因为 Adam 第一步约等于 lr·sign(g)，梯度接近 0 的参数对舍入误差很敏感。但这不是算法上的差别。
  - **处理：** 第三次运行（日志在 `logs/aborted_run3/`）在约 250 个 episode 时中止，修复后于 15:25 从头重跑。已经完成的单模型基线结果与 PPO 无关，予以保留。

- **断点续跑**（`train.py` 新增 `--resume` 和 `--checkpoint_every`，默认每 25 个 episode 存一次）：
  - **为什么需要：** 实验在个人笔记本上运行。休眠只会暂停进程，2026-11-02 16:50–22:10 休眠了 5.3 小时，醒来后自动继续，没有任何损失。但关机、重启或 Windows 更新会直接杀掉进程，而官方代码不支持从中途恢复。
  - **做法：** 定期把完整训练状态写入 `{experiment_name}_resume.pt`，包括三个网络和两个优化器、经验 buffer、历史记忆、running reward 和最佳值、三个随机数生成器的状态，以及下一个 episode 的编号。写入时先写临时文件再原子替换，所以写到一半关机也不会损坏旧存档。
  - **验证：** 用模拟数据跑两组，一组不中断跑完 20 个 episode，另一组在第 12 个 episode 中断、从第 10 个 episode 的存档恢复。最终参数差异为 0.0，历史记忆完全相同。恢复之后，已经付费的 LLM 回答都会命中缓存。
  - **注意：** 第四次运行（15:25 启动）的进程用的是加入续跑功能之前的代码，不会写存档。如果它被中断，需要从头重跑：已有回答从缓存重放，本地计算约每分钟 2–3 个 episode，不会多花钱。从那次重启起就会带上 `--resume`。

- **2026-11-03 05:00 的计划关机**：
  - **中断时的状态：** seed 0 已经跑完全部 1000 个 episode，不受影响。其余三个进程启动时还没有续跑功能，被停止时分别在 seed 1 的第 381 个、seed 2 的第 240 个、w/o History 的第 465 个 episode。
  - **如何拼出续跑存档：** 用 `repro/resume_from_latest.py`，把官方代码每 50 个 episode 自动保存的 `latest` checkpoint（网络、两个优化器、历史记忆），和 wandb 离线日志里逐 episode 记录的 running reward、best、episode reward 合在一起，生成 `--resume` 用的存档。恢复点分别是 seed 1 第 351 个、seed 2 第 201 个、w/o History 第 451 个 episode，分别丢掉了 31、40、15 个 episode。
  - **与不中断运行的差别：**
    1. PPO buffer 从空开始，大约 8 个 episode 后才进行下一次更新；
    2. 随机数状态没有保存下来，改用 (seed, episode) 重新设定种子，所以之后抽到的训练查询和原轨迹不同，但仍是从同一训练集均匀抽样；
    3. seed 1 的 best 文件来自被丢弃的那段 episode，已改名为 `*_best_lostbranch_*` 保留，暂时用第 350 个 episode 的 checkpoint 作为 best。
  - **验证：** 用模拟 LLM 从 seed 1 的存档续跑了 2 个 episode，状态正常加载，记忆从 1823 条增加到 1825 条。
  - **另一个选择：** 不拼存档、从缓存精确重放，结果会和不中断时完全一致，但要多花约 3 小时本地计算。用户不希望浪费这 3 小时，所以采用了上面的做法。

## 3. 试点实测（2026-11-02，新模型池，使用全新缓存）

| 项目 | 规模 | 花费 | 墙钟时间 | 失败 / 断网重试 |
|---|---|---|---|---|
| 随机策略评测（`pilot/pilot_random44.json`） | 44 条（每个任务 4 条），批大小 4 | $0.0052 | 599 s | 0 / 0 |
| GraphPlanner 训练（`pilot/pilot_train20.log`） | 20 个 episode，批大小 1 | $0.0050 | 1187 s（约 59 s/episode，其中一次约 6 分钟断网） | 0 / 20 |

随机策略在这 44 条上的准确率为 0.568，平均每条查询调用 LLM 3.98 次。

各模型平均每次调用的 token 数：

| 模型 | 随机评测：调用次数 / 输入 / 输出 | 训练 20 ep：调用次数 / 输入 / 输出 |
|---|---|---|
| Llama-3.3-70B | 40 / 304 / 137 | 39 / 412 / 118 |
| Llama-3.1-8B | 43 / 291 / 104 | 23 / 336 / 69 |
| Mistral-Nemo | 32 / 340 / 122 | 25 / 383 / 30 |
| Mistral-Small-24B | 42 / 479 / 100 | 41 / 525 / 90 |

由此推算正式实验的开销：每个 1000-episode 训练约 $0.25，完整测试集上每次多轮评测约 $0.07。全部实验预计约 $2，即使路由器偏向 70B 也在 $4 以内。本地计算耗时的估计需要更正：

- 原先的 5–8 s/episode 是用 MOCK_LLM 测的。模拟回答很短，Longformer 的输入也就很短，所以偏乐观。
- 第三次正式运行开头，episode 全部命中回答缓存，不发生任何 API 调用。此时 4 个训练进程并行（各 3 个线程），仍需约 43 s/episode。可见真实回答下，本地计算（对回答做 Longformer 编码、PPO 更新）才是主要耗时。
- 当时没有改用 GPU，依据的就是那个偏乐观的估计。如果以后要加快，GPU 确实能帮上忙。训练结束后用实际墙钟时间更新这一段。

## 4. 正式实验（`repro/run_phase2.sh`）

| 阶段 | 内容 | 输出 |
|---|---|---|
| train | GraphPlanner 种子 0/1/2，以及 w/o History 消融（种子 0），每个 1000 episode × 批大小 1，并行运行；同时在 556 条测试集上评测 4 个单模型基线和随机基线 | `checkpoints_phase2/`、`logs/train_*.log`、`wandb/offline-run-*`（逐 episode 曲线）、`results/fixed_*.json`、`results/random.json` |
| labels | 4 个模型分别单次回答 1000 条均匀抽样的训练查询（random_state=0），作为 Router-KNN / GraphRouter 的训练标签；测试集标签就是 fixed 基线的结果 | `results/labels_train1000_*.json` |
| eval | 每个 checkpoint 做 transductive 和 inductive 两种评测 | `results/gp_*_{trans,ind}.json` |

| router | 离线单轮路由基线（`repro/run_router_baselines.sh`，不调用 API） | `results/router_baselines.json`、`results/graphrouter.json` |

### 单轮路由基线的做法

- **为什么能离线评测：** 单轮路由器每条查询只选一个模型、调用一次，所以它在测试集上的结果，就等于它选中的那个模型在这条查询上的单次调用结果。训练集用 labels 阶段的结果，测试集用 fixed 基线的结果。
- **训练标签：** 按 Performance First 的设定，取效果最高的模型；有并列时选更便宜的那个。
- **Router-KNN：** 用 GraphPlanner 同一套 Longformer 查询 embedding，按余弦相似度找 k 个最近邻，投票选出模型，并列时选更便宜的。k 在 {1,3,5,10,20,50} 中用训练集 5 折交叉验证来选。
- **GraphRouter：**
  - 模型、训练循环和超参数都用官方代码（ulab-uiuc/GraphRouter，commit 0edfb87，MIT 许可，原样收录在 `repro/third_party/graphrouter/`）。超参数为 1000 epoch、lr 3e-4、wd 1e-4、mask rate 0.5、batch 32、embedding_dim 8、edge_dim 3。
  - 适配之处：用我们固定的训练集和测试集，代替官方按比例的随机划分；从训练查询中按任务留出 10% 做验证集；测试时加载验证集 macro-F1 最高的 checkpoint（官方保存了这个 checkpoint，但测试时用的是最后一个 epoch）；任务描述按官方风格为 11 个任务编写，用同一个 Longformer 编码；成本按任务做 min-max 归一化，只用训练边拟合。
  - 跑 3 个种子，报告均值和标准差。
- **Router-MLP / Router-SVM（`repro/router_mlp_svm.py`，2026-11-07）：** 标签和 Router-KNN 相同。特征是 Longformer 查询 embedding 加任务 one-hot（论文描述两者都用到查询特征和任务信息），按训练集标准化。超参数用 5 折交叉验证按路由准确率选择：MLP 选出单隐层 256、L2 1e-4（sklearn MLPClassifier，Adam，early stopping），跑 3 个种子；SVM 是 RBF 核，选出 C=10、gamma=1e-3，结果是确定性的，只跑 1 次。
- **RouterDC（`repro/routerdc_baseline.py`，2026-11-05～07）：**
  - 按官方代码（shuhao02/RouterDC，commit 168c026）重新实现。官方仓库没有 license，所以不收录代码。
  - 和官方相同的部分：编码器是 mdeberta-v3-base，取首 token，和每个 LLM 的可学习向量算余弦相似度；sample-LLM 对比损失（top_k=3、last_k=3）；聚类对比损失（预训练编码器 embedding → 5 维 t-SNE → k-means 8 类，权重 1）；AdamW，lr 5e-5，batch 64，最长 512 token。
  - 适配之处：
    - **训练长度：** 官方是 1000 步、5000 个样本，即 12.8 个 epoch。本组训练集约 900 题，保持 epoch 数，改为 180 步。
    - **内存：** batch 64 × 512 token 在 CPU 上放不进 15 GB 内存。改为先整批无梯度编码、算出 loss 对 embedding 的梯度，再按长度分成 8 条一组重新编码反传（gradient cache），已验证与整批梯度一致。
    - **选模型：** 和 GraphRouter 一样，每个任务留出 10% 做验证集，每 10 步评估一次，取验证集最好的那一步。官方是用训练集准确率挑选。
  - 3 个种子在 CPU 上各约 2.5 h（不含休眠）。选中的步数分别是 150、50、110。
- **Prompt LLM（`repro/prompt_llm_baseline.py`，2026-11-07，多轮）：**
  - prompt 和流程照 Router-R1 官方代码（ulab-uiuc/Router-R1，commit 801a240，Apache 2.0，prompt 收录在 `repro/third_party/router_r1/prompts.py`）：路由器读带模型描述的模板，每轮写 `<think>` 后用 `<search> 模型名: 子问题 </search>` 调用一个模型，或用 `<answer>` 作答；被调用的模型收到官方的 AGENT_PROMPT（最多 512 token，温度 0）；最多 4 轮调用，之后再给一轮只能作答（同官方 test.sh 的 max_turns=4）。
  - 适配之处：官方用 3B 模型当路由器，本组池子里没有，改用最小的 Llama-3.1-8B，温度 0；官方是在模型自己的输出后直接续写，API 上改为把整个对话记录作为一条 user 消息发送；题目用 GraphPlanner 执行器收到的同一份任务 prompt（含答题格式要求），评分用同一个 `eval_perf`；成本和调用次数把路由器自己的轮次也算进去，用和 GraphPlanner 相同的 token 计数和名义价格。
  - 556 题，8 个并发，约 35 分钟，花费约 $0.09。
- **结果与敏感性分析（2026-11-03）：**
  - **训练标签的分布：** 69% 是 8B，Nemo 15%，Mistral-Small 11%，70B 5%。原因是效用只有对/错两种取值，大部分查询上各个模型并列（要么都对，要么都错），而并列时规则选最便宜的模型。
  - **按论文描述实现的两个路由器：** Router-KNN（多数标签投票，CV 选出 k=1）的准确率是 0.511；GraphRouter 三个种子的准确率是 0.498±0.003，其中 95% 的查询都路由给了 8B。两者都和论文 Table 3 里单轮路由器的水平相当（49.7%–54.3%）。
  - **加做的变体 Router-KNN (score)：** 改为对每个模型分别求 k 个近邻上的平均效用，再选最高的那个。CV 选出 k=50（候选范围 1 到 400，最优点落在范围中间）。测试准确率 0.676，约一半路由给 70B、一半给 Mistral-Small，**高于最佳单模型（0.646）和 GraphPlanner seed 0（0.640）**。
  - **解读：** 二值效用下，"多数标签"这种训练方式丢掉了大量信息。报告里要如实说明：本组设置下，一个简单的基于效用的 KNN 就超过了 GraphPlanner。
- **两个参照：** best single（按训练集准确率选出的单个模型）和 oracle（每条查询都选对的上界）。
- **流程上的一个说明：** 论文里每条记录不带查询 ID，而批内结果是乱序返回的，所以无法按查询对齐。本组在评测记录里加了 `query` 字段。8B、Nemo、70B 三个 fixed 基线是在加字段之前跑的，之后从回答缓存重算了一遍（不产生新的 API 花费），重算前后的准确率对比记录在 after_labels 的输出里。

训练从 2026-11-02 10:38（PST）开始（10:08 的第一次运行因为代码执行死锁中止）。结果汇总、耗时和花费见第 5 节。

**训练曲线：** `curves/{gp_seed0,gp_seed1,gp_seed2,gp_nohist_seed0}.csv`，由 `python -m repro.export_curves` 从 wandb 离线日志导出，每行一个 episode，包括 reward、running reward、步数、熵、PPO loss、API 调用次数和花费。续跑过的 run 只保留到续跑恢复点为止的旧记录（被丢掉的分支不导出），接上续跑之后的记录；`wandb_run` 列标明每行出自哪个 run。seed 2 训练结束后重新导出一次。中止过的旧 run 原样保留在 `wandb/aborted_run{1,2,3}/`。

**图（`figs/`，由 `python -m repro.plot_figures` 生成，png + pdf）：**

- `training_curves.*`：Phase-2 训练曲线，三个种子加 w/o History（种子 0）叠在一张图里，三个子图分别是 running reward、policy entropy、episode length（后两者为 25-episode 滑动平均）。论文没有逐 episode 的训练曲线图，这是复现过程自己记录的；熵是论文描述的早停信号（"熵降到阈值以下就早停"），episode length 展示了 seed 2 收敛到长工作流的现象（见 NOTES "seed 2 为什么慢"）。
- `pareto.*`：复现论文 **Figure 4 的 Pareto 曲线**（准确率 vs 平均成本，全部方法画在一张图里，GraphPlanner 高亮，含 oracle 上界和 Pareto 前沿）。横轴用对数刻度。准确率和成本都从结果 JSON 读取，不写死。与论文的差别：论文通过扫描成本权重 α 画出 GraphPlanner 的一条前沿曲线，本组只跑了公开默认值（`cost_alpha=0`），所以 GraphPlanner 是各个种子的散点而非 α 曲线——三个种子的成本本身已经跨越很大范围（seed 0 约 300，seed 2 约 4356）。

## 5. 结果汇总与论文 Table 3 对比（2026-11-05，全部实验完成）

556 题测试集，各列为分类别准确率。Avg 是全部 556 题的总体准确率（按题数加权；论文 Average 列也不等于 5 类的简单平均，例如 GraphPlanner 5 类平均为 62.2% 而表中是 63.6%，应同样是按题数加权），Cost 用论文 Table 10 的名义价格。论文各行来自 Table 3（Phase 2）。本组 GraphPlanner 的数据来自 `python -m repro.summarize results/gp_seed{0,1,2}_*.json`，单轮路由器来自 `results/{router_baselines,router_mlp_svm,routerdc,graphrouter}.json`。

| 方法 | Math | Code | CS | WK | Popular | Avg Acc | Avg Cost | LLM 调用/题 |
|---|---|---|---|---|---|---|---|---|
| **论文** GraphPlanner | 67.0% | 76.0% | 78.0% | 38.0% | 52.0% | 63.6% | 605.0 | 8.1 |
| **论文** Router-KNN | 40.4% | 66.0% | 82.0% | 27.0% | 17.0% | 49.7% | 169.2 | 1 |
| **论文** Router-MLP | 43.3% | 67.0% | 82.0% | 25.0% | 7.0% | 48.2% | 168.4 | 1 |
| **论文** Router-SVM | 38.6% | 58.0% | 78.0% | 23.0% | 13.0% | 45.4% | 179.8 | 1 |
| **论文** RouterDC | 57.6% | 51.0% | 79.3% | 32.0% | 39.0% | 54.3% | 138.7 | 1 |
| **论文** GraphRouter | 53.2% | 59.0% | 82.7% | 28.0% | 21.0% | 51.9% | 178.4 | 1 |
| **论文** Prompt LLM | 37.7% | 56.0% | 76.2% | 24.0% | 10.0% | 40.8% | 1070.4 | 12.5 |
| **论文** Router-R1（最好的多轮基线） | 45.3% | 52.0% | 81.2% | 28.6% | 37.2% | 51.8% | 76.3 | 1.8 |
| 本组 GraphPlanner transductive（3 种子） | 69.2±3.0 | 68.3±5.8 | 84.2±5.1 | 38.7±5.7 | 46.3±4.9 | **63.5±1.4** | 1853±2189 | 4.3±4.7 |
| 本组 GraphPlanner inductive（3 种子） | 67.9±1.0 | 68.3±5.8 | 84.2±5.1 | 38.7±5.7 | 47.0±4.4 | **63.4±1.3** | 1838±2191 | 4.3±4.8 |
| 本组 w/o History（seed 0，trans = ind） | 65.1% | 65.0% | 85.3% | 39.0% | 43.0% | 61.9% | 362.0 | 1.73 |
| 本组 Router-KNN（k=1） | 49.1% | 65.0% | 70.7% | 34.0% | 27.0% | 51.1% | 97.6 | 1 |
| 本组 Router-MLP（3 种子） | 49.1±1.6 | 64.0±1.7 | 71.1±1.0 | 29.7±2.1 | 26.3±0.6 | 50.1±0.4 | 69.6 | 1 |
| 本组 Router-SVM（确定性，1 次） | 49.1% | 65.0% | 71.3% | 30.0% | 25.0% | 50.2% | 73.2 | 1 |
| 本组 RouterDC（3 种子） | 64.8±0.5 | 65.0±1.7 | 86.0±1.8 | 43.7±2.3 | 42.0±1.7 | 62.6±0.1 | 270.4 | 1 |
| 本组 GraphRouter（3 种子） | 47.5±2.2 | 62.3±0.6 | 71.3±0.0 | 30.0±0.0 | 27.3±1.2 | 49.8±0.3 | 56.9 | 1 |
| 本组 Prompt LLM（路由器 8B） | 30.2% | 21.0% | 71.3% | 32.0% | 22.0% | 38.5% | 1432.9 | 4.98 |
| 本组 Router-KNN (score)，k=50（加做） | 70.8% | 79.0% | 88.7% | 42.0% | 47.0% | 67.6% | 244.1 | 1 |
| 本组最好单模型 Mistral-Small-24B | 67.0% | 75.0% | 88.7% | 34.0% | 46.0% | 64.6% | 205.6 | 1 |
| 本组 random | 57.5% | 66.0% | 78.0% | 37.0% | 36.0% | 57.0% | 1013.0 | 4.56 |

所有 3 种子的行都是 mean±std（样本标准差）。路由基线脚本写进 JSON 的 `std_acc` 是总体标准差，所以表中数字比 JSON 里大约 1.22 倍。

各种子的明细：

| 种子 | trans Avg / Cost / 调用 | ind Avg / Cost / 调用 | 收敛到的主要工作流（trans） |
|---|---|---|---|
| 0 | 64.0% / 300.3 / 1.32 | 64.4% / 361.3 / 1.34 | 532/556 题只调一次 Mistral-Small executor |
| 1 | 64.6% / 902.8 / 1.90 | 63.8% / 797.6 / 1.71 | 500/556 题只调一次 70B executor，其余是 70B 多轮 |
| 2 | 61.9% / 4356.1 / 9.79 | 61.9% / 4356.1 / 9.79 | 全部是 70B 的 planner→executor→summarizer 长工作流（约 10 步） |

**与论文的对比结论：**

- **平均准确率与论文吻合：** 3 种子均值 63.5%（trans）、63.4%（ind），论文 63.6%。分类别看，Math、WK 与论文接近，CS 高约 6 个点，Code、Popular 低 5～8 个点，但种子间标准差有 5 个点左右，差异大多在噪声内。
- **相对"多数标签"类单轮路由器的优势复现了：** Router-KNN、Router-MLP、Router-SVM、GraphRouter 在本组都在 50%～51%，和论文的 45.4%～51.9% 同一量级；GraphPlanner 比它们高 12～13 个点（论文高 12～18 个点）。这 4 个方法都用"效用最高、并列选最便宜"的单一标签训练，而二值效用下 69% 的训练标签是 8B，所以它们有约 90% 的题选 8B（MLP 89%、SVM 89%、GraphRouter 95%），准确率接近 8B 单模型（49.6%）。
- **相对 RouterDC 的优势没有复现：** RouterDC（3 种子 62.6±0.1%）只比 GraphPlanner 低 0.9 个点，在噪声范围内；论文里差 9.3 个点（54.3% 对 63.6%）。RouterDC 的对比损失直接使用每个模型各自的得分（正样本是答对的模型，负样本是答错的模型），不会把并列压成"选最便宜的"，所以它学到 83% 的题选 70B，基本等价于"选最强的单模型"。它和加做的 Router-KNN (score)（67.6%）说明的是同一件事：本组设置下，用好每个模型的得分信息，单轮路由就已经能达到或超过 GraphPlanner。
- **Prompt LLM 复现了"明显不如 GraphPlanner"：** 本组 38.5%，论文 40.8%，都是表中最低的一档，比 GraphPlanner 低约 25 个点。路由器（8B）平均每题 3.1 轮、调用其他模型 1.9 次，被调用的模型里 69% 是它自己（8B）、19% 是 Nemo、只有 11% 是 70B，很少把问题交给更强的模型。有 10% 的题到最后一轮也没给出 `<answer>`；另有 45 道选择题只写了字母没加括号、16 道 MATH 没写 `\boxed{}`，按统一评分判错（如果放宽这 45 道选择题，会多对 22 道，约 +4 个点，仍远低于其他方法）。本组的调用次数（4.98，含路由器自己的轮次）少于论文的 12.5，成本（1433）高于论文的 1070，因为每一轮都要重新发送完整的长 prompt。
- **没有复现的部分：** 本组模型池里最好的单模型（Mistral-Small-24B，64.6%）和加做的 Router-KNN (score)（67.6%）都不低于 GraphPlanner。论文 Table 3 没有列单模型结果，所以这一点论文没有直接对照。
- **历史记忆与 inductive/transductive：** 论文 Figure 5 中 w/o History 明显下降、transductive 略好于 inductive。本组 w/o History 比 3 种子均值低 1.6 个点（单种子，556 题的标准误约 ±2 个点）；trans 与 ind 只差 0.1 个点。方向与论文一致，但幅度在噪声范围内。seed 2 的 trans 与 ind 556 题的工作流完全相同：它的策略几乎是确定性的（训练末期熵约 0.07），4974 条历史记忆已加载（见评测日志）但没有改变任何决策。
- **成本与工作流长度的种子差异很大：** 训练用的是公开代码的默认设置 `cost_alpha=0`（奖励里没有成本项），所以策略不会因为工作流长而被惩罚。seed 0/1 收敛到"基本只调一次强模型"，seed 2 收敛到约 10 步的长工作流，成本高一个数量级而准确率没有提高。论文的 8.1 次调用/题介于两者之间。这说明在没有成本惩罚时，工作流长度主要由随机种子决定，不是由任务决定。

**实际耗时（墙钟，PST）：** 4 个训练进程在 2026-11-02 10:38 同时开始（CPU 训练，LLM 走 API），seed 0 于 11-03 03:42 结束（约 17 h），w/o History 于 11-03 12:52（约 26 h），seed 1 于 11-03 13:37（约 27 h），seed 2 于 11-05 01:52（约 63 h）。seed 2 慢的原因：长工作流每个 episode 约 10 次 API 调用和 Longformer 编码；期间还经历了笔记本休眠、多次断网（按设计无限重试）和一次卡死（从 975 episode 续跑，命中缓存不花钱）。seed 2 的两次评测各约 9 h 和 1.5 h（inductive 全部命中缓存）。

**API 花费（`cache/api_spend.sqlite`，含试点和全部正式实验，截至 2026-11-07）：** 共 $1.793（37492 次调用，预算上限 $8）。其中 Prompt LLM 基线（含 22 题试点）约 $0.09，其余离线基线不花钱。

| 模型 | 调用次数 | 输入 / 输出 token | 花费 |
|---|---|---|---|
| Llama-3.3-70B-Turbo | 19756 | 6.86M / 2.29M | $1.419 |
| Mistral-Small-24B-2501 | 6344 | 2.96M / 0.66M | $0.201 |
| Mistral-Nemo-12B | 4598 | 1.33M / 0.43M | $0.052 |
| Llama-3.1-8B-Turbo | 6704 | 4.05M / 0.89M | $0.116 |
| 候选模型探测（Qwen3-235B、Gemma-4、Llama-4-Scout、Mistral-Small-3.2） | 90 | — | $0.007 |
