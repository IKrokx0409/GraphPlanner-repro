# 复现记录（交接用）

课程作业：复现 GraphPlanner（ICLR 2026，arXiv 2604.23626）。目标是验证论文的**结论和趋势**（learned routing 是否优于 single-LLM 和 random routing、历史记忆是否有用），不追求和论文数字一致。

最后更新：2026-10-30

## 当前状态

| 步骤 | 状态 |
|---|---|
| 环境、数据处理、模型配置 | ✅ 完成 |
| 用 mock LLM 跑通训练、评测和汇总 | ✅ 完成 |
| API 调通（4 个模型） | ✅ 完成 |
| Baseline：`nemotron-3.5-bolt-e4b` | ✅ 完成，结果见 `results/fixed_nemotron-3.5-bolt-e4b.json` |
| Baseline：`nemotron-3.5-lightning-30b-a3b` | ❌ 跑到一半中断，且有连接失败，需要用较低并发重跑（见"已知问题"） |
| Baseline：`llama-3.2-90b-instruct`、`minimax-m3.1-flash`、random | ⏳ 未跑 |
| GraphPlanner 主实验训练 | ⏳ 未开始 |
| 消融（`--no_history`）、inductive/transductive 对比 | ⏳ 未开始 |

已完成的 baseline 结果（110 题测试集）：

| 方法 | Math | Code | CS | WK | Popular | Avg Acc | Avg Cost |
|---|---|---|---|---|---|---|---|
| fixed: nemotron-3.5-bolt-e4b | 60.0% | 45.0% | 93.3% | 10.0% | 35.0% | 52.7% | 41.3 |

## 下一步（按顺序）

具体命令见 README 的"Quick start for this reproduction"。

1. 补跑剩下的 baseline。`nemotron-3.5-lightning-30b-a3b` 用 `--batch_size 2` 跑。
2. 检查各 baseline 的结果 JSON：`records` 里保存了每题的原始回答（`response`）、标准答案（`gt`）和得分（`utility`），抽查答案抽取和评分是否正常。
3. 根据 baseline 的实际调用量和 cost，估算主实验的花费。
4. 训练 GraphPlanner 主实验。训练时要看日志里的 reward 曲线，确认 router 确实在学习，比如不是每次都选同一个模型。
5. 评测训练好的模型（transductive 和 `--inductive` 各一次），并和 baseline 比较。
6. 视时间和额度做 `--no_history` 消融。

## 模型池

论文用的模型大多已从 NVIDIA NIM 下线。另外，**免费账号对目录里大部分模型都返回 404**（"Function ... Not found for account"），包括一开始选的替代模型（mistral-7b、gemma-3-12b、granite-3.0-8b 等）。2026-10-30 实测后，选定以下 4 个稳定可用的模型：

| key | NIM model id | 说明 |
|---|---|---|
| `nemotron-3.5-bolt-e4b` | `nvidia/nemotron-3.5-bolt-e4b` | 小，约 1 秒/次 |
| `nemotron-3.5-lightning-30b-a3b` | `nvidia/nemotron-3.5-lightning-30b-a3b` | 中（MoE）。是混合推理模型，通过 `chat_template_kwargs` 关闭思考 |
| `llama-3.2-90b-instruct` | `meta/llama-3.2-90b-vision-instruct` | 大，只用文本。闲置后首次调用要冷启动约 1 分钟 |
| `minimax-m3.1-flash` | `minimaxai/minimax-m3.1-flash` | 推理模型，思考关不掉，所以把 `max_tokens` 放宽到 2048 |

- 配置由 `python -m repro.build_config` 生成，写在 `config/llm_repro_phase2.json`。价格只是名义上的成本信号（NIM 免费），按模型规模参照论文表 10 设定。
- 测过但不用的模型：`openai/gpt-oss-20b`（3 次里 2 次连接超时）；`gemma-4-31b-it`、`glm-5.3-flash`、`deepseek-v4.1-flash`（连接超时）。

## 对官方代码的修改

| 修改 | 位置 | 原因 |
|---|---|---|
| API 地址可通过 `LLM_BASE_URL` 配置 | `shared/utils.py` | 方便换平台 |
| 增加 `MOCK_LLM=1` 模拟模式 | `shared/utils.py` | 离线测试流程，不花 API 额度 |
| API 失败时额外重试 3 次，仍失败则返回空答案（按答错计分）并计数 | `shared/utils.py` | 避免一次网络错误中断整个训练 |
| 去掉 `codegemma-7b → gemma-3-27b-it` 的替换 | `shared/utils.py` | 直接调用配置里指定的模型 |
| 支持在配置中给单个模型加 `max_tokens`、`extra_body` | `shared/utils.py`、`dynamic_router/route_env.py` | 关闭 lightning 的思考，放宽 minimax 的输出长度 |
| **MMLU/GPQA 改回用 `em_mc` 评分** | `shared/response_eval.py` | 见下方"评测问题"第 1 条 |
| 可选的成本惩罚 `--cost_alpha`（默认 0，即公开版本的行为） | `dynamic_router/route_env.py` | 论文公式 (4) 有成本项，用于 cost-performance 实验 |
| 缓存解析后的查询嵌入 | `dynamic_router/route_env.py` | 原来每次 reset 都要重新解析约 5000 条字符串，耗时 11.5 秒；改后训练提速约 2.7 倍，不影响训练逻辑 |
| 测试时按任务/领域统计 accuracy、LLM 调用次数和角色使用，并把每题的回答和标准答案保存为 JSON | `dynamic_router/train.py` | 原来只输出平均 reward |
| 增加 `--no_history`、`--inductive_test` | `dynamic_router/train.py` | 去掉历史记忆的消融；测试时不使用训练记忆 |
| `repro/` 目录下的脚本 | 新增 | 生成配置、处理数据、检查 API、评测 baseline 和模型、汇总结果 |

用 `git log` 和 `git diff 56010b1` 可以看到相对官方最后一次提交的全部改动。

## 数据和评测问题

1. **MMLU/GPQA 评分失效（已修复）。** 数据文件里这两个任务标注的评分方式是 `em_mc`，但公开代码把它们强制改成了 `cem`（标准答案是否出现在回答里）。对单个字母的选项，这种方式会失效："a" 会被当作英文冠词删掉，于是标准答案为 A 时任何回答（包括空回答）都判对；B/C/D 只要回答里有单词含有这个字母就判对。修复后，从回答中取最后一个 "(X)" 和标准答案比对。
2. **训练集和测试集重合（已处理）。** 公开的训练集包含 GPQA 和 HumanEval 的全部测试题，以及 MBPP 的 20 道测试题。`repro/prepare_data.py` 会从训练集中去掉这 121 条，生成 `data/router_data_train_clean.csv`。
3. **TriviaQA 有标注错误（未处理）。** 比如某道题的标准答案写成 "stillwell"（正确拼法 Stilwell），所有模型都会被判错。这是官方数据本身的问题，对各方法影响相同，未修改。
4. **MATH 要求 `\boxed{}` 格式。** 模型算对了但没写成这个格式，会被判错。这是论文评测本身的设定，未修改。

## 已知问题

- **连接失败。** `nemotron-3.5-lightning-30b-a3b` 在 8 个线程并发时频繁出现 "Connection error"。建议评测用 `--batch_size 2`，训练用 `--test_batch_size 2~4`。日志里会打印每次失败；结果是否可信，看失败次数（训练日志中的 `API calls ... (failed N)`）。
- **冷启动。** `llama-3.2-90b` 闲置后第一次调用约需 60 秒，属于正常现象。
- **额度和限流。** 平台没有在响应里返回限流信息，免费额度以 build.nvidia.com 账号页面为准。
- **API key 读取。** Ubuntu 默认的 `~/.bashrc` 开头有一段判断，非交互式 shell（包括 Claude Code 执行的命令）读到那里就返回，写在文件末尾的 `export NVIDIA_API_KEYS=...` 不会生效。可以在命令前加 `eval "$(grep '^export NVIDIA_API_KEYS=' ~/.bashrc)"`。
