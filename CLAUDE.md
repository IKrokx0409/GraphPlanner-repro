# CLAUDE.md

这是 GraphPlanner（ICLR 2026，arXiv 2604.23626）的**课程作业复现仓库**，fork 自官方仓库 `ulab-uiuc/GraphPlanner`，`upstream` remote 指向官方仓库。目标是复现论文的结论和趋势，规模有意缩小：只做 Phase 2（动态路由），router 在 CPU 上训练，LLM 全部通过 NVIDIA NIM 免费 API 调用。

**开始工作前先读 `repro/NOTES.md`**，里面有当前进度、下一步、对官方代码的修改记录，以及已知的数据、评测和 API 问题。

## 代码结构

- `router_planner/dynamic_router/`：主实验（Phase 2）。`train.py` 是 PPO 训练和测试，`route_env.py` 是工作流环境，`network.py` 是 GARNet 网络。
- `router_planner/static_router/`：Phase 1（固定工作流）。本次没有用，也没改。
- `router_planner/shared/`：LLM 调用（`utils.py`）、各任务的 prompt 和评分（`response_eval.py`）。
- `repro/`：复现脚本，`build_config`、`prepare_data`、`check_api`、`evaluate`、`summarize`。
- `config/llm_repro_phase2.json`：当前使用的 4 个 LLM，由 `repro/build_config.py` 生成，要改模型就改这个脚本。
- `results/`：评测结果 JSON（会提交）。`logs/`、`checkpoints*/`、`wandb/` 不提交。

## 环境和常用命令

- conda 环境 `graphplanner`（Python 3.10，CPU 版 PyTorch 2.4.1）。依赖版本固定在 `repro/requirements.txt`，安装步骤见 README。
- 运行前需要设置：`NVIDIA_API_KEYS`（多个 key 用逗号分隔）、`HF_HUB_OFFLINE=1`（模型已缓存时设置，避免 transformers 联网时卡住）、`WANDB_MODE=offline`。
- 所有命令都在仓库根目录下用 `python -m ...` 运行：
  - `python -m repro.check_api`：每个模型发一条请求，开跑前先确认
  - `python -m repro.evaluate --policy {fixed,random,graphplanner} ...`
  - `python -m router_planner.dynamic_router.train ...`
  - `python -m repro.summarize results/*.json`
- 修改代码后，先用 `MOCK_LLM=1` 跑几个 episode 冒烟测试，不消耗 API 额度。

## 注意事项

- `filelock` 必须用 3.16.1。新版本会让 HumanEval/MBPP 评测时 fork 子进程报错，导致程序崩溃。
- 不要提交 API key。key 只放在环境变量里。
- 选择题（MMLU/GPQA 等）用 `em_mc` 评分：取回答中最后一个 "(X)"。不要改回 `cem`，原因见 NOTES。
- 推理模型的思考过程会计入 `max_tokens`。新增模型时，要么关闭思考，要么在 `build_config.py` 里给它单独设置 `max_tokens`。
- API 并发过高会出现 Connection error。评测的 `--batch_size` 和训练的 `--test_batch_size` 保持在 2~4。
- 跑完评测后，要抽查结果 JSON 里的 `response` / `gt` / `utility`，不能只看准确率。
