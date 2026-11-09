# CLAUDE.md

这是 GraphPlanner（ICLR 2026，arXiv 2604.23626）的**课程作业复现仓库**，fork 自官方仓库 `ulab-uiuc/GraphPlanner`，`upstream` remote 指向官方仓库。目标是认真复现 Phase 2（动态路由）的结论和趋势（规模不过度缩小：1000 episode × 3 个种子，556 题完整测试集），之后接入 LycheeMAS。router 在 CPU 上训练，LLM 全部通过 DeepInfra API 调用。

**开始工作前先读 `repro/NOTES.md`**，里面有当前进度、下一步、对官方代码的修改记录、运行环境的坑，以及已知的数据和评测问题。实验的完整说明和报告素材在 `results/report/README.md`。

## 代码结构

- `router_planner/dynamic_router/`：主实验（Phase 2）。`train.py` 是 PPO 训练和测试，`route_env.py` 是工作流环境，`network.py` 是 GARNet 网络。
- `router_planner/static_router/`：Phase 1（固定工作流）。本次没有用，也没改。
- `router_planner/shared/`：LLM 调用（`utils.py`）、预算账本和回答缓存（`api_guard.py`）、各任务的 prompt 和评分（`response_eval.py`）、代码题隔离执行（`evaluation/isolated.py`）。
- `repro/`：复现脚本，清单见 NOTES。正式实验入口是 `repro/run_phase2.sh {train|labels|eval}`。
- `config/llm_repro_phase2.json`：当前使用的 4 个 LLM，由 `repro/build_config.py` 生成，要改模型就改这个脚本。
- `results/`：评测结果 JSON 和报告素材（会提交）。`logs/`、`checkpoints*/`、`wandb/`、`cache/` 不提交，用 `bash repro/backup_data.sh` 备份到 Windows 盘。

## 环境和常用命令

- conda 环境 `graphplanner`（Python 3.10，CPU 版 PyTorch 2.4.1）。依赖版本固定在 `repro/requirements.txt`，安装步骤见 README。
- 运行前需要设置：`DEEPINFRA_API_KEY`（非交互 shell 里用 `eval "$(grep '^export DEEPINFRA_API_KEY=' ~/.bashrc)"` 读取）、`HF_HUB_OFFLINE=1`（避免 transformers 联网时卡住）、`WANDB_MODE=offline`。
- 所有命令都在仓库根目录下用 `python -m ...` 运行：
  - `python -m repro.check_api`：每个模型发一条请求，开跑前先确认
  - `python -m repro.evaluate --policy {fixed,random,graphplanner} ...`
  - `python -m router_planner.dynamic_router.train ... [--resume]`
  - `python -m repro.summarize results/*.json`
- 修改代码后，先用 `MOCK_LLM=1` 跑几个 episode 冒烟测试，不消耗 API 额度。

## 注意事项

- **预算很紧：** DeepInfra 额度只有 $10 且不能充值，代码里的硬上限是 `LLM_BUDGET_USD=8`（所有进程共享 `cache/api_spend.sqlite`）。新实验开跑前先估算花费；同一种子同一代码重跑会命中 `cache/llm_responses.sqlite`，不花钱。
- **机器会休眠和关机：** 训练要能用 `--resume` 续跑（每 25 个 episode 存档）。不要在 bash 脚本运行时修改它；不要用 `pkill -f`，按 PID 杀进程。细节见 NOTES 的"运行环境和坑"。
- `filelock` 必须用 3.16.1。新版本会让 HumanEval/MBPP 评测报错。
- 不要提交或打印 API key。key 只放在环境变量里。
- 选择题（MMLU/GPQA 等）用 `em_mc` 评分：取回答中最后一个 "(X)"。不要改回 `cem`，原因见 NOTES。
- 新增模型时，检查它的代码题回答是否带 markdown 代码块、是否是推理模型（思考会计入 `max_tokens`），必要时在 `build_config.py` 里单独设置。
- 评测的 `--batch_size` 保持在 4~8。
- 跑完评测后，要抽查结果 JSON 里的 `response` / `gt` / `utility`，不能只看准确率。
- 提交信息里不要加 Co-Authored-By。
