# μ-Bench

### 面向共享语音助手的多人任务完成评测

[English](README.md) · [快速开始](#快速开始) · [安装部署](docs/INSTALL.md) · [模型适配](docs/MODELS.md) · [评分机制](docs/EVALUATION.md) · [Agent 安装指南](AGENTS.md)

**语音助手能否在一场共享对话中，完成多位用户相互关联的任务？**

个人助手帮助一个人完成事务；共同出行、会议协作和家庭安排则要求助手记住各人的要求，检查新提议对已有安排的影响，并跟进未完成事项。**μ-Bench 通过可响应模型行为的多人对话与明确的任务结果，检验这份共同委托能否成立。**

[![孩子插话要求去公园，助手查询预计到达时间、取得家长同意，并保留机场到达时限；下方展示 Attribution、Threading 和 Coordination。](assets/figures/intro-teaser.png)](assets/figures/intro-teaser.pdf)

*共享交互示意；点击图片查看矢量 PDF。*

家长要求 **10:00 前到机场**，孩子提出先去公园。助手需要检查绕路影响、取得同意并保留到达时限，同时分清“谢谢爸爸”是对同伴说话，“嗯”是附和。μ-Bench 从三个维度观察其中的职责：

- **Attribution：归属。** 分清谁在说话、对谁说话，以及正在处理谁的请求。
- **Threading：线程保持。** 持续维护、更新并接续各人的未完事项。
- **Coordination：协调。** 在权限、隐私和共同约束下，满足相互关联的目标。

## Benchmark 概览

**20 个领域 · 100 个基础场景 · 800 个案例 · 全双工语音**

| 子集 | 场景 | 案例 | 完成依据 |
|---|---:|---:|---|
| **无工具** | 49 | 392 | 口头回答、澄清、拒绝与协调 |
| **有工具** | 51 | 408 | 对话、真实虚拟工具执行及必要业务效果 |

场景涵盖出行、家庭、会议、医疗、教育、公共服务等领域。每场景有 **2 × 2 × 2 = 8 个配对变体**，分别改变群体规模（2–3／4–7 人）、声学风格（清晰／自然口语）和用户理解及追问需求（强／弱）。[案例集](cases/)支持固定场景及其余两项变量、只改变一项因素的对比。

## 如何运行一场测评？

**案例 Playbook → 响应式用户 ⇄ 待测助手与工具 → 交互证据 → 任务与 ATC 分数**

每案定义参与者、各自可知信息、目标、约束及事件图。调度器根据实际对话释放事件，编剧生成相应用户的发言，配音器按固定声纹合成语音；助手的发言和工具效果驱动后续插话、澄清与请求修订。评分器依据最终对话和环境，核验设计目标是否完成。

仓库包含案例、参考声纹、模型适配器、语音服务实现和评分器。[架构文档](docs/ARCHITECTURE.md)介绍组件关系，[测评协议](docs/PROTOCOL.md)说明播放、打断、逻辑时间暂停及信息边界。

## 快速开始


### 1. 克隆并验证

运行器支持 **Linux、Python 3.10–3.12**。安装 Git、`ffmpeg`、`libsndfile` 和所选后端要求的编译工具。仓库已包含声纹目录、**3,200 条参考录音**及固定参与者绑定；模型权重和 API 凭据单独配置。

**复现命令——新建本地副本：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
git clone https://github.com/lqh1106/mu-bench.git "$MUBENCH_REPO"
cd "$MUBENCH_REPO"
bash scripts/bootstrap.sh
.venv/bin/python -m mubench validate --assets
```

### 2. 启动服务并配置模型

[安装文档](docs/INSTALL.md)提供完整启动命令。运行器、语音服务及每个原生模型使用独立环境。

| 组件 | 实现 | 配置内容 |
|---|---|---|
| 用户 TTS | CosyVoice HTTP 服务 | 声纹目录、TTS 权重、`speech.tts_endpoint` |
| 助手 ASR | Faster-Whisper large-v3 | ASR 权重和网关使用的 ASR 地址 |
| ASR／文本网关 | 仓库内 HTTP 路由服务 | 文本上游地址、密钥及本地网关密钥 |
| 用户模拟器与评分器 | OpenAI 兼容文本 API | `text.base_url`、`text.api_key`、`text.model`，默认 `gpt-6-luna` |
| 待测模型 | 原生 GPU 服务或云端接口 | `sut` 中的模型、端点和提供方凭据 |

**复现命令——准备本地配置：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
cp .env.example .env
cp configs/run.native.example.json local.native.json
cp configs/run.closed.example.json local.closed.json
```

按 [INSTALL.md](docs/INSTALL.md) 填写 `.env` 与所选配置。示例通过环境变量引用密钥；模型路径和隔离环境见 [MODELS.md](docs/MODELS.md)。MOSS 提供[音频模板](configs/run.moss.example.json)及[有工具模板](configs/run.moss.tools.example.json)。本地配置、token、权重和运行结果不进入版本控制。

### 3. 先跑一案，再增加并发

**复现命令——服务就绪后运行原生模型无工具测试：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

**复现命令——使用已配置的云端模型运行有工具测试：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.closed.json --subset tool_augmented --case hospitality_01_party_size_fix__g-small__a-clean__c-strong --workers 1
```

运行结束后，在 `formal/<case>/` 中查看 `score.json` 的任务判定、`atc_score.json` 的诊断指标，并播放 `audio/both.wav` 回听对话。该目录位于 `output_root/<model>/<subset>/<condition>/` 下。[输出结构、全量运行与并发设置 →](docs/USAGE.md)

## 支持的模型与实验设置

| 模型 | 无工具音频 | 有工具音频 | 身份／语义消融 |
|---|:---:|:---:|:---:|
| MiniCPM-o | ✓ | — | ✓ |
| Moshi、Freeze-Omni、FLM-Audio、Voila、Raon-SpeechChat、PersonaPlex、Lychee-FD | ✓ | — | — |
| OpenAI Realtime、Grok voice、Gemini Live | ✓ | ✓ | — |
| MOSS Live | ✓ | ✓ | ✓ |

[模型部署文档](docs/MODELS.md)提供检查点、隔离环境和提供方配置。实际可用性与并发容量取决于部署端点。

**能力消融。** 三组设置保留原任务目标与资料，逐层消除交互要求：**串行 → 串行＋身份 → 串行＋身份＋结构化语义**。共 1,176 个变体，设计与模型无关；MiniCPM 和 MOSS 已实现相应辅助输入投递。[消融设置与送达核验 →](docs/USAGE.md#capability-ablations)

**文本诊断。** 可选的流式文本轨道按注册姓名标注发言，使用逻辑时间前缀截断和同一虚拟工具环境，无需语音服务或 GPU。默认待测文本模型为 `qwen3.8-flash`。[配置与结果解读 →](docs/TEXT_TRACK.md)

## 如何评价？

**Task Success** 判断案例适用目标、必要业务效果、终态与隐私约束是否满足。**授权及调用人来源违规单独报告**，并保留对应的原始记录。

**ATC** 解释任务结果背后的交互能力：

| 维度 | 含义 | 分项映射 |
|---|---|---|
| **A：Attribution** | 说话人、受话人及请求归属 | `A = Routing` |
| **T：Threading** | 任务线程保持与续接 | `T = Continuity` |
| **C：Coordination** | 边界遵守与适用的多人服务公平性 | `C = (Isolation + Equity) / 2`；无适用 Equity 时，`C = Isolation` |

分数范围为 `[0, 1]`；不适用的 Equity 不计入分母，实际得零则参与平均。ATC 按案从保留的 RICE 分项计算，无需额外调用 Judge。

评价结合确定性工具／状态检查、语义初评、独立全复核、必要时的冲突仲裁与定向来源复核。Judge 只按案例设计目标评分，模拟器追加的请求不升级为必须完成的目标。完整轨迹可以离线重评，无需重跑待测模型。

[评分与证据流](docs/EVALUATION.md) · [有工具评价契约](docs/TOOL_EVALUATION.md) · [输出与自动清理](docs/USAGE.md#outputs-and-cleanup)

## 文档导航

| 需要做什么 | 入口 |
|---|---|
| 部署 TTS、ASR 与运行器 | [安装部署](docs/INSTALL.md) |
| 配置或部署待测模型 | [模型文档](docs/MODELS.md) |
| 全量运行、消融或离线补评 | [使用指南](docs/USAGE.md) |
| 理解共享频道交互 | [测评协议](docs/PROTOCOL.md) |
| 检查任务判定与诊断分数 | [评分机制](docs/EVALUATION.md) |
| 运行流式文本实验 | [文本诊断](docs/TEXT_TRACK.md) |
| 查找组件或由 Agent 配置 | [架构文档](docs/ARCHITECTURE.md) · [Agent 指南](AGENTS.md) |

## 引用与许可证

作者与维护者：[Lin Qihao（@lqh1106）](https://github.com/lqh1106)。软件引用信息见 [CITATION.cff](CITATION.cff)。

原创代码和案例采用 [Apache-2.0](LICENSE)。第三方代码、参考声纹和外部权重保留各自许可证，详见 [NOTICE](NOTICE) 与[第三方组件说明](docs/THIRD_PARTY.md)。
