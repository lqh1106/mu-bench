# μ-Bench

**面向持续多人对话的共享语音助手评测。**

[English](README.md) · [安装部署](docs/INSTALL.md) · [测评协议](docs/PROTOCOL.md) · [评分机制](docs/EVALUATION.md) · [模型适配](docs/MODELS.md) · [Agent 安装指南](AGENTS.md)

作者与维护者：[Lin Qihao（@lqh1106）](https://github.com/lqh1106)。

μ-Bench 检验语音助手能否在同一段对话中，为多位用户推进相互关联的事务。用户可能插话、相互交谈、修订先前要求，或回到尚未完成的话题。助手需要持续辨明正在处理谁的请求、哪些约束仍然有效，以及谁有权改变已有安排。

仓库提供 **100 个基础场景、800 个具体案例**，包含完整案例设计、在线用户模拟器、参考声纹、原生音频适配器、虚拟业务工具、基于证据的评分器与结果保存流程。补充模型权重或 API 凭据，并部署仓库提供的 ASR/TTS 服务后，即可运行音频测评。

## 评测什么能力？

例如，Avery 要求把预订改为八人，Blake 随后补充实际只有六人，Avery 再回来确认修订。助手需要把这些发言连接到同一事务，正确识别每次业务调用的请求来源，并完成最终更新。环境会保存成功操作、被拒调用和业务终态。

无工具的机场案例中，Avery 询问登机口和行动协助，另一位用户同时播报不同航班的登机口，第三位旅客又提出自己的行程。助手需要在持续交互中保持航班、说话人和请求的对应关系。

场景涵盖出行与酒店、共享家庭与日程安排、会议、教育、医疗、客户支持、公共服务等多人日常事务。

## 案例与受控变量

| 子集 | 基础场景 | 具体案例 | 完成依据 |
|---|---:|---:|---|
| 无工具 | 49 | 392 | 实际说出的回答、设计目标、约束与隐私要求 |
| 有工具 | 51 | 408 | 对话、真实虚拟工具返回、必要业务效果及终态 |
| **合计** | **100** | **800** | |

每个场景包含 **2 × 2 × 2 = 8 个变体**：

| 受控变量 | 案例 ID 中的取值 |
|---|---|
| 群体规模 | `g-small` / `g-large` |
| 声学条件 | `a-clean` / `a-realistic` |
| 用户理解水平 | `c-strong` / `c-weak` |

例如 `airline_04_gate_announcement_collision__g-small__a-clean__c-strong` 对应一个具体变体。完整设计位于 [cases/](cases/)。

## 从克隆到首案运行

### 1. 安装并核验

运行器使用 Linux 与 Python **3.10–3.12**，需预先安装 Git、`ffmpeg`、`libsndfile` 以及所选后端要求的编译工具。运行器、模型、ASR 和 TTS 分别使用独立环境。

**复现命令——新建本地副本：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
git clone https://github.com/lqh1106/mu-bench.git "$MUBENCH_REPO"
cd "$MUBENCH_REPO"
bash scripts/bootstrap.sh
.venv/bin/python -m mubench validate --assets
```

仓库包含 **3,200 个参考声纹的目录、参考录音和固定的参与者声纹绑定**。模型权重和 API 凭据单独配置。

### 2. 部署服务并填写配置

按 [安装文档](docs/INSTALL.md) 启动以下组件：

| 组件 | 实现与用途 |
|---|---|
| TTS | CosyVoice HTTP 服务，按指定声纹合成用户发言 |
| ASR | Faster-Whisper large-v3，转写实际播放的助手声音 |
| ASR／文本网关 | 将音频请求路由到本地 ASR，将模拟器和评分请求路由到文本 API |
| 待测模型 | 原生模型服务或云端适配器，每案维护独立会话 |

用户模拟器和评分器默认使用 `gpt-6-luna`；具体模型与服务地址由本地配置指定。各模型权重、运行环境及辅助资源见 [模型文档](docs/MODELS.md)。

**复现命令——准备本地配置文件：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
cp .env.example .env
cp configs/run.native.example.json local.native.json
cp configs/run.closed.example.json local.closed.json
```

按安装文档填写本地环境变量、模型端点和语音服务路径。配置支持通过环境变量引用密钥；`.env`、`local.*.json`、本地 token 和生成结果由 Git 忽略。MOSS 分别提供[音频配置](configs/run.moss.example.json)和[有工具配置](configs/run.moss.tools.example.json)。

### 3. 先跑一案

**复现命令——服务就绪后，使用原生模型测试无工具案例：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

**复现命令——使用已配置的云端模型测试有工具案例：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.closed.json --subset tool_augmented --case hospitality_01_party_size_fix__g-small__a-clean__c-strong --workers 1
```

省略 `--case` 即运行整个所选子集；填写基础场景名可选择该场景的八个变体。提高 `--workers` 时，需同时调整模型服务允许的会话数。多个运行器应共享输出根目录，以通过领取锁避免重复跑案。MOSS 每路需要独立 key 端点，同一会话中保持 key 不变。

## 支持的模型

| 模型适配 | 无工具音频 | 有工具音频 | 身份／语义消融投递 |
|---|:---:|:---:|:---:|
| MiniCPM-o | ✓ | — | ✓ |
| Moshi、Freeze-Omni、FLM-Audio、Voila、Raon-SpeechChat、PersonaPlex、Lychee-FD | ✓ | — | — |
| OpenAI Realtime、Grok voice、Gemini Live | ✓ | ✓ | — |
| MOSS Live | ✓ | ✓ | ✓ |

八个开源模型均附模型加载器和依赖记录；云端适配器使用各自的原生流式接口。业务工具使用公开参与者编号和逐次调用的 `caller_id`；身份报告的返回值只确认接收，不告知识别正误。

模型检查点、部署方式与能力范围见 [MODELS.md](docs/MODELS.md)。注册、初始资料投递、打断和模拟器计算期间的暂停见 [PROTOCOL.md](docs/PROTOCOL.md)。

## 三组消融

在相同任务目标、约束和业务规则上，逐层提供额外辅助：

| 设置 | CLI 参数 | 改动 |
|---|---|---|
| 串行 | `serial` | 用户发言依次送达，消除用户重叠和实时插话要求 |
| 串行＋身份 | `serial_identity` | 进一步提供说话人、受话人、请求归属及线程信息 |
| 串行＋身份＋语义 | `serial_identity_semantics` | 进一步提供当前发言及已听上下文的结构化解释 |

**三组共 1,176 个案例，设计与模型无关**，由三个设置分别作用于 392 个无工具案例。当前身份／语义辅助投递适配 MiniCPM 和 MOSS。案例在模型间共用，运行实现由独立指纹记录；结构化输入生成及最后候选采用情况保留审计。

**复现命令——原生服务就绪后测试身份组：**

```bash
export MUBENCH_REPO="$HOME/mu-bench"
cd "$MUBENCH_REPO"
set -a
source .env
set +a
.venv/bin/python -m mubench run --config local.native.json --subset tool_free --ablation serial_identity --case airline_04_gate_announcement_collision__g-small__a-clean__c-strong --workers 1
```

## 评分机制

证据包组织实际送达的用户发言、实际播放的助手音频及其 ASR、工具返回、参与者映射和设计任务契约。评分依次经过：**初评 → 独立全复核 → 有冲突时仲裁 → 按需进行来源专项核查**，并在发布前验证证据和确定性条件。

- **Task Success** 检查设计目标、必要业务效果、终态和隐私；授权违规单独报告，保留后续更换聚合口径所需的记录。
- **模拟器追加要求**不升级为案例必须完成的目标。
- **无工具案例**按要求的口头行为判断，不要求不存在的数据库写入。
- **有工具案例**区分口头报告和真实执行；终态与执行历史分别用于业务效果和过程、授权核验。

| RICE 维度 | 测量内容 |
|---|---|
| **R：Routing** | 身份与受话人归属、请求来源、用户返回时的指代锚点 |
| **C：Continuity** | 保持并续接正确的任务线程与约束 |
| **I：Isolation** | 用户之间的隐私、权限与信息隔离 |
| **E：Equity** | 适用的公平性、群体协作和低显著性输入处理 |

各轴分数在 `[0, 1]`，按可用子项聚合，不适用项留空。逐项依据及分母记录保存在输出中。完整定义见 [评分文档](docs/EVALUATION.md)与[有工具评分契约](docs/TOOL_EVALUATION.md)。

## 输出、自动清理与补评

默认输出结构如下：

```text
outputs/<model>/<subset>/<condition>/
├── formal/<case>/                # 完整采用结果
│   ├── score.json
│   ├── rice_score.json
│   ├── audio/both.wav            # 最终双声道录音
│   └── ...                      # Case、ASR、轨迹及评分证据
├── runs/<case>/attempt_001/      # 尝试日志、回执和证据
└── claims/                      # 案例领取锁
```

每案新建独立会话，已发布案例再次运行时跳过。交互和评分有效的任务失败仍作为真实结果保留；技术错误单独记录，保存证据后推进后续案例。

示例配置默认开启自动清理：在分数、RICE 与最终录音通过校验后，删除冗余单声道和音频分片；保留最终立体声录音、ASR、轨迹、契约和评分审计。同一文件系统内通过硬链接发布，避免 `formal` 与 `runs` 重复占用音频空间。不自动删除未完整的尝试证据。

完整交互可以离线重新评分，无需重新运行待测模型；新评价写入单独的续评目录。操作见[离线补评](docs/INSTALL.md#offline-judging)。

## 文本诊断

文本适配器按注册姓名标注发言，复用在线模拟器和虚拟工具。按逻辑时间触发打断时，保留已送达前缀、丢弃未送达后缀，再加入打断发言；完整工具调用保留实际效果，未完成的调用不执行。

默认待测文本模型为 `qwen3.8-flash`。将[文本配置模板](configs/run.text.example.json)复制为 `local.text.json`，填写支持所选模型的端点即可。该轨道不需要 ASR、TTS 或 GPU，分数标记为文本诊断结果。时钟估计与轨迹格式见 [TEXT_TRACK.md](docs/TEXT_TRACK.md)。

## 仓库导航

| 目录 | 内容 |
|---|---|
| [cases/](cases/) | 完整无工具、有工具及模型无关的消融案例 |
| [runtime/](runtime/) | 模拟器、时钟、事件调度与运行入口 |
| [adapters/](adapters/) | 原生模型加载器及云端适配器 |
| [judges/](judges/) | 证据组织、语义复核、确定性评分与 RICE |
| [services/](services/) | ASR、TTS、网关和模型服务 |
| [assets/voices/](assets/voices/) | 声纹目录、参考音频及固定绑定 |
| [configs/](configs/) | 运行与模型路径模板 |
| [environments/](environments/) | 分模型依赖记录 |
| [third_party/](third_party/) | 上游源码及原许可证 |
| [scripts/](scripts/)、[tests/](tests/) | 安装辅助、校验和回归测试 |

自动配置 agent 请从 [AGENTS.md](AGENTS.md) 开始；组件关系见[架构文档](docs/ARCHITECTURE.md)。

## 引用与许可证

软件引用信息见 [CITATION.cff](CITATION.cff)。

原创 benchmark 代码和案例内容采用 [Apache-2.0](LICENSE)。第三方代码、参考录音和外部模型权重保留各自许可证，详见 [NOTICE](NOTICE) 和[第三方组件说明](docs/THIRD_PARTY.md)。
