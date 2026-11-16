# Verifiable Agent Gym

**面向工具型 Agent 的可执行研究任务、环境与独立验证。**

[English](README.md) · [安装与接入](docs/install.md) · [运行协议](docs/protocol.md) · [任务包开发](docs/task-packages.md)

这里的核心不是题目文件夹，也不是让模型评价自己，而是一套可以实际运行的研究单元：Agent 接收任务，通过工具取得资料、测量或仿真，提交结果，再由独立验证器给出判断。Python、CLI、JSONL 和 MCP 使用同一个执行核心，不绑定某家模型或 Harness。

仓库公开，采用 MIT 许可证。可独立运行，也可作为 Skill、MCP 服务或插件接入现有 Agent。它负责环境和验证，不替代研究 OS，也不内置一套 RL 优化器。

## 直接运行

要求 Python 3.11 以上。基础运行不需要额外 Python 依赖、模型密钥、Bash 或 WSL。

```sh
git clone https://github.com/Cherish0925/Verifiable-Agent-Gym.git
cd Verifiable-Agent-Gym
python -m pip install -e .
agent-gym run --tasks all --count 10 --difficulty 3 --jobs 4 --output runs/baseline --require-pass
agent-gym export runs/baseline --output runs/baseline.jsonl
```

Windows 可以用 `py -3.11` 代替 `python`；macOS/Linux 可以用 `python3`。默认运行的是公开的脚本参考策略，不是 LLM 成绩。参考策略只使用任务公开的观察和工具。

## 接入已有 Agent

```sh
python -m pip install -e ".[mcp]"
agent-gym connect --output agent-gym.mcp.json
```

生成的配置指向当前已安装环境中的 Python。MCP 提供四个工具：`gym_tasks` 发现任务，`gym_open` 创建独立环境，`gym_action` 调用环境工具或提交结果，`gym_close` 结束未完成的任务。Codex 与 Claude Code 的插件清单、共用 Skill 也在仓库内；具体要求见[安装说明](docs/install.md)。Skill 本身是使用方法，实际计算仍由环境执行。

## 已实现的研究单元

| 任务族 | 实际工作 | 验证方式 |
| --- | --- | --- |
| `evidence-aggregation` | 读取合成文献，处理新旧版本、排除条件与混合单位，综合测量结果。 | 核对引用集合、逆方差加权均值与标准误。 |
| `rc-identification` | 主动选择频率，测量带噪声的虚拟仪器，辨识参数。 | 用独立表达式核对参数及未参与测量的频率响应。 |
| `thin-film-design` | 选择材料和厚度，设计单层减反膜。 | 独立特征矩阵计算；达到目标即可，不要求复制某个标准答案。 |

三类任务都能按种子重复生成，包含三档难度和 train/validation/test 划分。它们是可运行的基础环境，不代表已经覆盖所有研究领域。

## 长期运行与扩展

每次尝试独立保存任务、观察、工具动作、数值验证和结果。`--jobs` 控制本机并发数；外部 Agent 各自使用进程和工作目录，不依赖 Shell 拼接命令。`--resume` 跳过已完成结果，未完成的尝试在新环境中重开；`--retry-errors` 可显式重试基础设施故障，旧记录不会被覆盖。

答错、运行故障和时间耗尽不是一回事。只有正式验证后的提交产生奖励；故障、取消和截断保留明确状态与空奖励，不伪装成模型失败，也不会从整体运行统计里消失。

数据库、知识库、仿真软件与实验工具通过独立任务包加入。任务包负责具体环境、依赖和验证器，核心负责执行协议、并发与记录。[校准任务示例](examples/calibration-package/) 可以独立安装，展示真实的扩展入口，不需要修改核心。IOP、光学求解器、远端计算与训练框架的专用接入尚待开发，当前不冒充已经接通。

这是单机执行与评测基础，不是分布式训练平台，也不是不可信程序的操作系统沙箱。内置任务较轻；耗时的外部求解器需要其适配层负责超时和作业回收。公开种子生成器不等于保密测试集，不应把“换了种子”说成已经证明跨领域泛化。

维护原则很直接：增加真正可用的能力，保留来源与单位，解决实际故障，不重复封装已有软件。新任务应当包含可行实例、错误答案对照和独立验证。基础包保持轻量，领域能力独立演进。
