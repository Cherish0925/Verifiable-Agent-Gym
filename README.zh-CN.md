# Verifiable Agent Gym

**面向学习型 Agent 的任务分布、可执行环境与实际运行反馈。**

[English](README.md) · [训练接口](docs/training.md) · [运行协议](docs/protocol.md) · [任务包](docs/task-packages.md) · [Agent 接入](docs/install.md)

Agent 接收任务，通过环境工具行动，提交结果后获得独立评价。本项目将这些经验组织为可供不同策略复用的数据：任务套件定义尝试什么，批量 slot 执行交互，导出保留尝试与结果，供外部学习器消费。

核心不绑定模型，服务于评测、合成经验采集、后训练和科学实验。[Data Engineering](https://github.com/Cherish0925/Data-Engineering)负责带版本的任务与轨迹产品，Gym 提供可执行环境，训练器负责策略推理与优化。[头部 AI 实验室的公开工作负载](https://github.com/Cherish0925/Data-Engineering/blob/main/docs/frontier-workloads.md)说明了这些接口的实际需求。

## 运行评测与分组 rollout

要求 Python 3.11 及以上，核心使用标准库，采用 MIT 许可证。

```sh
git clone https://github.com/Cherish0925/Verifiable-Agent-Gym.git
cd Verifiable-Agent-Gym
python -m pip install -e .
agent-gym list
agent-gym run --tasks all --count 10 --difficulty 3 --jobs 4 --output runs/baseline --require-pass
agent-gym export runs/baseline --output runs/baseline.jsonl
python examples/batched_rollouts.py --output runs/grouped
```

评测命令与批量示例使用脚本参考策略。批量示例通过四个环境 slot，为四个 RC 辨识任务分别采集四次尝试，再写出 `runs/grouped/trajectories.jsonl`。把示例中的动作生成步骤替换为模型批量推理，即可驱动相同环境；评测 runner 的 `empty` 策略提供未成功的对照。

`python -m agent_gym` 等价于 `agent-gym`；按系统使用 `python3` 或 `py -3.11`。

## 任务套件与学习器接口

| 对象 | 责任 |
| --- | --- |
| `TaskSuite` | 有限任务集合、每任务尝试次数及由内容计算的套件版本。 |
| `SuiteSampler` | 确定性任务顺序、epoch 与采样种子、按完整任务组分配 rank、可恢复的派发游标。 |
| `PolicyVersion` | 为每次尝试记录策略名称、版本和生成参数。 |
| `RolloutBatch` | 并发环境 slot，支持选择性 `reset`、`step`、`close_slots`，已完成 slot 可独立补充任务。 |
| Rollout 导出 | 任务、分组、尝试和策略身份，有序交互，调用者提供的策略统计量，终止状态与验证结果。 |

```python
from agent_gym import TaskSpec
from agent_gym.suite import TaskSuite

suite = TaskSuite(
    [TaskSpec("rc-identification", seed=i) for i in range(100)],
    attempts_per_task=4,
    name="rc-policy-comparison",
)
sampler = suite.sampler(seed=17, rank=0, world_size=4)
tickets = sampler.take(8)
checkpoint = sampler.state_dict()
```

同一采样任务组的全部尝试分配到同一个 rank。每个 slot 返回观察以及 `terminated`、`truncated`、`scored`、`reward`。策略可随动作传入 token 统计量，由训练器据此构造损失。sampler 检查点恢复后续派发位置；完整循环以及采样状态与运行环境状态的区别见[训练接入](docs/training.md)。

```sh
agent-gym export-rollouts runs/grouped --output runs/grouped-export.jsonl
```

## 由数据定义可执行任务

Task pack 将数据集变成可运行任务。SQL profile 接收数据库快照、任务描述和参考查询或结果；Agent 查看 schema、执行查询，再提交 SQL 或结果行。验证器按明确的行顺序与数值容差比较结果，接受得到相同结果的其他查询写法。

```sh
agent-gym pack-info examples/sql-workload/tasks.jsonl --family warehouse-sql
agent-gym --task-pack warehouse-sql=examples/sql-workload/tasks.jsonl list
python examples/sql-workload/agent.py --output runs/sql-example
```

示例包含四个人工编写的关系数据库问题，明确指定 train/validation/test。`load_taskpack` 可读取 JSONL 或 Data Engineering 发布目录，保留来源身份，枚举各分区中的有限任务。安装 Data Engineering 后，同一批记录可构建成数据版本再读取：

```sh
ai-data build examples/sql-workload/data.recipe.json --output runs/sql-tasks
agent-gym pack-info runs/sql-tasks --family warehouse-sql
```

使用 `pack.specs("train")` 构造 `TaskSuite`，将 `pack.make` 传给 batch 作为任务工厂。[任务包说明](docs/task-packages.md)介绍自定义 provider；[校准任务包](examples/calibration-package/)演示 `agent_gym.tasks` 扩展入口。

Data Engineering 的[完整经验流程示例](https://github.com/Cherish0925/Data-Engineering/blob/main/examples/experience_cycle.py)贯通任务版本、八次分组 SQL 尝试、轨迹版本、训练分片和可恢复的 rank 读取。

## 可生成的科学环境

| 任务族 | Agent 的工作 | 评价方式 |
| --- | --- | --- |
| `evidence-aggregation` | 选择有效文献版本、统一单位、综合测量。 | 核对证据集合、逆方差估计与标准误。 |
| `rc-identification` | 选择测量频率，辨识带噪声仪器的参数。 | 独立计算参数误差与留出频率上的响应。 |
| `thin-film-design` | 选择材料与厚度，设计减反膜。 | 用特征矩阵检查波段平均指标。 |

这些任务族可重复生成实例，包含三档难度和独立的切分种子空间；SQL pack 则枚举数据集记录。研究迁移能力时，可进一步按任务机制、领域或组合方式留出任务。

## 接入 Agent

```sh
python -m pip install -e ".[mcp]"
agent-gym connect --output agent-gym.mcp.json
```

兼容的 MCP host 使用 `gym_tasks`、`gym_open`、`gym_action`、`gym_close`。仓库包含 Codex、Claude Code 插件清单及共用 [Skill](skills/verifiable-agent-gym/SKILL.md)，详见[安装说明](docs/install.md)。

外部进程也可以从 stdin 接收 JSON 观察，向 stdout 返回 JSON 动作：

```sh
agent-gym run --tasks rc-identification --count 20 --agent command --command-json '["/absolute/path/to/python", "/absolute/path/to/my_agent.py"]' --jobs 4 --output runs/model
```

[运行协议](docs/protocol.md)给出交互格式。评测保留每次尝试；`--resume` 跳过已完成项，并在新环境重开中断任务；`--retry-errors` 重试记录中的基础设施故障。正式验证的答案产生奖励，截断、中断和环境故障保留各自状态与未评分奖励。

## 当前执行范围

当前在本机执行：batch 以线程推进不同环境，命令 runner 管理外部 Agent 进程，耗时求解器由任务 provider 管理作业。运行中环境的状态恢复、远程 worker 和训练框架专用适配仍待接入。分词、模型概率与优化属于外部学习器。内置数据与策略用于运行这些接口，实际学习收益和大规模工作负载结果需要消费者实验。

```sh
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/package_plugin.py
```

参与开发前阅读 [AGENTS.md](AGENTS.md)。新增工作应交付可运行的环境、可互操作的学习器适配，或能够改变任务构造和采样方法的研究结果。
