# 作业 1:构建智能体执行框架(Agent Harness)

> 译注:原文见 [ASSIGNMENT.md](ASSIGNMENT.md)。本文件为中文参考翻译,代码块、命令与专有名词保留原文;与英文原文如有出入以原文为准。

智能体执行框架(agent harness)提供了让语言模型(一个只会生成"概率上的字符串"的东西)能够观察环境并在其中行动的接口。构建 agent harness 最流行的框架之一是 [ReAct](https://arxiv.org/abs/2210.03629),它在语言模型的提示中交错排列环境观察、推理/思维链以及智能体动作对应的文本。在本作业中,你将用 ReAct 框架构建一个 harness。

在 Part 1,你将实现基类 [`Agent`](./src/assignment/agent/base.py) 的 ReAct 循环,并把它实例化为一个 [`CodeAgent`](./src/assignment/agent/code_agent.py)——在终端中行动、解决软件问题。你实现的 ReAct 循环是抽象的,可应用于多个领域,让你看到 agent harness 的通用结构。你将实现让编码智能体在终端环境中观察和行动的机制,并应用可复用的智能体技能(skill)来完成任务。然后,你将让这个编码智能体修复一个国际象棋应用中的问题。

在 Part 2,你将研究编码智能体在解决更复杂的软件问题时如何使用上下文。你将实现一种简单的上下文压缩(context compaction)方法,管理智能体对语言模型上下文窗口的占用。

在 Part 3,你将把之前构建的基类 `Agent` 实例化为 `ChessAgent`,用它来玩你在 Part 1 中用 `CodeAgent` 修复的国际象棋应用。你将实现这个智能体的工具规范(tool specification),让语言模型可以玩运行在应用中的棋局、对抗基于规则的机器人(你可以旁观!)。你还将探索程序化工具调用(programmatic tool calling)——把程序的力量带入智能体的动作,让你的智能体能够通过 Python 代码执行实现更复杂的策略。

三个部分的关系如下:

```text
                SWE-Bench issue(软件问题)
                      |
                      v
buggy chess app -> CodeAgent -> fix.patch -> repaired chess server(修复后的棋局服务器)
                                              ^
                                              |
                                    ChessAgent tools
```

## 环境配置(Setup)

安装 [uv](https://docs.astral.sh/uv/),然后运行:
```bash
make setup
```
这会用 `uv` 搭建 Python 环境、下载并安装相关依赖。如果固定的 `chess_app` 源码(你的智能体要处理的任务目标)缺失或不在正确的 commit 上,此步骤会失败。

本作业会使用 [Modal](https://modal.com/),一个用于运行代码的云平台。该平台让你能在云机器上安全地运行智能体生成的代码。智能体所处的环境位于远程 Modal 沙箱中,执行智能体动作的代码也在这里运行。首先,注册 Modal 账户(如果没有的话),并按照发给你的说明获取算力额度。然后运行
```bash
uv run modal setup
```
登录并在你的环境中设置 Modal。

设置好 Modal 之后,运行
```bash
cp .env.example .env
```
创建包含环境密钥的文件——主要是 LLM API 信息。如果你已选课,会收到关于获取 LLM 服务商额度的说明。按照说明生成 API key,并确保正确配置服务地址(base URL)。

```dotenv
OPENAI_BASE_URL=<OpenAI 兼容的 base URL>
OPENAI_API_KEY=...
OPENAI_MODEL=deepseek/deepseek-v4-flash-0731
OPENAI_MAX_RETRIES=5
```
默认情况下你将使用 DeepSeek-V4-Flash 模型(如上所示)。在指定时要换用其他模型。如果额度允许,也欢迎探索同一服务商的其他模型,但我们希望作业是用这个模型完成的。

我们把任何会消耗 API 额度(Modal 或 LLM 服务商)的活动称为_计费(billable)_活动。整个作业过程中你会运行多次计费评测。建议在相关面板上监控用量,确保好好利用 API 额度。

在计费运行之前,可以用以下命令验证子模块、Modal 认证和模型端点,而不会启动沙箱或生成 token:
```bash
make doctor
```

_(可选)_ 如果你想测试所有 Modal 组件是否正常,运行
```bash
make test-modal
make test-chess-modal
```

一个通用提示:你可以用
```bash
modal container list
```
检查是否有计费的 Modal 沙箱在运行。如果环境没有正确关闭,沙箱可能残留并消耗额度。可以用 `modal container stop <容器 ID>` 停掉未正确终止的沙箱。

## 公开测试(Public tests)

`make test` 快速、离线、不计费。起步代码故意让测试失败,直到你完成相应的 TODO。把这些测试当作里程碑;如果后续增加了澄清性测试,pytest 的确切数量可能会变化。

通过公开测试不代表完全正确。私有测试还覆盖:失败时的清理、重复/格式错误的技能、并行的下棋调用、传输错误、产物一致性、补丁回放以及真实的 Modal 集成。

## 规则(Rules)

- 不要修改 `tests/`、`tasks/`、`chess_app/`。
- 不要改动提供的日志/清理逻辑,也不要在子类中重复实现共享的 ReAct 循环。
- 不要硬编码智能体任务的预期解法来通过测试。你的智能体应当为给定任务生成走法或补丁。
- **永远不要暴露、记录或提交 API key 等凭据。** 小心 `.env` 文件的内容。
- 在整个作业中,你会使用多个智能体及其多个版本。任何时候都只使用该智能体应有的一组工具。

## Part 1:构建编码智能体并修复国际象棋应用

共享的 `Agent` 必须实现常规的 ReAct 循环:构建请求、获取助手的动作、执行其工具调用、添加关联的观察结果,如此重复直到完成。

### 1. 构建提示(prompt)

给语言模型的提示(prompt)是一个消息序列,在交互的每一步都用于查询语言模型。消息是一个字典对象,最少包含 `role` 和 `content` 两个键,还可以有附加信息。`content` 可以是字符串,也可以是包含推理 token 的结构化对象,或(本作业未涵盖的情况)图像。消息的 `role` 可以取以下几个值。

`system` 角色用于提供常设指令——关于领域的说明、环境信息、要遵守的规则、跨任务可能有用的通用策略等。`user` 角色提供任务信息[^1]以及(如有必要)解决问题的附加指引。`assistant` 角色的消息表示 LLM 的生成结果,`tool` 角色的消息表示作为环境观察的工具调用结果。你的任务是组织这些消息的序列。按照惯例,第一条是且仅是一条 `system` 消息,并且任何 assistant 消息出现之前要有一条 `user` 消息。由于 LLM 一次只生成一个响应,`assistant` 消息后面应跟 `tool` 消息(展示工具调用的结果)或 `user` 消息。你可以在 [OpenAI API 参考文档](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)和 [vLLM 文档的这个示例](https://docs.vllm.ai/en/v0.7.2/getting_started/examples/openai_chat_completion_client_with_tools.html)中了解这些消息结构的更多信息。

你需要实现 `Agent.build_prompt` 方法。给定作为 agent 类属性的系统提示和任务提示,以及你可能添加的任何簿记信息,该方法应准备好给语言模型的输入。如你所见,我们提供了 `Agent.query_language_model` 方法,它直接用此方法的输出来查询语言模型 API。你的实现需要为这个查询提供正确的输入。`Agent.query_language_model` 方法会向语言模型 API 提供工具和推理参数,并返回一个包含所有相关信息的结构,它还维护步数(查询 API 的次数)。API 负责把提供的提示和工具渲染成单个 token 序列,你不需要处理这部分。它还提供了把请求记录下来用于评分的额外机制,你不应修改。

智能体的 LLM 客户端会在临时性的服务商故障时最多重试 `OPENAI_MAX_RETRIES` 次。

> **TODO(1.1.a)**
> 添加机制来维护智能体在执行动作和观察结果过程中的状态。构造构成语言模型提示的消息序列,包括常设指令、任务说明,以及之前轮次的交互(包括观察、推理和动作)。注意,此方法应当与领域无关,要以能适用于任何继承它的领域特定智能体的方式来构造提示。

接下来,为 `CodeAgent` 构造系统提示和任务提示。系统消息_必须_逐字包含以下代码块,并使用环境暴露的值:
```text
<system_information>
{
  "machine": <machine>,
  "release": <release>,
  "system": <system>,
  "version": <version>
}
</system_information>
```

> **TODO(1.1.b)**
> 为 `CodeAgent` 构造系统提示和 task_prompt,使其可被 `Agent.build_prompt` 方法使用。

### 2. 运行 ReAct 循环

实现 `Agent.run` 的主体。

> **TODO(1.2)**
> 运行 ReAct 循环。编排以下序列:提示语言模型生成推理和动作、提取模型产出的工具调用、执行工具调用以获得智能体下一步的观察。确保通过设置 `Agent.finished` 来识别智能体完成了任务。如果智能体超过 `step_limit`,抛出 `StepLimitError`。

### 3. 执行编码工具

实现 `CodeAgent.execute_tool_calls`。编码智能体支持两个工具:`execute` 和 `send_message`。工具定义见 [tools.py](./src/assignment/agent/tools.py)。你需要实现执行这些工具的机制,并准备它们的输出以展示给语言模型。这些输出也是 `{"role": str, "tool_call_id": str, "content": str}` 形式的消息。每条消息是执行某个工具的观察,属于一种叫 `tool` 的特殊角色。[^2] 你应使用 `Environment.execute` 方法确保代码在正确的 Modal 沙箱上执行。可选参数可以保持当前默认值,但要允许智能体在工具调用中覆盖它们。

> **TODO(1.3)**
> 让 `execute` 和 `send_message` 工具对智能体可用。解析每个调用,执行可识别的工具,并为每个调用返回一条消息(一个智能体响应中可能有多个工具调用!)。格式错误的 JSON 和未知工具必须变成可恢复的观察并转达给智能体,而不是抛出异常。

### 4. 加载技能(skills)

你可能以为可以开始在软件问题上运行智能体了,但还差一点!`CodeAgent` 需要的最后一部分知识是:如何提交解法。我们使用的具体协议是:智能体解决问题后,把解法保存到名为 `patch.txt` 的文件中。这个文件会从环境中提取出来用于评测。

为了教会智能体这个协议,我们使用技能(skill)。技能是用专门的、可复用的工作流扩展智能体能力的方式。[Agent Skills 协议](https://agentskills.io/home)定义了技能应遵循的标准,使其能被多个智能体框架支持。`tasks/code-skills/submit-task` 定义了一个用于按此协议提交解法的初级技能。虽然该协议还有更多高级特性可供探索,但本作业中你将使用一个最小化的技能。

你需要实现的核心特性是_渐进式披露(progressive disclosure)_。为了让智能体能够使用许多(可能很复杂的)技能,技能中的信息以细节逐级增加的方式披露给智能体,由智能体获取它需要的信息。本作业中你将实现一种简单的渐进式披露:在系统提示中给出所有可用技能的描述,同时给智能体提供 `invoke_skill` 工具——智能体可以用技能名调用它来查看完整的技能内容。这样,当智能体查看完整的 `submit-task` 技能后,就知道如何提交修复用于评分。我们会检查系统提示中提到了技能名和描述。当没有技能可用时,提示的任何部分都不应提到 `patch.txt` 或给智能体提交说明。

技能文件在本地 [tasks/code-skills/](./tasks/code-skills/) 提供,当智能体运行且使用技能时,`Agent.skills_path` 指向此位置(否则设为 `None`)。本作业中智能体只会使用一个技能(`./tasks/code-skills/` 下的一个子文件夹),但原则上智能体可以使用许多技能,这会让渐进式披露更加重要和有用。

> **TODO(1.4)**
> 校验 `skills_path`,在每个子目录中发现一个 `SKILL.md`,解析其 YAML frontmatter(文件开头 `---` 标签之间的内容),并返回以 frontmatter 中 `name` 为键的映射。每个值必须包含:给模型技能目录用的简明 `metadata` 字符串,以及供 `invoke_skill` 使用的技能文件完整 `content`。对重复名称、格式错误或缺失的 frontmatter,用清晰的 `ValueError` 拒绝。如果有任何技能可用,把它们的描述/metadata 放进提示中给智能体。

例如,`SKILL.md` 的 `content` 是
```
---
name: hello-world
description: Write "hello, world" to the terminal
---

echo "hello, world"
```
而 `metadata` 是
```
name: hello-world
description: Write "hello, world" to the terminal
```

### 5. 运行并检查修复

现在,让智能体修复一个软件问题。

```bash
make run-code-agent
make check-part1
```

默认模型是 `deepseek/deepseek-v4-flash-0731`。智能体接收 `tasks/chess-terminal-move/problem_statement.md`,在 `/testbed` 内工作,必须复现、修复并验证故障。一次运行会产生:

- `artifacts/fix.patch`
- `artifacts/part1-trajectory.json`

`make check-part1` 会把生成的补丁应用到全新的 testbed,运行公开回归测试和国际象棋应用的现有测试套件。不要直接编辑 `chess_app/` 中的目标代码。

完成 Part 1 后,你的解法应通过关于提示构造、截断、步数限制、格式错误/未知工具和补丁提交的测试。

## Part 2:实现上下文压缩

漫长的 ReAct 记录会增加成本,并最终挤占有用的上下文。随着智能体承担越来越复杂、长时间跨度的任务,它们最终会触及语言模型上下文窗口的极限。为了让智能体高效地处理更长时间运行的任务,我们希望只保留与先前动作和观察相关的信息。这通过把上下文压缩进工作记忆(working memory)来实现。现在,你将在共享的 `Agent` 中实现一个由模型生成的工作记忆;不要使用服务商特定的压缩端点。

> **TODO(2.1)**
> 实现 `Agent.compact_context`。提示模型压缩上下文。压缩的系统提示应要求简洁、事实性的工作记忆,并保留:目标、约束、文件、命令、修改、具体结果、失败的尝试、测试、阻塞点和下一步动作。只对旧前缀做摘要;逐字保留原始的系统/任务消息,并至少保留最近一次完整的助手动作及其所有关联的工具观察。得到的摘要应改变 `build_prompt` 的输出,并缩短提示的长度。

**不要**动 agent 类的 `api_prompt` 和 `api_responses` 属性。它们用于簿记和评测。

> **TODO(2.2)**
> 在共享循环中每次发起新的动作请求之前调用 `maybe_compact_context()`。它已经估算活跃 token 数、处理阈值,并记录压缩事件用于日志。

以 6000 token 阈值运行内置的 `django__django-15368` 任务:

```bash
COMPACT_THRESHOLD=6000 \
SWEBENCH_PATCH=artifacts/django__django-15368.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368
make check-swebench INSTANCE=django__django-15368
```

设置 `COMPACT_THRESHOLD=0` 可去掉压缩标志,运行完整上下文基线。如果要同时保留两次运行,请使用不同的输出文件名,例如:

```bash
COMPACT_THRESHOLD=0 \
SWEBENCH_PATCH=artifacts/django__django-15368-baseline.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-baseline-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368
```

提交的压缩运行必须触发至少一次压缩、切实减少活跃上下文,并产生通过 `check-swebench` 的补丁。生成是随机的,所以它不必比每个基线样本使用更少的 ReAct 步数。

产生了压缩和无压缩两条轨迹后,对比两条轨迹的 token 用量。把观察写进 `artifacts/token-usage-analysis.md`,并简要解释你观察到的趋势:压缩与不压缩两种条件在上下文使用上有哪些权衡?

## Part 3:构建 ChessAgent

你的智能体在 Part 1 中修复了国际象棋应用的问题,现在可以构建一个下棋的智能体了。`ChessAgent` 复用 Part 1 和 2 的同一个循环。它执白棋,服务器的确定性机器人执黑棋。每当白方走出合法走法后,服务器自动回应。

### 1. 实现 `play_move`

你将首先定义一个新工具,让智能体可以玩应用中运行的游戏。遵循 [OpenAI 函数调用指南](https://developers.openai.com/api/docs/guides/function-calling)来定义工具。

> **TODO(3.1.a)**
> 定义名为 `play_move` 的 OpenAI 函数工具 schema。它必须只接受一个必填的字符串参数 `move`,说明走法使用 [UCI 记谱法](https://en.wikipedia.org/wiki/Universal_Chess_Interface)(例如 `e2e4` 和 `e7e8q`),并拒绝多余参数。

然后,实现运行该工具的机制。[^3]

> **TODO(3.1.b)**
> 在 `chess_tools.py` 中实现 `_play_move`。解析参数并向 `/api/move` 发送 POST `{"move": <uci move>}`。返回其序列化的 JSON 对象。捕获工具抛出的任何错误,把错误信息包在 `<chess_error></chess_error>` 之间返回,让智能体去处理。要覆盖以下情况:格式错误的 JSON 参数、参数不是对象、缺失或非字符串的 fen、非字符串的 move、服务器拒绝的位置或走法,以及传输失败。

你可以查看 `chess_app` 了解 API 的工作方式。

用 `format_state` 方法格式化成功返回的状态,更新 `last_state`,并根据 `game_over` 设置 `finished`。用原始的 `tool_call_id` 关联观察。格式错误的 JSON、未知工具、非法走法和网络错误都变成 `<chess_error>...</chess_error>` 观察。因为实时局面在走子后会变化,一组并行调用中最多执行一个走法,其余以可恢复的方式拒绝。

初始观察和成功走子后的观察已经展示棋盘、黑方回应和下一步合法走法,不需要单独的读盘工具。

运行:

```bash
make run-chess-agent
```

这会应用你的 `fix.patch`、启动修复后的服务器,并保存:

- `artifacts/part3-trajectory.json`
- `artifacts/game-result.json`

打印出来的 HTTPS URL 提供棋盘和 API。智能体下棋时棋盘会轮询实时状态;棋盘上的按钮会刷新状态而不重置游戏。`CHESS_TIMEOUT=1800` 控制沙箱的生存时间。

### 2. 运行观察 A/B 实验

为了了解工具界面对智能体行为的影响,你将对两个模型比较"只有棋盘"与"棋盘加合法走法"两种观察。运行器无需修改源码即可配置,并使用不同的文件名:

```bash
make run-obs-deepseek-no-legal
make run-obs-deepseek-legal
make run-obs-gpt-oss-no-legal
make run-obs-gpt-oss-legal
```

对四次运行中的每一次,记录:`play_move` 总调用数、因非法被拒绝的调用数、无效走法率,以及是否达到 `game_over: true`。把简短比较写进 `artifacts/observation-experiment.md`。评分依据是实验和证据,而不是某个特定结果或是否赢棋。

### 3. 添加 `simulate_move`

为了让智能体能够规划更复杂的策略,你要给它模拟走子的能力。模拟让智能体可以模拟走出某步棋的效果,但不真正改变实际棋盘的状态。你将定义并注册 `SIMULATE_MOVE_TOOL`,实现 `_simulate_move`,并把它加进现有的分发器(dispatcher)。

> **TODO(3.3.a)**
> 像 `play_move` 工具那样定义 `simulate_move` 工具。

`simulate_move(fen, move=None)` 调用 `POST /api/simulate`。

只给 `simulate_move` 传一个完整的六字段 [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) 会返回该局面及其合法走法。传 FEN 加 UCI 走法则返回恰好一着之后的局面,任意一方皆可。返回 JSON,让 Python 能使用 `fen`、`squares`、`turn`、`legal_moves` 和终局结果字段。

> **TODO(3.3.b)**
> 在 `chess_tools.py` 中实现 `_simulate_move`。解析参数,用 FEN 和可选走法调用给定的 `/api/simulate` 端点,并返回其序列化 JSON。捕获工具抛出的任何错误,把错误信息包在 `<chess_error></chess_error>` 之间返回,让智能体去处理。要覆盖以下情况:格式错误的 JSON 参数、参数不是对象、缺失或非字符串的 fen、非字符串的 move、服务器拒绝的位置或走法,以及传输失败。

你应该用 `ChessAgent.chess_client` 来发送请求。

### 4. 添加 `run_python`

有了在棋盘上模拟走子的能力,智能体现在可以进行更复杂的规划形式。能够执行代码将让智能体更可靠地执行它想出的计划,你将通过程序化工具调用(programmatic tool calling)启用这一点。定义并注册 `RUN_PYTHON_TOOL`,实现 `_run_python`,并把它加进分发器。模型提供 `run_python(code)`,代码片段可以把 `simulate_move` 和 `play_move` 当作普通的同步 Python 函数调用。

> **TODO(3.4)**
> 解析参数,并在沙箱中运行代码,让已注册的工具按名字可用。`/opt/assignment/sandbox_python.py` 是 `env` 沙箱上的一个脚本,能访问本文件中相同的工具定义。用它来运行模型作为 run_python 工具参数生成的代码。该脚本接受两个位置参数——`port` 和 base64 编码的代码字符串(避免引号问题)。实现这个工具调用。
> 该脚本打印一个包含运行代码所得 `stdout`、`stderr` 和 `error` 的 JSON 对象——原样返回该字符串。非零返回码意味着沙箱本身失败,而不是模型代码失败。把 `exception_info` 或 `stderr` 报告为 <chess_error>。
> 如果出现类型不匹配或解析失败等问题,返回 <chess_error>{message}</chess_error>。

模型写的代码不得在本地智能体进程中运行。`_run_python` 把 base64 编码的代码通过 `env.execute` 发送给提供的沙箱运行器:

```text
python /opt/assignment/sandbox_python.py <port> <base64-code>
```

返回运行器带 `stdout`、`stderr` 和 `error` 的 JSON 字符串。沙箱命令非零退出变成 `<chess_error>`。代码片段抛出的 Python 异常属于成功的运行器调用,归入其 `error` 字段。每个片段执行后,重新读取实时棋盘、更新 `last_state` 和 `finished`,并把格式化的状态附到观察后面;否则模型可能重复走出代码片段已经提交的走法。

用以下命令启用这些工具:

```bash
uv run assignment-play-chess --programmatic-tools
```

实现之后,你的智能体应当能产生一段选择走法的代码,并在游戏中走出该走法。你可能仍会发现下棋智能体并不使用手头的工具走出好棋,而是经常直接用 `play_move` 行动。为了让它更有结构和策略,我们再次回到之前探索过的技能概念。

### 5. 加载并使用棋类技能

虽然运行 Python 代码的能力给了智能体执行复杂计划的选择,但智能体可能没有这个倾向。为了给它一个可执行的具体策略,我们在 `tasks/chess-skills/select-move` 提供了另一个技能。为了让模型使用它,你要赋予 `ChessAgent` `CodeAgent` 的一部分技能使用能力。由于工具执行机制略有不同,你需要在 `_invoke_skill(skills, arguments)` 中为 `ChessAgent` 重新实现 `invoke_skill` 工具的处理,仅当技能已加载时注册 `INVOKE_SKILL_TOOL`,并返回指定技能的完整内容。

> **TODO(3.5)**
> 解析参数并返回指定技能的内容。如果出现类型不匹配或解析失败等问题,返回 <chess_error>{message}</chess_error>。

```bash
uv run assignment-play-chess \
  --programmatic-tools \
  --skills-path tasks/chess-skills \
  --trajectory artifacts/part3-python-skill-trajectory.json
```

轨迹必须显示:先 `invoke_skill`,然后 `run_python` 代码调用 `simulate_move` 进行搜索,并用一次 `play_move` 提交。读完技能后每回合只做一次直接的 `play_move` 调用,不能证明这些工具在协同工作。游戏不必下完或获胜。

## 评分(Grading)

本作业满分 **100 分**。每行独立评分;一次随机的模型运行失败不会抹掉不相关实现的得分。

| 部分 | 评分标准 | 分数 | 证据 |
|---|---|---|---|
| 1 | 提示构造与有效的动作/观察历史 | 6 | 私有单元测试 |
| 1 | ReAct 生命周期、纯文本恢复、步数限制、清理、轨迹 | 6 | 私有单元测试 |
| 1 | 编码工具分发、可恢复错误、补丁提交 | 6 | 私有单元测试 |
| 1 | 国际象棋补丁可应用并通过私有/回归测试 | 8 | 在全新 testbed 中回放补丁 |
| 1 | 技能发现与 `invoke_skill` 行为 | 4 | 私有测试 |
| 2 | 压缩触发与模型生成的摘要 | 6 | 私有测试与轨迹 |
| 2 | 原始指令和完整的近期工具步骤保持有效 | 6 | 私有测试与轨迹 |
| 2 | 可审计的压缩切实减少活跃上下文 | 4 | 压缩事件与用量 |
| 2 | 完整上下文 vs. 压缩的 token 用量分析报告 | 4 | 报告 |
| 2 | SWE-bench 补丁通过 FAIL_TO_PASS 和 PASS_TO_PASS | 8 | 补丁回放 |
| 3 | `play_move` schema 与注册 | 4 | 私有测试 |
| 3 | `play_move` 状态更新与可恢复错误 | 6 | 私有测试 |
| 3 | 基础下棋轨迹达到终局状态 | 4 | 轨迹与结果 |
| 3 | 四次运行的观察 A/B 实验完整 | 4 | 四条轨迹、四个结果 |
| 3 | 观察 A/B 实验报告 | 4 | 报告 |
| 3 | `simulate_move` 无状态行为与错误 | 6 | 私有单元/集成测试 |
| 3 | `run_python` 沙箱执行、状态刷新、错误 | 6 | 使用沙箱替身的私有测试 |
| 3 | 技能轨迹结合技能、程序化搜索与实盘走子 | 6 | 轨迹回放 |
| — | 完整、可解析、合规的提交 | 2 | 归档校验 |
| | **总分** | **100** | |

评分器会回放补丁和提交的轨迹;不会发起新的 LLM 调用。证据缺失或不一致只扣除受影响行的分数。教师的测试和参考补丁不包含在本仓库中。

## 提交(Submission)

提交一个 ZIP 压缩包,包含你在 `src/assignment/agent/` 下修改的文件以及以下产物:

```text
artifacts/fix.patch
artifacts/part1-trajectory.json
artifacts/django__django-15368.patch
artifacts/django__django-15368-trajectory.json
artifacts/token-usage-analysis.md
artifacts/part3-trajectory.json
artifacts/game-result.json
artifacts/part3-no-legal-moves-deepseek.json
artifacts/part3-no-legal-moves-deepseek-result.json
artifacts/part3-legal-moves-deepseek.json
artifacts/part3-legal-moves-deepseek-result.json
artifacts/part3-no-legal-moves-gpt-oss.json
artifacts/part3-no-legal-moves-gpt-oss-result.json
artifacts/part3-legal-moves-gpt-oss.json
artifacts/part3-legal-moves-gpt-oss-result.json
artifacts/observation-experiment.md
artifacts/part3-python-skill-trajectory.json
```

如果你修改了 `src/assignment/prompts.py`,也要一并包含。**不要**提交凭据、`.env`、任务文件、测试、子模块内容或教师文件。你的 ZIP 应包含 `src/` 和 `artifacts` 两个子文件夹,以及一个 `AI_USAGE.md` 文件。`AI_USAGE.md` 应详述你在本作业中对 AI 技术的使用:列出你用过的所有工具,并清楚描述你如何使用其中每一个。如果你没有使用任何 AI 辅助,也要在此文件中声明。我们不会评阅你提交的 `AI_USAGE.md`,但会通过测验(细节另行通知)检查你对所提交代码的理解。

[^1]: 在这类离线评测环境中,你通常把任务构造为来自用户的请求来指定它。更多示例见这些[文档](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/chatgpt?tabs=python-key%2Cdotnet-secure%2Cjavascript-secure&pivots=programming-language-python)。

[^2]: 来自服务商的重复 `call_0` ID 是有效的:把每条工具观察与同一助手动作中的调用配对,不要假设 ID 在整条轨迹中全局唯一。

[^3]: `_play_move` 和其他工具被隔离在 `chess_tools.py` 中,是为了让在远程沙箱中执行这些工具成为可能。请在这个代码结构内正确使用 Modal 沙箱来执行下棋动作。
