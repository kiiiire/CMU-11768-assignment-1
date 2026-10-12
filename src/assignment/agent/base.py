"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.

【译】两个智能体共享的、与领域无关的 ReAct 循环。
Part 1 在这里补全通用循环；本包中的两个子类只需提供各自的工具和工具执行器。
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
# 【译】TODO(Part 2)：编写指令，让模型为软件智能体生成简洁的工作记忆。
# 提示应保留具体的进展、失败、测试结果、约束和下一步行动，而不是照抄原始输出。
COMPACTION_SYSTEM_PROMPT = ""


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget.

    【译】当智能体耗尽模型调用预算（超过步数限制）时抛出。
    """


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation.

    【译】把终端执行结果格式化为紧凑的、带标签的模型观察。
    """

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            # 【译】给省略提示留出余量，使格式化后的值本身——而不只是
            # 保留的首尾片段——保持在长度限制以内。
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer.

    【译】在不依赖特定服务商分词器的情况下估算 prompt 的 token 数。
    """

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools.

    【译】带可插拔工具的 ReAct 智能体基类。
    """

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        # 【译】阈值为 None 表示关闭压缩。此时另外两个参数描述的是一次
        # 永远不会发生的压缩，因此回落到默认值，而不是留着一个 None
        # 让后续代码踩坑。
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        # 【译】每个智能体自己提供开场消息：常设指令（system_prompt），
        # 以及开启本次运行的任务说明（task_prompt）。
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # TODO(1.1.a): Add machinery to maintain agent state as it takes actions
        # and observes the results.
        # 【译】TODO(1.1.a)：添加机制，维护智能体在执行动作、观察结果
        # 过程中的状态。
        self.messages: list[dict[str, Any]] = []

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent.

        【译】加载暴露给该智能体的技能文件夹。
        """

        # TODO(1.4): Validate ``skills_path``, discover one ``SKILL.md``
        # per child directory, parse its YAML frontmatter (what's between the
        # `---` tags at the head of the file), and return a mapping
        # keyed by the frontmatter ``name``. Each value must contain a concise
        # ``metadata`` string for the model's skill catalog and the complete
        # ``content`` of the skill file for ``invoke_skill``. Reject duplicate
        # names and malformed or missing frontmatter with a clear
        # ``ValueError``.
        # 【译】TODO(1.4)：校验 ``skills_path``，在每个子目录中发现一个
        # ``SKILL.md``，解析其 YAML frontmatter（文件开头 `---` 标签之间的
        # 内容），并返回以 frontmatter 中的 ``name`` 为键的映射。每个值必须
        # 包含：给模型技能目录用的简明 ``metadata`` 字符串，以及供
        # ``invoke_skill`` 使用的技能文件完整 ``content``。对重复名称、
        # 格式错误或缺失的 frontmatter，要用清晰的 ``ValueError`` 拒绝。
        raise NotImplementedError

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it.

        【译】发送一次带工具的 Chat Completions 请求，并把响应标准化。
        """

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response.

        【译】返回语言模型响应中的相关部分。
        """

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        # TODO(1.1.a): Construct a sequence of messages that form the language
        # model prompt. This should include standing instructions, task
        # specification, prior interaction including observations, reasoning,
        # and actions from previous turns. Note that this method should be
        # domain-agnostic and construct the prompt in a way that would apply
        # to any of the inheriting domain-specific agents.
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        *self.messages,
        ]
        # 【译】TODO(1.1.a)：构造构成语言模型提示的消息序列。应包含常设
        # 指令、任务说明，以及之前轮次的交互（包括观察、推理和动作）。
        # 注意：该方法应当与领域无关，要以适用于任何继承它的领域特定
        # 智能体的方式来构造提示。

        # You want to be careful about which attributes of the class you modify
        # here as they may also be handled by the subclasses.
        # 【译】注意谨慎选择在这里修改类的哪些属性——子类可能也会处理它们。

        raise NotImplementedError

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage.

        【译】估算下一次提示的 token 数，用服务商最近一次返回的用量做校准。
        """

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all.

        【译】该智能体是否启用了上下文压缩。
        """

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits.

        【译】用模型生成的工作记忆替换提示的一部分，改变 `build_prompt`
        输出的内容。
        """

        # TODO(2.1): Prompt the model to compact the context. The system
        # prompt should ask for concise factual working memory and preserve
        # the objective, constraints, files, commands, edits, concrete
        # results, failed approaches, tests, blockers, and next action.
        # Summarize only an old prefix; retain the original system/task
        # messages verbatim and at least the latest complete assistant action
        # with all linked tool observations. The resulting summary should change
        # what `build_prompt` emits, and reduce the length of the prompt.

        # 【译】TODO(2.1)：提示模型压缩上下文。压缩用的系统提示应要求
        # 简洁、事实性的工作记忆，并保留：目标、约束、文件、命令、修改、
        # 具体结果、失败的尝试、测试、阻塞点和下一步动作。只对旧前缀做
        # 摘要；逐字保留原始的 system/task 消息，并至少保留最近一次完整
        # 的助手动作及其所有关联的工具观察。得到的摘要应改变
        # `build_prompt` 的输出，并缩短提示的长度。

        raise NotImplementedError

        compaction_prompt = []

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.
        # 【译】用 `compaction_response` 的结果去更新 `build_prompt` 的输出
        # 内容，但不要修改该对象本身，让它原样返回。

        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached.

        【译】当达到阈值时，在下一次动作请求之前执行压缩。
        """

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        # 【译】上下文还太短，暂时不需要压缩。
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        # 【译】步数还不够（每个 assistant 回合算一步），不足以触发压缩。
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal.

        【译】运行 ReAct 步骤，无论如何都会保存轨迹并停止 Modal 沙箱。
        """

        try:
            # TODO(1.2) Run the ReAct loop. Orchestrate the sequence of
            # prompting the language model to produce reasoning and actions,
            # extracting the tool calls produced by the model, and executing
            # the tool calls to obtain the agent's observation for the next
            # step. Ensure you identify when the agent has completed the task
            # by setting `Agent.finished`. If the agent exceeds the
            # `step_limit`, raise `StepLimitError`.

            # 【译】TODO(1.2)：运行 ReAct 循环。编排以下序列：提示语言模型
            # 生成推理和动作、提取模型产出的工具调用、执行工具调用以获得
            # 智能体下一步的观察。确保通过设置 `Agent.finished` 来识别智能体
            # 何时完成任务。如果智能体超过 `step_limit`，抛出
            # `StepLimitError`。

            # TODO(2.2) Call `maybe_compact_context()` before each new action
            # request in your shared loop. It already estimates active tokens
            # and handles the threshold, and tracks compaction events for
            # logging.

            # 【译】TODO(2.2)：在共享循环中每次发起新的动作请求之前调用
            # `maybe_compact_context()`。它已经完成活跃 token 的估算和阈值
            # 判断，并记录压缩事件用于日志。

            raise NotImplementedError
        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            # 【译】这一块是提供的基础设施，不要修改：即使运行失败，
            # 也必须留下轨迹。
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations.

        【译】执行领域特定的工具调用，并返回与各调用关联的观察消息。
        """

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        # 【译】这里不需要实现任何东西。该方法是领域特定的，
        # 由相应的子类实现。
        raise NotImplementedError
