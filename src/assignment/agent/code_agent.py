"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment
from assignment.prompts import CODE_AGENT_SYSTEM_PROMPT_TEMPLATE

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
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
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.append(EXECUTE_TOOL)
        self.tools.append(SEND_MESSAGE_TOOL)

        system_information = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )
        self.system_prompt = CODE_AGENT_SYSTEM_PROMPT_TEMPLATE.render(
            cwd=self.env.cwd,
            system_information=system_information,
        )
        self.task_prompt = (
            f"Fix the following issue in the repository at {self.env.cwd}:\n\n"
            f"{self.task}"
        )
        # Progressive disclosure: advertise only each skill's metadata here;
        # the full instructions stay behind `invoke_skill`.
        # 【译】渐进式披露：这里只展示每个技能的 metadata；
        # 完整说明保留在 `invoke_skill` 之后按需提供。
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.

        observations = []
        for call in tool_calls:
            call_id = call["id"]
            name = call["function"]["name"]
            raw_arguments = call["function"]["arguments"]
            if name == "execute":
                observations.append(self._run_execute(raw_arguments, call_id))
            elif name == "send_message":
                observations.append(self._run_send_message(raw_arguments, call_id))
            elif name == "invoke_skill" and self.skills:
                # Only dispatch when the tool is actually registered; without
                # skills a stray call falls through to the unknown-tool branch.
                # 【译】仅在该工具确已注册时分发；无技能时误入的调用
                # 落入未知工具分支。
                observations.append(self._run_invoke_skill(raw_arguments, call_id))
            else:
                observations.append(
                    self._tool_observation(
                        call_id,
                        f"Error: unknown tool '{name}'. Available tools: "
                        "`execute`, `send_message`"
                        + (", `invoke_skill`" if self.skills else "")
                        + ".",
                    )
                )
        return observations

    @staticmethod
    def _tool_observation(call_id: str, content: str) -> dict[str, str]:
        """Link one observation to the tool call it answers, as the API requires.

        【译】把一条观察关联到它所回答的工具调用上，这是 API 协议的要求。
        """
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    def _run_execute(self, raw_arguments: str, call_id: str) -> dict[str, str]:
        """Run one ``execute`` call in the sandbox and format its result.

        【译】在沙箱中执行一次 ``execute`` 调用，并格式化其结果。
        """
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            # Recoverable: report the problem so the model can retry.
            # 【译】可恢复：报告问题，让模型可以重试。
            return self._tool_observation(
                call_id,
                f"Error: the `execute` arguments are not valid JSON ({exc}). "
                'Pass an object like {"command": "ls -la"}.',
            )
        if not isinstance(args, dict) or not args.get("command"):
            return self._tool_observation(
                call_id,
                "Error: the `execute` tool requires a non-empty `command`.",
            )
        # Unlisted optional arguments fall through as None, which
        # `Environment.execute` treats as "use the sandbox default".
        # 【译】未列出的可选参数以 None 透传，`Environment.execute`
        # 会将其视为“使用沙箱默认值”。
        output = self.env.execute(
            args["command"],
            cwd=args.get("cwd"),
            timeout=args.get("timeout"),
            env=args.get("env"),
            shell=args.get("shell"),
        )
        return self._tool_observation(call_id, format_tool_output(output))

    def _run_send_message(self, raw_arguments: str, call_id: str) -> dict[str, str]:
        """Record the model's closing message and end the run.

        【译】记录模型的收尾消息，并结束本次运行。
        """
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return self._tool_observation(
                call_id,
                f"Error: the `send_message` arguments are not valid JSON ({exc}).",
            )
        summary = args.get("summary", "") if isinstance(args, dict) else ""
        # Delivering a message is this agent's signal that it considers the
        # task complete; the shared loop stops on `finished`.
        # 【译】发送消息是该智能体“认为任务已完成”的信号；共享循环
        # 通过 `finished` 停止。
        self.finished = True
        return self._tool_observation(
            call_id, f"Message delivered to the user: {summary}"
        )

    def _run_invoke_skill(self, raw_arguments: str, call_id: str) -> dict[str, str]:
        """Return one skill's full instructions, as asked for by name.

        【译】按名字返回一个技能的完整说明。
        """
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return self._tool_observation(
                call_id,
                f"Error: the `invoke_skill` arguments are not valid JSON ({exc}). "
                'Pass an object like {"name": "submit-task"}.',
            )
        skill_name = args.get("name") if isinstance(args, dict) else None
        skill = self.skills.get(skill_name or "")
        if skill is None:
            available = ", ".join(sorted(self.skills)) or "(none)"
            return self._tool_observation(
                call_id,
                f"Error: unknown skill '{skill_name}'. Available skills: {available}.",
            )
        # Reading a skill is not finishing the task: the model still has to
        # follow the instructions it just loaded.
        # 【译】读取技能不等于完成任务：模型还要照刚取到的说明继续干活。
        return self._tool_observation(call_id, skill["content"])
