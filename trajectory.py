"""Trajectory collection: run an agent (with tool use + reference RAG) and evaluate."""

from __future__ import annotations
import ast
import contextvars
import json
import logging
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Callable

from models import TaskInstance, Trajectory, TaskType, SkillItem
from prompts import (
    EVALUATE_SYSTEM, EVALUATE_PROMPT,
    AGENT_SYSTEM, AGENT_SYSTEM_WITH_TOOLS,
    AGENT_PROMPT,
    SKILL_SECTION_TEMPLATE,
)
import llm

# Agent config

@dataclass
class AgentConfig:
    """Unified config for running an agent. Swap model/temperature/tool settings here."""
    model: str = "openai/gpt-5.4-mini"
    judge_model: str = "openai/gpt-5.4-mini"
    temperature: float = 0.0
    max_tool_rounds: int = 8
    max_env_steps: int = 30
    # Tool capabilities (only meaningful when a skill is provided)
    enable_web_search: bool = False   # give agent a real web_search tool
    execute_scripts: bool = False     # actually exec() skill scripts instead of mocking

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "judge_model": self.judge_model,
            "temperature": self.temperature,
            "max_tool_rounds": self.max_tool_rounds,
            "max_env_steps": self.max_env_steps,
            "enable_web_search": self.enable_web_search,
            "execute_scripts": self.execute_scripts,
        }


DEFAULT_CONFIG = AgentConfig()


# Evaluation


def _apply_grader_result(
    traj: Trajectory,
    result: dict,
    meta_key: str,
    *,
    extra_fields: dict[str, str] | None = None,
) -> Trajectory:
    """Populate a trajectory with results from a benchmark-specific grader.

    Handles the common result->trajectory mapping shared by all benchmark
    graders: sets success/score, merges the full result dict under `meta_key`,
    copies any optional extra fields, and records the error summary on failure.
    """
    traj.success = bool(result.get("passed"))
    traj.score = result.get("score")
    if traj.metadata is None:
        traj.metadata = {}
    traj.metadata[meta_key] = result
    for src_key, dst_key in (extra_fields or {}).items():
        if result.get(src_key):
            traj.metadata[dst_key] = result[src_key]
    if not traj.success:
        traj.error_summary = result.get("error_message")
    return traj


def evaluate_trajectory(
    traj: Trajectory,
    instance: TaskInstance,
    task_type: TaskType,
    *,
    rubric: str | None = None,
    model: str = "openai/gpt-5.4-mini",
) -> Trajectory:
    """Fill in traj.success and traj.score via a benchmark grader or LLM judge."""
    bench = instance.metadata.get("benchmark")
    model_output = str(traj.final_output or "")

    if bench in {"tau_bench", "skillsbench"}:
        # These custom runners return trajectories already scored by their
        # official benchmark verifiers. Do not replace that score with an LLM
        # judge.
        return traj

    if bench == "livecodebench":
        from benchmarks.livecodebench_adapter import evaluate_livecodebench_output
        result = evaluate_livecodebench_output(model_output=model_output,
                                               instance_metadata=instance.metadata)
        return _apply_grader_result(traj, result, "livecodebench_eval",
                                    extra_fields={"extracted_code": "extracted_code"})

    if bench == "spreadsheetbench":
        from benchmarks.spreadsheetbench_executor import evaluate_spreadsheetbench_output
        result = evaluate_spreadsheetbench_output(model_output=model_output,
                                                  instance_metadata=instance.metadata)
        return _apply_grader_result(traj, result, "spreadsheetbench_eval",
                                    extra_fields={"extracted_code": "extracted_code"})

    if bench == "mind2web":
        from benchmarks.mind2web_grader import evaluate_mind2web_output
        result = evaluate_mind2web_output(model_output=model_output,
                                          instance_metadata=instance.metadata)
        return _apply_grader_result(traj, result, "mind2web_eval")

    if bench == "pubmedqa":
        from benchmarks.pubmedqa_grader import evaluate_pubmedqa_output
        result = evaluate_pubmedqa_output(model_output=model_output,
                                          instance_metadata=instance.metadata)
        return _apply_grader_result(traj, result, "pubmedqa_eval")

    if bench == "chemllmbench":
        from benchmarks.chemllmbench_grader import evaluate_chemllmbench_output
        result = evaluate_chemllmbench_output(
            model_output=model_output,
            ground_truth=instance.ground_truth,
            instance_metadata=instance.metadata,
        )
        return _apply_grader_result(traj, result, "chemllmbench_eval",
                                    extra_fields={"extracted": "extracted_answer"})

    if bench == "aime":
        # AIME answers are always a 0-999 integer; the generic LLM judge
        # adds unnecessary noise, so we use rule-based exact-match instead.
        from benchmarks.aime_grader import evaluate_aime_output
        md = dict(instance.metadata or {})
        md.setdefault("ground_truth", instance.ground_truth)
        result = evaluate_aime_output(model_output=model_output, instance_metadata=md)
        return _apply_grader_result(traj, result, "aime_eval")

    if bench == "mcp_bench":
        from benchmarks.mcp_bench_grader import evaluate_mcp_bench_output
        result = evaluate_mcp_bench_output(model_output=model_output,
                                           instance_metadata=instance.metadata,
                                           judge_model=model)
        return _apply_grader_result(traj, result, "mcp_bench_eval")

    if bench == "toolbench":
        from benchmarks.toolbench_grader import evaluate_toolbench_output
        result = evaluate_toolbench_output(model_output=model_output,
                                           instance_metadata=instance.metadata,
                                           judge_model=model)
        return _apply_grader_result(traj, result, "toolbench_eval")

    if bench == "scienceworld":
        from benchmarks.scienceworld_grader import evaluate_scienceworld_output
        result = evaluate_scienceworld_output(model_output=model_output,
                                              instance_metadata=instance.metadata,
                                              judge_model=model)
        return _apply_grader_result(traj, result, "scienceworld_eval")

    if bench == "socialmaze":
        from benchmarks.socialmaze_grader import evaluate_socialmaze_output
        result = evaluate_socialmaze_output(model_output=model_output,
                                            instance_metadata=instance.metadata)
        return _apply_grader_result(traj, result, "socialmaze_eval")

    # Generic LLM judge fallback
    rubric_section = f"## Evaluation Rubric\n{rubric}\n" if rubric else ""
    result = llm.chat_json(
        EVALUATE_PROMPT.format(
            input=instance.input,
            ground_truth=instance.ground_truth or "(none)",
            output=traj.final_output,
            task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
            rubric_section=rubric_section,
        ),
        system=EVALUATE_SYSTEM,
        model=model,
    )
    traj.success = result["pass"]
    traj.score = result.get("score")
    return traj


# Tool definitions for agent

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. "
            "Use when you need up-to-date facts, documentation, or domain knowledge "
            "that is not in your training data or in the skill guidance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up.",
                }
            },
            "required": ["query"],
        },
    },
}


def _parse_script_functions(script_src: str) -> list[tuple[str, str, list[str]]]:
    """Parse a script string and return list of (func_name, docstring, param_names).

    Only TOP-LEVEL function definitions are returned. Nested helpers (e.g.
    `_coerce_list` inside `rank_suzuki_ligand`) are internal implementation
    detail - exposing them to the agent's tool schema would let the LLM
    call them by name, but they wouldn't exist in the exec() namespace
    after the enclosing function returns, so every such call would 500.
    """
    try:
        tree = ast.parse(script_src)
    except SyntaxError:
        return [("run_script", "Execute this helper script.", ["input"])]

    funcs = []
    for node in tree.body:  # iterate module body, NOT ast.walk - skip nested defs
        if not isinstance(node, ast.FunctionDef):
            continue
        # Skip private helpers (leading underscore) - the generator is
        # encouraged to use them for its own decomposition, but they are
        # not part of the tool surface.
        if node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node) or ""
        params = [
            arg.arg for arg in node.args.args
            if arg.arg not in ("self", "cls")
        ]
        if not params:
            params = ["input"]
        funcs.append((node.name, docstring, params))
    return funcs or [("run_script", "Execute this helper script.", ["input"])]


def _build_tools_from_skill(skill: SkillItem | None, config: AgentConfig | None = None) -> list[dict]:
    """Build OpenAI function-calling tool definitions from skill scripts."""
    tools: list[dict] = []

    if config and config.enable_web_search:
        tools.append(_WEB_SEARCH_TOOL)

    if not skill or not skill.scripts:
        return tools

    for script_idx, script_src in enumerate(skill.scripts):
        parsed = _parse_script_functions(script_src)
        for func_name, docstring, param_names in parsed:
            # Sanitise function name to be a valid OpenAI tool name
            safe_name = f"skill_{func_name}" if not func_name.startswith("skill_") else func_name
            safe_name = safe_name.replace("-", "_")[:64]

            properties = {
                p: {"type": "string", "description": f"Value for parameter '{p}'"}
                for p in param_names
            }
            tools.append({
                "type": "function",
                "function": {
                    "name": safe_name,
                    "description": (
                        f"[script_idx={script_idx}] "
                        + (docstring.split("\n")[0][:300] if docstring else script_src[:200])
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(param_names),
                    },
                },
            })
    return tools


def _build_reference_context(skill: SkillItem | None) -> str:
    """Build the reference-materials section for the agent prompt."""
    if skill and skill.reference_docs:
        lines = []
        for doc in skill.reference_docs:
            name = doc.get("name", "")
            summary = (doc.get("summary") or "").strip().replace("\n", " ")
            if not name:
                continue
            lines.append(f"- `{name}` - {summary}" if summary else f"- `{name}`")
        if lines:
            return (
                "\n## Reference documents (load on demand)\n"
                "The following reference documents are shipped with this skill but "
                "are NOT in your context yet. Fetch one by calling the "
                "`skill_load_reference` tool with `name=<exact filename>` - only "
                "when the document is directly relevant to the instance you are "
                "solving. Loading everything is wasteful.\n"
                + "\n".join(lines) + "\n---\n"
            )

    if not skill or not skill.references:
        return ""
    ref_parts = [f"- {ref}" for ref in skill.references]
    return (
        "\n## Reference Materials (optional)\n"
        "The following references may be relevant. Consult them only if they "
        "relate to the specific task at hand.\n"
        + "\n".join(ref_parts) + "\n---\n"
    )


def _serialize_tool_calls(tool_calls) -> list[dict] | None:
    if not tool_calls:
        return None
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in tool_calls
    ]


def _normalise_message(message: dict) -> dict:
    normalised = {
        "role": message.get("role", ""),
        "content": str(message.get("content", "") or ""),
    }
    for key in ("tool_calls", "tool_call_id", "name"):
        value = message.get(key)
        if value is not None:
            normalised[key] = value
    return normalised


def _build_tau_bench_system_prompt(
    wiki: str,
    skill_bundle: SkillItem | None,
) -> str:
    skill_section = ""
    if skill_bundle:
        skill_section = SKILL_SECTION_TEMPLATE.format(skill_body=skill_bundle.body).strip()
    reference_section = _build_reference_context(skill_bundle).strip()

    sections = [
        (
            "You are a customer-support agent operating inside an interactive benchmark.\n"
            "Follow the domain policy exactly.\n"
            "Use at most one tool call at a time.\n"
            "If you call a tool, do not also send a user-facing reply in the same turn.\n"
            "When you reply to the customer, write only the exact message they should see."
        ),
        "## Domain Policy\n" + wiki.strip(),
    ]
    if skill_section:
        sections.append(skill_section)
    if reference_section:
        sections.append(reference_section)
    return "\n\n".join(sections)


def _run_tau_bench_agent(
    instance: TaskInstance,
    skill_bundle: SkillItem | None,
    config: AgentConfig,
) -> Trajectory:
    from benchmarks.tau_bench_adapter import (
        create_tau_bench_env,
        get_tau_bench_types,
        summarise_tau_bench_outcome,
    )

    env = create_tau_bench_env(instance)
    Action, respond_action_name = get_tau_bench_types()
    tools = list(env.tools_info) + _build_tools_from_skill(skill_bundle, config)
    has_tools = bool(tools)
    system_prompt = _build_tau_bench_system_prompt(env.wiki, skill_bundle)

    env_reset_res = env.reset(task_index=instance.metadata["task_index"])
    all_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": env_reset_res.observation},
    ]

    t0 = time.time()
    msg = None
    reward = 0.0
    terminated = False
    last_observation = env_reset_res.observation
    tools_called: list[str] = []
    final_info = env_reset_res.info.model_dump()
    final_output = ""
    _tools_disabled = False
    max_env_steps = int(instance.metadata.get("max_env_steps", config.max_env_steps))

    for _ in range(max_env_steps):
        active_tools = None if _tools_disabled else (tools or None)
        try:
            msg = llm.chat_multi_turn(
                all_messages,
                model=config.model,
                temperature=config.temperature,
                tools=active_tools,
            )
        except Exception as exc:
            err_str = str(exc)
            if (
                "404" in err_str
                and ("tool" in err_str.lower() or "function" in err_str.lower())
                and not _tools_disabled
            ):
                logging.getLogger("trajectory").warning(
                    "Model %s does not support tool use - disabling tools and retrying.",
                    config.model,
                )
                _tools_disabled = True
                msg = llm.chat_multi_turn(
                    all_messages,
                    model=config.model,
                    temperature=config.temperature,
                    tools=None,
                )
            else:
                raise

        serialized_tool_calls = _serialize_tool_calls(msg.tool_calls)
        if serialized_tool_calls:
            serialized_tool_calls = serialized_tool_calls[:1]
        all_messages.append(
            _normalise_message(
                {
                    "role": "assistant",
                    "content": msg.content,
                    **({"tool_calls": serialized_tool_calls} if serialized_tool_calls else {}),
                }
            )
        )

        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            fn_name = tool_call.function.name
            tools_called.append(fn_name)

            if fn_name in env.tools_map:
                raw_args = tool_call.function.arguments
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        try:
                            args, _ = json.JSONDecoder().raw_decode(raw_args.lstrip())
                        except Exception:
                            args = {}
                            tool_err = f"[tool_error] invalid JSON arguments: {raw_args[:200]}"
                            all_messages.append(
                                _normalise_message(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": fn_name,
                                        "content": tool_err,
                                    }
                                )
                            )
                            continue
                    if not isinstance(args, dict):
                        args = {}
                else:
                    args = raw_args or {}
                try:
                    env_response = env.step(Action(name=fn_name, kwargs=args))
                except Exception as exc:
                    all_messages.append(
                        _normalise_message(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": fn_name,
                                "content": f"[tool_error] {type(exc).__name__}: {exc}",
                            }
                        )
                    )
                    continue
                reward = env_response.reward
                final_info = env_response.info.model_dump()
                last_observation = env_response.observation or ""
                all_messages.append(
                    _normalise_message(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": env_response.observation,
                        }
                    )
                )
                if env_response.done:
                    terminated = True
                    break
                continue

            tool_result = _execute_tool_call(tool_call, skill_bundle, config, tool_defs=tools)
            all_messages.append(
                _normalise_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": tool_result,
                    }
                )
            )
            continue

        final_output = msg.content or ""
        env_response = env.step(
            Action(name=respond_action_name, kwargs={"content": final_output})
        )
        reward = env_response.reward
        final_info = env_response.info.model_dump()
        last_observation = env_response.observation or ""
        all_messages.append(
            _normalise_message({"role": "user", "content": env_response.observation})
        )
        if env_response.done:
            terminated = True
            break

    latency = time.time() - t0
    threshold = float(instance.metadata.get("reward_threshold", 1.0))
    success = terminated and reward >= threshold - 1e-6
    error_summary, traj_metadata = summarise_tau_bench_outcome(
        instance=instance,
        reward=reward,
        final_info=final_info,
        tool_calls=tools_called,
        terminated=terminated,
        last_observation=last_observation,
    )

    agent_config = config.to_dict()
    agent_config["skill_id"] = skill_bundle.skill_id if skill_bundle else None
    agent_config["tools_used"] = has_tools and not _tools_disabled
    agent_config["tools_disabled_by_model"] = _tools_disabled
    agent_config["tools_called"] = tools_called
    agent_config["inference_model"] = config.model
    agent_config["benchmark"] = "tau_bench"

    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        instance_id=instance.instance_id,
        agent_config=agent_config,
        messages=[_normalise_message(m) for m in all_messages],
        final_output=final_output or last_observation,
        success=success,
        score=reward,
        error_summary=None if success else error_summary,
        latency=latency,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata=traj_metadata,
    )


# Agent runner


def run_agent(instance: TaskInstance, skill_bundle: SkillItem | None = None,
              config: AgentConfig | None = None) -> Trajectory:
    """Run an LLM agent on a single instance, optionally augmented with a skill bundle."""
    if config is None:
        config = DEFAULT_CONFIG

    if instance.metadata.get("benchmark") == "tau_bench":
        return _run_tau_bench_agent(instance, skill_bundle, config)

    if instance.metadata.get("benchmark") == "skillsbench":
        from benchmarks.skillsbench_adapter import run_skillsbench_agent
        return run_skillsbench_agent(instance, skill_bundle, config)

    skill_section = ""
    if skill_bundle:
        skill_section = SKILL_SECTION_TEMPLATE.format(skill_body=skill_bundle.body)

    reference_section = _build_reference_context(skill_bundle)
    tools = _build_tools_from_skill(skill_bundle, config)

    has_tools = bool(tools)
    system_prompt = AGENT_SYSTEM_WITH_TOOLS if has_tools else AGENT_SYSTEM

    prompt = AGENT_PROMPT.format(
        skill_section=skill_section,
        reference_section=reference_section,
        input=instance.input,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    all_messages = list(messages)

    t0 = time.time()
    msg = None
    tools_called: list[str] = []
    _tools_disabled = False  # set True if model reports it doesn't support tool use

    # Multi-turn loop for tool calls
    for _ in range(config.max_tool_rounds):
        active_tools = None if _tools_disabled else (tools or None)
        try:
            msg = llm.chat_multi_turn(
                all_messages, model=config.model, temperature=config.temperature,
                tools=active_tools,
            )
        except Exception as exc:
            # OpenRouter returns 404 when a model doesn't support function calling.
            # Fall back to no-tools mode and retry immediately.
            err_str = str(exc)
            if (
                "404" in err_str
                and ("tool" in err_str.lower() or "function" in err_str.lower())
                and not _tools_disabled
            ):
                logging.getLogger("trajectory").warning(
                    "Model %s does not support tool use - disabling tools and retrying.",
                    config.model,
                )
                _tools_disabled = True
                msg = llm.chat_multi_turn(
                    all_messages, model=config.model, temperature=config.temperature,
                    tools=None,
                )
            else:
                raise

        serialized_tool_calls = _serialize_tool_calls(msg.tool_calls)

        all_messages.append(_normalise_message({
            "role": "assistant",
            "content": msg.content,
            **({"tool_calls": serialized_tool_calls} if serialized_tool_calls else {}),
        }))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            tool_result = _execute_tool_call(tc, skill_bundle, config, tool_defs=tools)
            tools_called.append(tc.function.name)
            all_messages.append(_normalise_message({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            }))

    latency = time.time() - t0
    final_output = msg.content or "" if msg else ""

    agent_config = config.to_dict()
    agent_config["skill_id"] = skill_bundle.skill_id if skill_bundle else None
    agent_config["tools_used"] = has_tools and not _tools_disabled
    agent_config["tools_disabled_by_model"] = _tools_disabled
    agent_config["tools_called"] = tools_called
    agent_config["inference_model"] = config.model

    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        instance_id=instance.instance_id,
        agent_config=agent_config,
        messages=[_normalise_message(m) for m in all_messages],
        final_output=final_output,
        latency=latency,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def _execute_tool_call(
    tool_call,
    skill: SkillItem | None,
    config: AgentConfig | None = None,
    tool_defs: list[dict] | None = None,
) -> str:
    """Dispatch a tool call to web_search or a skill script."""
    fn_name = tool_call.function.name
    raw_args = tool_call.function.arguments
    try:
        args: dict = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON arguments for {fn_name}: {exc}. Raw: {str(raw_args)[:400]}"
    if not isinstance(args, dict):
        args = {}

    # Web search
    if fn_name == "web_search":
        query = args.get("query", "")
        if not query:
            return "Error: no query provided to web_search."
        try:
            return llm.web_search(query)
        except Exception as exc:
            return f"web_search error: {exc}"

    # Skill script execution
    if skill and skill.scripts:
        script_src: str | None = None

        desc = ""
        if tool_defs:
            for td in tool_defs:
                if td.get("function", {}).get("name") == fn_name:
                    desc = td["function"].get("description", "")
                    break

        # Description encodes script index as "[script_idx=N] ..."
        if "[script_idx=" in desc:
            try:
                idx = int(desc.split("[script_idx=")[1].split("]")[0])
                if 0 <= idx < len(skill.scripts):
                    script_src = skill.scripts[idx]
            except (ValueError, IndexError):
                pass

        if script_src is None:
            func_bare = fn_name.removeprefix("skill_")
            for src in skill.scripts:
                if f"def {func_bare}" in src or f"def {fn_name}" in src:
                    script_src = src
                    break

        if script_src is None and skill.scripts:
            script_src = skill.scripts[0]

        if script_src is not None:
            if config and config.execute_scripts:
                return _exec_script(script_src, fn_name, args)
            else:
                # Mock: return the script body so the model can reason from it
                return (
                    f"[Script content for '{fn_name}' - not executed in this mode]\n\n"
                    f"{script_src[:1000]}"
                )

    return f"Unknown tool '{fn_name}' called with args: {json.dumps(args)}"


def _exec_script(script_src: str, fn_name: str, args: dict) -> str:
    """Actually exec() a skill script and call the requested function."""
    namespace: dict = {}
    try:
        exec(compile(script_src, "<skill_script>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return f"Script compilation error: {exc}\n{traceback.format_exc(limit=5)}"

    bare_name = fn_name.removeprefix("skill_")
    func = namespace.get(fn_name) or namespace.get(bare_name)
    if func is None:
        candidates = [
            v for k, v in namespace.items()
            if callable(v) and not k.startswith("_")
        ]
        func = candidates[0] if candidates else None

    if func is None:
        return f"No callable found in script for tool '{fn_name}'."

    try:
        import inspect
        sig = inspect.signature(func)
        bound = sig.bind(**args)
        result = func(*bound.args, **bound.kwargs)
    except TypeError:
        try:
            # Try passing all args as a single positional string
            first_val = next(iter(args.values()), "") if args else ""
            result = func(first_val)
        except Exception as exc:
            return f"Script call error: {exc}\n{traceback.format_exc(limit=5)}"
    except Exception as exc:
        return f"Script execution error: {exc}\n{traceback.format_exc(limit=5)}"

    return str(result) if result is not None else "(no output)"


# Concurrent trajectory collection

def _run_and_eval(instance: TaskInstance, task_type: TaskType,
                  skill_bundle: SkillItem | None, config: AgentConfig,
                  rubric: str | None) -> Trajectory:
    """Run agent + evaluate a single instance (for concurrent use).

    One instance's failure (API error, malformed tool-call args, env crash, etc.)
    must NOT kill the whole batch. On unhandled exceptions, record a failed
    trajectory with the traceback and return it so aggregation continues.
    """
    try:
        traj = run_agent(instance, skill_bundle=skill_bundle, config=config)
        return evaluate_trajectory(traj, instance, task_type,
                                   rubric=rubric, model=config.judge_model)
    except Exception as exc:
        if instance.metadata.get("benchmark") == "skillsbench":
            # A BenchFlow/container/verifier error is not evidence that the
            # agent failed the task. Stop instead of poisoning SkillGen's
            # negative-example pool with an infrastructure failure.
            raise
        err_text = f"[agent_exception] {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
        traj = Trajectory(
            trajectory_id=f"err_{instance.instance_id}_{uuid.uuid4().hex[:8]}",
            instance_id=instance.instance_id,
            agent_config={
                "baseline_agent": getattr(config, "baseline_agent", None),
                "with_skill": skill_bundle is not None,
            },
            messages=[{"role": "system", "content": err_text}],
            final_output=err_text,
            success=False,
            score=0.0,
            error_summary=err_text.splitlines()[0][:500],
            metadata={"agent_exception": type(exc).__name__},
        )
        return traj


def collect_trajectories(
    instances: list[TaskInstance],
    task_type: TaskType,
    *,
    skill: SkillItem | None = None,
    config: AgentConfig | None = None,
    rubric: str | None = None,
    runs_per_instance: int = 1,
    max_workers: int = 16,
    progress_desc: str | None = None,
    on_trajectory: Callable[[Trajectory], None] | None = None,
) -> list[Trajectory]:
    """Run agent on all instances concurrently and evaluate each trajectory."""
    if config is None:
        config = DEFAULT_CONFIG
    args_list = [
        (inst, task_type, skill, config, rubric)
        for inst in instances
        for _ in range(runs_per_instance)
    ]
    # SkillsBench infrastructure/verifier failures are not negative examples.
    # Keep only ``max_workers`` paid slots in flight, submit replacements one at
    # a time, and stop submitting as soon as any running slot fails.  Already
    # running slots are drained so their official scores can still be cached
    # and checkpointed.  Results are returned in the frozen input order even
    # though the callback observes completion order.
    if args_list and all(
        args[0].metadata.get("benchmark") == "skillsbench" for args in args_list
    ):
        worker_count = max(1, int(max_workers))
        if worker_count == 1:
            trajectories: list[Trajectory] = []
            for args in args_list:
                trajectory = _run_and_eval(*args)
                trajectories.append(trajectory)
                if on_trajectory is not None:
                    on_trajectory(trajectory)
            return trajectories

        ordered: list[Trajectory | None] = [None] * len(args_list)
        first_error: tuple[BaseException, TracebackType | None] | None = None
        next_index = 0

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures: dict[object, int] = {}

            def submit_one(index: int) -> None:
                # Each worker needs its own Context so stage-level token and
                # budget accounting are not lost across thread boundaries.
                context = contextvars.copy_context()
                future = pool.submit(context.run, _run_and_eval, *args_list[index])
                futures[future] = index

            while next_index < len(args_list) and len(futures) < worker_count:
                submit_one(next_index)
                next_index += 1

            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                # Stable processing makes simultaneous completions auditable.
                for future in sorted(done, key=lambda item: futures[item]):
                    index = futures.pop(future)
                    try:
                        trajectory = future.result()
                        ordered[index] = trajectory
                        if on_trajectory is not None:
                            on_trajectory(trajectory)
                    except BaseException as exc:  # re-raised after in-flight drain
                        if first_error is None:
                            first_error = (exc, exc.__traceback__)

                if first_error is None:
                    while (
                        next_index < len(args_list)
                        and len(futures) < worker_count
                    ):
                        submit_one(next_index)
                        next_index += 1

        if first_error is not None:
            error, traceback_object = first_error
            raise error.with_traceback(traceback_object)
        if any(item is None for item in ordered):
            raise RuntimeError(
                "SkillsBench bounded collector returned without completing all slots"
            )
        return [item for item in ordered if item is not None]
    return llm.run_concurrent(
        _run_and_eval,
        args_list,
        max_workers=max_workers,
        progress_desc=progress_desc,
    )
