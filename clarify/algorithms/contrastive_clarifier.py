# SPDX-FileCopyrightText: 2026 Dinh <dinhtd.b23cc038@stu.ptit.edu.vn>
# SPDX-License-Identifier: MIT

"""ContrastiveClarifier: draft an implementation first, then always ask one
confirmation question about the draft's riskiest assumption, regenerate with the
answer, repair against the prompt's own examples, and pick the final code by
execution agreement among a few regenerated candidates.

Team: Dinh
Team Members: Dinh
Main Contact: dinhtd.b23cc038@stu.ptit.edu.vn
"""

import ast
import json
import re

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import TooManyQuestionException

GENERATE = """Implement the requested Python function. Use only the standard
library. Preserve the required function name/signature. Public requirements and
the user's clarification are authoritative; when a clarification contradicts an
example in the prompt, follow the clarification. Do not invent facts.
Return only a fenced python implementation, no questions or explanations.
"""

SELF_TEST = """You will be shown a programming requirement and a candidate
implementation. Write a short Python test script using plain `assert`
statements that checks the implementation ONLY against behavior explicitly
given in the requirement (doctest-style ">>> " examples, or literal
input/output pairs stated in the prompt). Write one assert per example, each
with a message showing the call, e.g.
assert f(1) == 2, "f(1) expected 2"
Do not invent new edge cases. Do not redefine the function. If the requirement
gives no concrete example to check, return an empty fenced code block.
Return only a fenced ```python``` block, no explanations.
"""

ASK = """You are a coding assistant about to deliver the draft implementation
below for the user's requirement. The requirement may be ambiguous, incomplete,
vague, or contradictory. You may ask the user exactly ONE question before
delivering.

Pick the single assumption in the draft that is most likely to differ from
what the user actually wants, i.e. the one where a different reading changes
the output on ordinary inputs. Priorities:
1. If the draft disagrees with an example in the requirement, the requirement
   contradicts itself: ask which behavior is intended, quoting the example.
2. Otherwise, what the function returns (its exact type and format, e.g. list
   vs tuple vs string, True/False vs a message) or what its inputs are (number,
   order and kind of arguments) whenever the requirement leaves them open.
3. Otherwise, the core meaning of a vague or missing phrase (what exactly is
   computed, ordering, ties, inclusive/exclusive bounds, what counts as X).
Do not ask about edge cases (empty input, zero, negatives, invalid input)
unless the requirement leaves nothing above open; the tests rarely hinge on
them.

{avoid}Never ask for the algorithm, tests, or code.

Phrase it as a confirmation of the draft's concrete behavior, e.g.
"I will assume <specific behavior, with a small example input and output>. Is
that right, or should it <the main alternative>?"
Reply with only the question, on one line.
"""

AVOID = """Never ask about: Unicode/accented characters, invalid or malformed inputs,
input validation, type checking, performance, or error handling — unless the
requirement itself mentions them. """

PROBE = """Write {num_probes} small, valid inputs for the function {entry_point}
described below: typical cases and boundary cases (empty input, ties, duplicates,
zero, bounds) that a correct implementation must handle. Respect the user's
clarification if one is given. Use only inputs the description allows. Return only
a fenced python block that defines a list named INPUTS; each element is a tuple of
the positional arguments of one call, e.g. INPUTS = [([1, 2, 3], 2), ([], 0)].
"""

HARNESS = r"""
import copy, json, signal

def _alarm(signum, frame):
    raise TimeoutError("probe timeout")

signal.signal(signal.SIGALRM, _alarm)
_rows = []
_inputs = []
try:
    _ns = {}
    signal.alarm(5)
    exec(PROBE_CODE, _ns)
    signal.alarm(0)
    _inputs = list(_ns["INPUTS"])[:12]
except BaseException:
    signal.alarm(0)
for _code in CANDIDATES:
    _row = []
    try:
        _cns = {"__name__": "candidate"}
        signal.alarm(5)
        exec(_code, _cns)
        signal.alarm(0)
        _fn = _cns[ENTRY]
    except BaseException:
        signal.alarm(0)
        _rows.append(None)
        continue
    for _inp in _inputs:
        _args = _inp if isinstance(_inp, tuple) else (_inp,)
        try:
            signal.alarm(2)
            _out = "ok:" + repr(_fn(*copy.deepcopy(_args)))[:200]
            signal.alarm(0)
        except BaseException as _e:
            signal.alarm(0)
            _out = "raises " + type(_e).__name__
        _row.append(_out)
    _rows.append(_row)
print("@@PROBE@@" + json.dumps({"n": len(_inputs), "rows": _rows}))
"""

EXAMPLE_FAIL = "Example check: the draft FAILS an example stated in the requirement:\n{output}"
EXAMPLE_OK = "Example check: the draft passes the examples stated in the requirement."
EXAMPLE_NONE = "Example check: the requirement states no checkable examples."


def extract_code(response, entry_point):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.S)
    for code in blocks + [response]:
        try:
            tree = ast.parse(code.strip())
        except (SyntaxError, ValueError):
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
            for node in tree.body
        ):
            return code.strip()
    return None


def extract_fenced_code(response):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.S)
    return blocks[0].strip() if blocks and blocks[0].strip() else None


def clean_question(text):
    text = text.strip().strip("`").strip()
    text = re.sub(r"^(question|q)\s*:\s*", "", text, flags=re.I)
    return " ".join(text.split())


class ContrastiveClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "repair_attempts": 1,  # Syntax/entry-point repairs, not hidden-test feedback.
        "self_test_attempts": 1,  # Example-based repair after clarification.
        "example_check": False,  # Run the draft on the prompt's own examples before asking.
        "avoid_topics": False,  # Steer questions away from validation/Unicode/error handling.
        # Final code is chosen by execution consensus among this many candidates
        # (the first is the usual one; ties keep it).
        "final_candidates": 3,
    }

    def generate_code(self, env, messages, entry_point):
        response = env.llm(messages)
        for attempt in range(int(self.config["repair_attempts"]) + 1):
            code = extract_code(response, entry_point)
            if code is not None:
                return code, response
            if attempt == int(self.config["repair_attempts"]):
                break
            messages = messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": "Return syntactically valid Python defining "
                    + entry_point
                    + ". Return code only, not a clarification question.",
                },
            ]
            response = env.llm(messages)
        return None, response

    def example_check_run(self, env, public, code):
        """Returns (test_code, failure_output or None). test_code is None if no examples."""
        test_response = env.llm(
            [
                {"role": "system", "content": SELF_TEST},
                {
                    "role": "user",
                    "content": public + "\n\nImplementation:\n```python\n" + code + "\n```",
                },
            ]
        )
        test_code = extract_fenced_code(test_response)
        if not test_code:
            return None, None
        output = env.exec_code(code + "\n\n" + test_code)
        if "Traceback" in output or "Error" in output:
            return test_code, output[-1200:]
        return test_code, None

    def run(self, env, problem):
        # Deliberately read only public prompt/entry point, never task IDs or oracles.
        entry_point = problem["entry_point"]
        public = json.dumps(
            {"prompt": problem["prompt"], "entry_point": entry_point}, ensure_ascii=False
        )
        base = [{"role": "system", "content": GENERATE}, {"role": "user", "content": public}]

        draft, raw = self.generate_code(env, base, entry_point)

        clarification = None
        question = None
        if env.can_ask():
            check = EXAMPLE_NONE
            if draft is not None and self.config["example_check"]:
                test_code, failure = self.example_check_run(env, public, draft)
                if failure:
                    check = EXAMPLE_FAIL.format(output=failure)
                elif test_code:
                    check = EXAMPLE_OK
            question = clean_question(
                env.llm(
                    [
                        {
                            "role": "system",
                            "content": ASK.replace(
                                "{avoid}", AVOID if self.config["avoid_topics"] else ""
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Requirement:\n"
                            + problem["prompt"]
                            + "\n\nDraft:\n```python\n"
                            + (draft or raw)
                            + "\n```\n\n"
                            + check,
                        },
                    ]
                )
            )
            if question and "```" not in question:
                try:
                    clarification = env.ask_human(question)
                except TooManyQuestionException:
                    clarification = None

        messages = list(base)
        code = draft
        if clarification:
            messages += [
                {"role": "assistant", "content": question},
                {"role": "user", "content": clarification},
            ]
            new_code, raw = self.generate_code(env, messages, entry_point)
            code = new_code or code
        final_messages = list(messages)
        if code is None:
            # Never return an unfenced or truncated response: the evaluator only
            # accepts a closed ```python block.
            code = "def " + entry_point + "(*args, **kwargs):\n    return None"

        # Example-based repair. Skipped when the user's answer may have overridden
        # an example (the clarification is authoritative over stated examples).
        for _ in range(int(self.config["self_test_attempts"])):
            if env.prompt_cost >= env.prompt_budget * 0.8:
                break
            test_code, failure = self.example_check_run(env, public, code)
            if not failure:
                break
            messages += [
                {"role": "assistant", "content": "```python\n" + code + "\n```"},
                {
                    "role": "user",
                    "content": "Self-test against the requirement's examples failed:\n```\n"
                    + failure
                    + "\n```\nIf this example conflicts with my clarification above, keep "
                    "following the clarification and return the code unchanged. Otherwise "
                    "fix the implementation. Return only the fenced python implementation.",
                },
            ]
            new_code = extract_code(env.llm(messages), entry_point)
            if new_code is None:
                break
            code = new_code

        if int(self.config["final_candidates"]) > 1:
            code = self.select_by_consensus(env, problem, final_messages, clarification, code)
        return "```python\n" + code + "\n```"

    def select_by_consensus(self, env, problem, messages, clarification, code):
        """Sample more final candidates and keep the behavior most of them agree on."""
        entry_point = problem["entry_point"]
        candidates = [code]
        for _ in range(int(self.config["final_candidates"]) - 1):
            if env.prompt_cost >= env.prompt_budget * 0.8:
                break
            extra, _ = self.generate_code(env, messages, entry_point)
            if extra is not None:
                candidates.append(extra)
        if len(candidates) < 2:
            return code
        context = problem["prompt"]
        if clarification:
            context += "\n\nUser clarification: " + clarification
        probe = extract_fenced_code(
            env.llm(
                [
                    {
                        "role": "system",
                        "content": PROBE.format(num_probes=10, entry_point=entry_point),
                    },
                    {"role": "user", "content": context},
                ]
            )
        )
        if not probe:
            return code
        output = env.exec_code(
            "PROBE_CODE = "
            + repr(probe)
            + "\nCANDIDATES = "
            + repr(candidates)
            + "\nENTRY = "
            + repr(entry_point)
            + "\n"
            + HARNESS
        )
        marker = output.rfind("@@PROBE@@")
        if marker < 0:
            return code
        try:
            result = json.loads(output[marker + len("@@PROBE@@") :].strip().splitlines()[0])
        except (ValueError, IndexError):
            return code
        rows = result.get("rows", [])
        if not result.get("n") or len(rows) != len(candidates):
            return code
        # Exceptions never count as agreement: make each one unique to its candidate.
        signatures = [
            None
            if row is None
            else tuple(out if out.startswith("ok:") else (index, out) for out in row)
            for index, row in enumerate(rows)
        ]
        support = [
            0 if sig is None else sum(sig == other for other in signatures) for sig in signatures
        ]
        best = max(range(len(candidates)), key=lambda index: support[index])
        if support[best] <= max(support[0], 1):
            return code
        return candidates[best]
