# SPDX-FileCopyrightText: 2026 Dinh <dinhtd.b23cc038@stu.ptit.edu.vn>
# SPDX-License-Identifier: MIT

"""ContrastiveClarifier: confirm the draft's contract with one executed example, then
compile the answer into checks that choose and repair the final code.

Sentences that contradict the rest of the requirement (impossible implementation limits,
demands that cancel the main statement) are deleted before drafting; the edit may only
delete prose, never examples. Three independent drafts run on the examples stated in the
prompt and on probe inputs (kept only if they satisfy an LLM-written input precondition).
A structured description of the best draft (arguments, result format, rule) is rendered
into one fixed-form confirmation question whose example call shows the draft's real,
executed output; a question about two contradicting statements is asked only when both
are quoted verbatim from the requirement. The answer is compiled into checks. Only the
confirmed example and calls the answer explicitly relates to a result ("f(4) should
return 3", "for input 4, return 3"; negated statements excluded) are binding; stated
examples and calls the compiler derived are provisional, because prompt examples may be
corrupted, and a provisional check that no fresh implementation satisfies is dropped.
Implementations are ranked by binding checks, provisional checks, answer-entailed
properties and execution agreement; repairs and an answer-compliance review are kept
only if no satisfied binding check is lost. The delivered function retries failing calls
with adjusted arguments and gets builtin-safe name aliases.

Related ideas appear in the literature on interactive and test-driven code generation
(TiCoder, AssumptionMiner).

Team: Dinh
Team Members: Dinh (Posts and Telecommunications Institute of Technology),
    Van-Hoang Le (University of Luxembourg)
Main Contact: dinhtd.b23cc038@stu.ptit.edu.vn
"""

import ast
import builtins
import concurrent.futures
import difflib
import json
import keyword
import re

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import TooManyQuestionException

CONTRACT = """You help phrase ONE clarification question about a draft implementation of
{entry}. Read the requirement, the draft, and the table of what several independent
drafts returned on probe inputs. Describe what the DRAFT does, and name the most likely
point where the user wants something else.

Return only a JSON object with these keys:
"arguments": what the draft takes, with types and shape (a short phrase);
"returns": what it returns, with the exact type and format (a short phrase);
"rule": the rule it computes on ordinary inputs (one sentence, no code);
"probe": the number of the probe input whose result best shows that rule (prefer one
         where the drafts disagree);
"alternative": the most plausible different behaviour a user could want for the least
         clear part of the requirement, written to complete "or should it ...";
"conflict": null, or {{"a": "...", "b": "..."}} with two short VERBATIM quotes from the
         requirement text that demand different core behaviour (never an example against
         the text, never style, naming or implementation constraints).
Never mention how the function is named, efficiency or implementation constraints,
tests, or code."""

COMPILE = """A user answered one clarification question about a programming requirement.
Turn the answer into checks. Return only a fenced python block defining:
CONFIRMS = True if the answer accepts the example call in the question as correct,
           False if it rejects or changes it, None if it does not say;
CALLS = [((positional args,), expected), ...] for calls whose exact result the answer
        states, including values written in prose ("for 4 it should return 3") and a
        corrected example; [] if none;
PROPERTIES = up to three Python boolean expressions over `args` (tuple of the positional
        arguments) and `out` (the returned value) that the answer directly entails for
        every valid input, e.g. "isinstance(out, list)"; [] if unsure;
EXAMPLES = one of "CONFIRMED", "KEEP", "DROP", "CONFLICT" per stated example, in order:
        CONFIRMED if the answer supports it, DROP if the answer contradicts it, CONFLICT if
        it contradicts an explicit statement of the requirement text that the answer does
        not settle, KEEP otherwise.
Never invent a result the answer does not state."""

FINAL = """Requirement:
{prompt}
Entry point: {entry}

Before writing the code you asked the user: {question}
The user answered: {answer}

Treat the answer as authoritative for what it covers, even where it disagrees with the
requirement text or one of its examples. Everything it does not cover follows the
requirement text, its examples and the function name.{obligations}"""

OBLIGATIONS = """

The answer fixes these exact results; your implementation must produce them:
{lines}"""

REPAIR = """Running your implementation shows it does not meet these results, which come
from the user's answer or from the requirement's own examples:
{failures}

Change the behaviour so these results hold while keeping everything else as it is. If a
result contradicts the user's answer, follow the answer. Return only the fenced python
implementation."""

REVIEW = """Requirement:
{prompt}

You asked the user: {question}
The user answered: {answer}

Implementation of {entry}:
```python
{code}
```

Does the implementation follow the answer, which overrides the requirement text where
they disagree? If yes, reply exactly OK. If not, write "CONTRADICTION:" followed by the
sentence of the answer it violates, then a corrected implementation (change nothing else)
in one fenced python block."""

CLEAN = """Some requirements were edited after they were written: a sentence was added that cannot
be satisfied together with the rest. Typical added sentences demand an impossible implementation
limit (constant time, no loops or no recursion for a task that must look at every element), demand
that numbers or booleans be returned as strings, demand accepting a different kind of input, or end
the description with a "but ..." or "instead ..." clause that cancels its main statement. An edited
requirement may also contain one example whose result disagrees with the description and with the
other examples.

If the requirement below contains such a sentence or such an example, return it with exactly those
parts deleted. Do not change anything else: keep the signature, every other sentence and every other
example character for character, and do not add words. Vague wording, missing details, open choices
("in ascending or descending order") and unusual but consistent demands are not contradictions; keep
them. Most requirements contain no contradiction; then reply with the single word UNCHANGED.

Requirement:
{prompt}

Reply UNCHANGED, or the edited requirement inside one ```text block."""

GENERATE = """Implement the requested Python function. Use only the standard
library. Preserve the required function name/signature. Public requirements and
the user's clarification are authoritative; when a clarification contradicts an
example in the prompt, follow the clarification. Do not invent facts.
Return only a fenced python implementation, no questions or explanations.
"""

ROBUST = """
Coding conventions (they never override the requirement or the user's answer):
- Do not narrow the input domain beyond what is stated: do not assume list
  elements are hashable, numbers are positive, strings are ASCII, or inputs are
  non-empty. Prefer equality-based membership and counting over set/dict tricks
  when element types are not stated.
- Prefer the conventional reading of ordinary terms: split words on any
  whitespace (str.split()) unless a specific delimiter is required.
- Handle empty and vacuous cases naturally (an empty pattern occurs in any
  string, all() of nothing is True, no matches gives an empty result) unless the
  requirement says otherwise.
- The function name was chosen by the user: when the text is vague, prefer the
  behaviour the name describes.
- Do not add input validation that raises unless the requirement asks for
  errors; compute the answer for every input the requirement allows.
- Requirements on HOW to implement (no loops, no recursion, no built-in
  functions, a time or space complexity) cannot be observed by the user's tests:
  ignore them and write ordinary, correct code. Never fake the computation.
- Define the function at module level; no input(), prints, tests or top-level
  calls.
"""


PROBE = """Write {num_probes} small, valid inputs for the function {entry_point}
described below: typical cases first, then boundary cases (empty input, ties,
duplicates, zero, bounds) that a correct implementation must handle. Respect the
user's clarification if one is given. Use only inputs the description allows.
Return only a fenced python block that defines a list named INPUTS; each element
is a tuple of the positional arguments of one call, e.g.
INPUTS = [([1, 2, 3], 2), ([], 0)].
In the same block, also define valid(*args) -> bool: the input precondition,
i.e. True exactly when the call arguments satisfy what the description (and the
clarification) requires of each variable: number of arguments, types, value
ranges, structure (e.g. non-empty, sorted, distinct, length relations). Use only
stated or clearly implied conditions; when unsure about a condition, allow it.
"""

CALL_ADAPTER = r"""

def _call_adapter(fn, name):
    import copy, functools, inspect

    try:
        _params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        _params = None

    def _alternatives(args, error):
        alts = []
        message = str(error)
        positional = None
        required = None
        if _params is not None and not any(p.kind == p.VAR_POSITIONAL for p in _params):
            positional = [p for p in _params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if positional is not None:
            limit = len(positional)
            required = sum(p.default is p.empty for p in positional)
            sized_types = (list, tuple, str, dict, set)
            # f(seq, len(seq), ...) style calls: drop length companions first.
            companions = [i for i in range(1, len(args))
                          if isinstance(args[i], int) and not isinstance(args[i], bool)
                          and isinstance(args[i - 1], sized_types) and args[i] == len(args[i - 1])]
            if companions and len(args) - len(companions) >= required:
                alts.append(tuple(a for i, a in enumerate(args) if i not in companions))
            if len(args) > limit:
                alts.append(tuple(args[:limit]))
            if len(args) < required:
                missing = required - len(args)
                sized = next((a for a in args if hasattr(a, "__len__")), None)
                if sized is not None:
                    alts.append(tuple(args) + (len(sized),) * missing)
                alts.append(tuple(args) + (None,) * missing)
            if len(args) > 1 and required == 1:
                alts.append((list(args),))
                alts.append((tuple(args),))
            if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == required > 1:
                alts.append(tuple(args[0]))
        if len(args) == 2:
            a, b = args
            seq = (list, tuple, str, dict, set)
            if isinstance(a, seq) != isinstance(b, seq):
                alts.append((b, a))
        as_str = tuple(str(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in args)
        if as_str != args:
            alts.append(as_str)
        as_int = tuple(int(x) if isinstance(x, str) and x.strip().lstrip("-").isdecimal() else x for x in args)
        if as_int != args:
            alts.append(as_int)
        if "not iterable" in message:
            for i, x in enumerate(args):
                if isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 10000:
                    alts.append(tuple(args[:i]) + (list(range(1, x + 1)),) + tuple(args[i + 1:]))
                    alts.append(tuple(args[:i]) + ([x],) + tuple(args[i + 1:]))
        if "has no attribute 'get'" in message or "has no attribute 'items'" in message:
            for i, x in enumerate(args):
                if isinstance(x, (list, tuple)):
                    try:
                        alts.append(tuple(args[:i]) + (dict(x),) + tuple(args[i + 1:]))
                    except (TypeError, ValueError):
                        pass
        return alts

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Recursive calls inside fn resolve the module-level name: point it at the
        # original function while this call runs, so recursion depth is unchanged.
        namespace = fn.__globals__
        outer = namespace.get(name)
        namespace[name] = fn
        try:
            try:
                saved = copy.deepcopy(args)
            except Exception:
                saved = args
            try:
                return fn(*args, **kwargs)
            except (TypeError, AttributeError) as first:
                for alt in _alternatives(saved, first):
                    try:
                        return fn(*copy.deepcopy(alt), **kwargs)
                    except Exception:
                        continue
                raise first
        finally:
            namespace[name] = outer

    return wrapper


{entry} = _call_adapter({entry}, "{entry}")
"""

ALIAS = """

for _alias in {aliases!r}:
    if _alias not in globals():
        globals()[_alias] = {entry}
"""

HARNESS = r"""
import copy, hashlib, json, math, signal, time

def _alarm(signum, frame):
    raise TimeoutError("probe timeout")

signal.signal(signal.SIGALRM, _alarm)
_deadline = time.monotonic() + 40

def _same(a, b):
    # Like the evaluator's ==, with a small tolerance for floats.
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    try:
        return a == b
    except Exception:
        return False

def _call(fn, args, seconds):
    signal.alarm(seconds)
    try:
        return fn(*copy.deepcopy(args))
    finally:
        signal.alarm(0)

_inputs = []
try:
    _ns = {}
    signal.alarm(5)
    exec(PROBE_CODE, _ns)
    signal.alarm(0)
    _inputs = [i if isinstance(i, tuple) else (i,) for i in list(_ns.get("INPUTS", []))[:10]]
    _valid = _ns.get("valid")
    if callable(_valid):
        _kept = []
        for _inp in _inputs:
            try:
                if bool(_call(_valid, _inp, 2)):
                    _kept.append(_inp)
            except BaseException:
                signal.alarm(0)
        _inputs = _kept
except BaseException:
    signal.alarm(0)

_rows = []
for _code in CANDIDATES:
    _row = {"load": False, "checks": [], "probes": [], "props": 0}
    try:
        _cns = {"__name__": "candidate"}
        signal.alarm(5)
        exec(_code, _cns)
        signal.alarm(0)
        _fn = _cns[ENTRY]
    except BaseException:
        signal.alarm(0)
        _row["load"] = True
        _rows.append(_row)
        continue
    for _args, _expected in CHECKS:
        if time.monotonic() > _deadline:
            _row["checks"].append([False, "skipped"])
            continue
        try:
            _out = _call(_fn, _args, 2)
            _row["checks"].append([bool(_same(_out, _expected)), repr(_out)[:120]])
        except BaseException as _e:
            signal.alarm(0)
            _row["checks"].append([False, "raises " + type(_e).__name__])
    for _args in _inputs:
        if time.monotonic() > _deadline:
            _row["probes"].append(["skipped", "", ""])
            continue
        try:
            _out = _call(_fn, _args, 2)
            _row["probes"].append(["ok", hashlib.sha1(repr(_out).encode()).hexdigest()[:16], repr(_out)[:150]])
            for _p in PROPS:
                try:
                    signal.alarm(1)
                    if eval(_p, {"args": _args, "out": _out, "len": len, "isinstance": isinstance, "list": list,
                                 "tuple": tuple, "str": str, "int": int, "float": float, "bool": bool,
                                 "dict": dict, "set": set, "all": all, "any": any, "sorted": sorted, "sum": sum,
                                 "abs": abs, "min": min, "max": max, "type": type}):
                        _row["props"] += 1
                    signal.alarm(0)
                except BaseException:
                    signal.alarm(0)
        except BaseException as _e:
            signal.alarm(0)
            _row["probes"].append(["raises", type(_e).__name__, ""])
    _rows.append(_row)
print("@@HARNESS@@" + json.dumps({"rows": _rows, "inputs": [repr(i) for i in _inputs]}))
"""

_LITERAL_ERRORS = (ValueError, SyntaxError, TypeError, MemoryError, RecursionError)

# Words and arrows that connect a call written in prose to its result.
CONNECTOR_WORDS = ("==>", "==", "->", "=>", "➞", "→", r"returns?", r"should\s+return", r"should\s+be",
                   r"outputs?", r"gives?", "is", ":")
CONNECTOR = re.compile(r"\s*(?:" + "|".join(CONNECTOR_WORDS) + r")\s*", re.I)
OPERATORS = frozenset("+-*/%^&|<>=")


def extract_code(response, entry_point, declared=None):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.S)
    for code in blocks + [response]:
        try:
            tree = ast.parse(code.strip())
        except (SyntaxError, ValueError):
            continue
        names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if entry_point in names:
            return code.strip()
        if declared and declared in names:
            return re.sub(r"\b" + re.escape(declared) + r"\b", entry_point, code.strip())
    return None


def extract_fenced_code(response):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.S)
    return blocks[0].strip() if blocks and blocks[0].strip() else None


def clean_question(text):
    text = text.strip().strip("`").strip()
    text = re.sub(r"^(question|q)\s*:\s*", "", text, flags=re.I)
    return " ".join(text.split()).replace("`", "'")


def name_variants(entry_point):
    """Lower-case and snake_case spellings of the entry point that shadow nothing."""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", entry_point).lower()
    snake = re.sub(r"_+", "_", snake)
    reserved = set(dir(builtins)) | set(keyword.kwlist)
    return sorted({entry_point.lower(), snake} - {entry_point} - reserved)


def declared_name(prompt, entry_point):
    """The one function name the prompt declares with `def`, when it is not the entry point."""
    declared = set(re.findall(r"\bdef\s+([A-Za-z_]\w*)\s*\(", prompt))
    if len(declared) == 1 and entry_point not in declared:
        return declared.pop()
    return None


def leading_literal(text):
    ok, value, _ = literal_span(text)
    return ok, value


def literal_span(text):
    """(ok, value, characters consumed) for the longest literal at the start of text; a literal
    directly followed by an operator ("1/2", "2 + 2") is an expression and does not count."""
    lead = len(text) - len(text.lstrip())
    body = text.strip()
    candidates = (body[:end].strip().rstrip(".,;") for end in range(min(len(body), 400), 0, -1))
    for piece in candidates:
        if not piece:
            continue
        try:
            value = ast.literal_eval(piece)
        except _LITERAL_ERRORS:
            continue
        after = body[len(piece):].lstrip()
        if after[:1] in OPERATORS:
            return False, None, 0
        return True, value, lead + len(piece)
    return False, None, 0


def edited_text(reply):
    """The requirement inside the reply's text block, or None for UNCHANGED or no block."""
    match = re.search(r"```(?:text)?\s*\n(.*?)```", reply or "", re.S)
    if not match or (reply or "").strip().upper().startswith("UNCHANGED"):
        return None
    return match.group(1).strip("\n")


def accept_deletion(original, edited, names):
    """Keep an edit only if it deletes prose (never adds, rewrites or deletes examples), keeps the
    signature line and most of the requirement. Corrupted examples are left to the check filter."""
    if not edited or edited.strip() == original.strip():
        return None
    spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", original)]
    new = re.findall(r"\w+|[^\w\s]", edited)
    if not new or len(new) >= len(spans) or len(new) < 0.6 * len(spans):
        return None
    example = re.compile(r">>>|(?:" + "|".join(re.escape(n) for n in names if n) + r")\s*\(")
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, [s[0] for s in spans], new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "delete" or example.search(original[spans[i1][1]: spans[i2 - 1][2]]):
            return None
    for line in original.splitlines():
        if line.strip().startswith("def ") and line.strip() not in edited:
            return None
    return edited


def literal_calls(text, name):
    """(args, end offset) for every call `name(...)` in text whose arguments are all literals.
    The call ends at the first closing parenthesis that makes it parse."""
    calls = []
    for hit in re.finditer(r"\b" + re.escape(name) + r"\s*\(", text):
        opening = hit.end() - 1
        node = None
        for close in (m.start() for m in re.finditer(r"[)\]}]", text[opening:])):
            source = text[opening:opening + close + 1]
            try:
                node = ast.parse(name + source, mode="eval").body
            except _LITERAL_ERRORS:
                if not balanced_prefix(source):
                    break
                continue
            end = opening + close + 1
            break
        if not isinstance(node, ast.Call) or node.keywords:
            continue
        try:
            args = tuple(ast.literal_eval(arg) for arg in node.args)
        except _LITERAL_ERRORS:
            continue
        calls.append((args, end))
    return calls


def balanced_prefix(source):
    """False once the brackets of source (outside quotes) close below zero depth."""
    depth, quote, skip = 0, None, False
    for char in source:
        if skip:
            skip = False
        elif quote:
            if char == "\\":
                skip = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth <= 0:
                return False
    return True


def doctest_output(prompt, end):
    """The output lines that follow a `>>> call` ending at offset end, joined; '' if none."""
    lines = []
    for raw in prompt[end:].split("\n")[1:]:
        line = raw.strip()
        if line in ("", '"""', "'''", "```") or line.startswith(">>>"):
            break
        lines.append(line)
    return " ".join(lines)


def parse_examples(prompt, names):
    """Stated examples, at most eight: doctests and inline `f(args) <connector> value` pairs."""
    pairs, seen = [], set()
    for name in names:
        for args, end in literal_calls(prompt, name):
            line_head = prompt[prompt.rfind("\n", 0, end) + 1:end].lstrip()
            rest = prompt[end:]
            if line_head.startswith(">>>") and not rest.split("\n", 1)[0].strip():
                output = doctest_output(prompt, end)
                ok, value = leading_literal(output) if output else (False, None)
            else:
                link = CONNECTOR.match(rest)
                ok, value = (False, None)
                if link and rest[:link.end()].strip():
                    ok, value = leading_literal(rest[link.end():].split("\n", 1)[0])
            if ok and repr((args, value)) not in seen:
                seen.add(repr((args, value)))
                pairs.append((args, value))
    return pairs[:8]


def assigned_literals(code_block, names):
    """Top-level `NAME = <literal>` assignments for the wanted names in a code block."""
    if not code_block:
        return {}
    try:
        tree = ast.parse(code_block)
    except SyntaxError:
        return {}
    values = {}
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        if isinstance(target, ast.Name) and target.id in names:
            try:
                values[target.id] = ast.literal_eval(node.value)
            except _LITERAL_ERRORS:
                pass
    return values

def read_block_values(response, names):
    return assigned_literals(extract_fenced_code(response or ""), names)


def is_demo(node):
    """`if __name__ == ...` blocks and bare top-level print()/input() calls."""
    if isinstance(node, ast.If):
        return "__name__" in ast.unparse(node.test)
    call = node.value if isinstance(node, ast.Expr) else None
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in ("print", "input")


def sanitize(code):
    """Remove demo code from a module; anything else stays as written."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    body = [node for node in tree.body if not is_demo(node)]
    if len(body) == len(tree.body):
        return code
    tree.body = body
    return ast.unparse(tree)


FALLBACK_ASK = """You are about to deliver the draft below for the user's requirement and may ask
exactly ONE question first. State the draft's riskiest assumption about ordinary inputs
concretely, with one small example call and its output, and ask whether it is right or
what it should be instead. Never ask about naming, efficiency, tests or code. Plain text,
one line, no backticks. Reply with only the question."""


def call_text(entry, args_repr):
    """entry(<args>) from the repr of an argument tuple."""
    inner = args_repr.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
        if inner.endswith(","):
            inner = inner[:-1]
    return entry + "(" + inner + ")"


def phrase(text, limit=220):
    text = " ".join(str(text or "").replace("`", "'").split())
    return text[:limit].rstrip(" .;,")


def quoted_in(prompt, quote):
    norm = lambda s: " ".join(s.lower().split())
    quote = norm(quote.strip("\"' "))
    return len(quote) >= 8 and quote in norm(prompt)


POST_CALL = re.compile(
    r"\s*,?\s*(?:it|this|that|the\s+function|the\s+(?:correct\s+|expected\s+)?(?:result|output|return\s+value|value))?\s*"
    r"(?:(?:should|must|would|will|to)\s+)?(?:==>|==|->|=>|➞|→|returning|returns?|giving|gives?|outputs?|equals?|be|is|:)\s*", re.I)
LITERAL_RELATION = re.compile(r"\s*(?:should\s+(?:return|give|be)|returns|gives|==>|==|->|=>|➞|→)\s*", re.I)
NEGATION = re.compile(r"\b(?:not|never|instead\s+of|rather\s+than)\b|n't\b", re.I)
PROSE_HEAD = re.compile(r"\bfor\s+(?:the\s+|an?\s+)?(?:input|argument|value|call|n|x)?\s*(?:of\s+|=\s*)?", re.I)
PROSE_VERB = re.compile(r"\s*,?\s*(?:it\s+|the\s+function\s+|the\s+result\s+|the\s+output\s+)?(?:should\s+|must\s+|would\s+|will\s+)?"
                        r"(?:returns?|gives?|outputs?|be|is|equals?)\s+", re.I)


def answer_relations(answer, names):
    """(args, value) pairs the answer states explicitly: a written-out call with its result, or
    prose such as "for input 4, return 3". Negated statements state no relation."""
    text = str(answer or "")
    found = []
    for name in names:
        for args, end in literal_calls(text, name):
            rest = text[end:end + 300]
            connector = POST_CALL.match(rest)
            if not connector or not rest[: connector.end()].strip(" ,"):
                continue
            call_start = text.rfind(name + "(", 0, end)
            if NEGATION.search(text[max(0, call_start - 12):call_start] + rest[: connector.end()]):
                continue
            ok, value = leading_literal(rest[connector.end():].split("\n", 1)[0])
            if ok:
                found.append((tuple(args), value))
    # A quoted or bracketed literal that "should return" another one (single-argument calls).
    for start in re.finditer(r"['\"\[(]", text):
        if start.start() and text[start.start() - 1].isalnum():
            continue
        ok_arg, arg, used = literal_span(text[start.start(): start.start() + 300])
        verb = LITERAL_RELATION.match(text, start.start() + used) if ok_arg and used else None
        if not verb or NEGATION.search(text[max(0, start.start() - 40): verb.end()]):
            continue
        ok_value, value = leading_literal(text[verb.end(): verb.end() + 300].split("\n", 1)[0])
        if ok_value:
            found.append(((arg,), value))
    for head in PROSE_HEAD.finditer(text):
        after = text[head.end(): head.end() + 400]
        ok_arg, arg, used = literal_span(after)
        verb = PROSE_VERB.match(after[used:]) if ok_arg and used else None
        if not verb or NEGATION.search(after[: used + verb.end()]):
            continue
        ok_value, value = leading_literal(after[used + verb.end():].split("\n", 1)[0])
        if ok_value:
            found.append(((arg,), value))
    return found


def stated_in(relations, args, expected):
    """True when the answer explicitly states that these arguments give this result."""
    def same(a, b):
        return type(a) is type(b) and a == b
    return any(len(a) == len(args) and all(same(x, y) for x, y in zip(a, args)) and same(v, expected)
               for a, v in relations)


def safe_property(expr):
    expr = str(expr or "").strip()
    if not expr or len(expr) > 200 or "__" in expr or "import" in expr or "lambda" in expr:
        return None
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return expr


class ContrastiveClarifier(ClarificationAlgorithmBase):
    DEFAULT_CONFIG = {
        "drafts": 3,  # independent drafts run on stated examples and probes before asking
        "finals": 3,  # fresh implementations conditioned on the answer
        "probes": 8,  # LLM-written probe inputs, filtered by an LLM-written precondition
        "repairs": 2,  # repairs kept only if they gain checks without losing a hard one
        "review": True,  # answer-compliance review, kept under the same rule
        "call_adapter": True,  # retry failing calls with adjusted arguments
        "name_aliases": True,  # lower/snake_case spellings of the entry point
        "clean_contradictions": True,  # delete injected contradicting prose before drafting
    }

    # Model calls ----------------------------------------------------------------------

    def complete(self, env, messages):
        if env.prompt_cost >= env.prompt_budget * 0.85:
            return ""
        try:
            return env.llm(messages) or ""
        except Exception:
            return ""

    def parallel(self, env, batches):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(batches))) as pool:
            return list(pool.map(lambda messages: self.complete(env, messages), batches))

    @staticmethod
    def code_messages(user_content):
        return [{"role": "system", "content": GENERATE + ROBUST}, {"role": "user", "content": user_content}]

    def to_code(self, env, response, messages, entry, declared):
        code = extract_code(response, entry, declared)
        if code is None and response:
            retry = self.complete(env, messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Return syntactically valid Python defining " + entry + ". Return code only."},
            ])
            code = extract_code(retry, entry, declared)
        return sanitize(code) if code else None

    # Execution ----------------------------------------------------------------------------

    def finish(self, code, entry):
        """Append the call adapter and builtin-safe aliases to code that defines the entry point."""
        if not re.search(r"def\s+" + re.escape(entry) + r"\s*\(", code):
            return code
        if self.config["call_adapter"]:
            code = code.rstrip() + CALL_ADAPTER.replace("{entry}", entry)
        aliases = name_variants(entry) if self.config["name_aliases"] else []
        if aliases:
            code = code.rstrip() + ALIAS.format(aliases=aliases, entry=entry)
        return code

    def execute(self, env, codes, entry, checks, probe_code, props=()):
        """Run finished candidates on checks, probe inputs and properties in one sandbox call."""
        if not codes:
            return None
        inputs = {"PROBE_CODE": probe_code or "", "CANDIDATES": [self.finish(c, entry) for c in codes], "ENTRY": entry,
                  "CHECKS": [(tuple(a), e) for a, e in checks], "PROPS": list(props)}
        script = "".join(f"{name} = {value!r}\n" for name, value in inputs.items()) + HARNESS
        try:
            reports = re.findall(r"@@HARNESS@@\s*([^\n]*)", env.exec_code(script) or "")
            result = json.loads(reports[-1])
        except Exception:
            return None
        return result if isinstance(result, dict) and len(result.get("rows", [])) == len(codes) else None

    @staticmethod
    def passed(row, indices):
        return sum(1 for i in indices if i < len(row["checks"]) and row["checks"][i][0])

    @staticmethod
    def support(rows, i):
        def sig(row, k):
            return tuple(p[1] if p[0] == "ok" else (k, p[0], p[1]) for p in row["probes"])
        if rows[i]["load"]:
            return 0
        return sum(1 for k, r in enumerate(rows) if not r["load"] and sig(r, k) == sig(rows[i], i))

    def rank(self, rows, hard, soft):
        def key(i):
            row = rows[i]
            if row["load"]:
                return (-1, -1, -1, -1, -i)
            return (self.passed(row, hard), self.passed(row, soft), row.get("props", 0), self.support(rows, i), -i)
        return max(range(len(rows)), key=key)

    # Question -------------------------------------------------------------------------------

    @staticmethod
    def table(result, entry):
        inputs = (result or {}).get("inputs", [])
        rows = [r for r in (result or {}).get("rows", []) if not r["load"]]
        lines = []
        for k, source in enumerate(inputs):
            outs = [r["probes"][k][2] if r["probes"][k][0] == "ok" else "raises " + r["probes"][k][1]
                    for r in rows if k < len(r["probes"])]
            mark = " (drafts disagree)" if len(set(outs)) > 1 else ""
            lines.append(f"{k + 1}. {call_text(entry, source)} -> " + " | ".join(outs) + mark)
        return "\n".join(lines) or "(no probe inputs)"

    def contract_question(self, env, prompt, entry, anchor, result, anchor_row):
        """One fixed-form question; returns (question, example_check or None)."""
        inputs = (result or {}).get("inputs", [])
        user = ("Requirement:\n" + prompt + "\n\nDraft:\n```python\n" + anchor + "\n```\n\nProbe results "
                "(one column per draft, the first column is this draft):\n" + self.table(result, entry))
        fields = None
        for _ in range(2):
            reply = self.complete(env, [{"role": "system", "content": CONTRACT.replace("{entry}", entry)},
                                        {"role": "user", "content": user}])
            match = re.search(r"\{.*\}", reply or "", re.S)
            try:
                fields = json.loads(match.group(0)) if match else None
            except ValueError:
                fields = None
            if isinstance(fields, dict) and fields.get("returns") and fields.get("rule"):
                break
            fields = None

        conflict = (fields or {}).get("conflict")
        if isinstance(conflict, dict) and quoted_in(prompt, str(conflict.get("a", ""))) and quoted_in(prompt, str(conflict.get("b", ""))):
            a, b = phrase(conflict["a"], 160), phrase(conflict["b"], 160)
            return f'About {entry}: the requirement says "{a}" but also "{b}". Which of the two is intended?', None

        # An executed example from the anchor draft (a real output, not a claim).
        probes = anchor_row["probes"] if anchor_row and not anchor_row["load"] else []
        order = []
        try:
            wanted = int((fields or {}).get("probe", 0)) - 1
            if 0 <= wanted < len(probes):
                order.append(wanted)
        except (TypeError, ValueError):
            pass
        order += [k for k in range(len(probes)) if k not in order]
        example = None
        for k in order:
            if k < len(inputs) and probes[k][0] == "ok" and len(probes[k][2]) <= 120:
                example = (k, call_text(entry, inputs[k]), probes[k][2])
                break

        if fields:
            parts = f"About {entry}: my draft takes {phrase(fields.get('arguments'))} and returns {phrase(fields.get('returns'))}; {phrase(fields.get('rule'), 300)}."
            if example:
                parts += f" For example, {example[1]} returns {example[2]}."
            # A generic closing: a specific alternative pulls the single answer onto that
            # point instead of the draft's biggest deviation.
            question = parts + " Is that what you intend, or should it behave differently or return a different format?"
        else:
            reply = self.complete(env, [{"role": "system", "content": FALLBACK_ASK},
                                        {"role": "user", "content": "Requirement:\n" + prompt + "\n\nDraft:\n```python\n" + anchor + "\n```"}])
            question = clean_question(reply)
            if question and entry not in re.findall(r"\w+", question):
                question = f"About {entry}: " + question
            example = None
        check = None
        if example:
            try:
                args = ast.literal_eval(inputs[example[0]])
                check = (args if isinstance(args, tuple) else (args,), ast.literal_eval(example[2]))
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                check = None
        return clean_question(question), check

    # Main ---------------------------------------------------------------------------------

    def run(self, env, problem):
        # Deliberately read only public prompt/entry point, never task IDs or oracles.
        prompt, entry = problem["prompt"], problem["entry_point"]
        declared = declared_name(prompt, entry)
        if self.config["clean_contradictions"]:
            # The simulated user still sees the original text; only our own steps use the cleaned one.
            reply = self.complete(env, [{"role": "user", "content": CLEAN.replace("{prompt}", prompt)}])
            prompt = accept_deletion(prompt, edited_text(reply), [entry, declared, "candidate"]) or prompt
        public = json.dumps({"prompt": prompt, "entry_point": entry}, ensure_ascii=False)
        fallback = "def " + entry + "(*args, **kwargs):\n    return None"
        examples = parse_examples(prompt, [entry] + ([declared] if declared else []))
        stated = list(range(len(examples)))

        # 1. Drafts and probe inputs, then run them on the stated examples and probes.
        draft_messages = self.code_messages(public)
        probe_system = PROBE.format(num_probes=int(self.config["probes"]), entry_point=entry)
        replies = self.parallel(env, [draft_messages] * int(self.config["drafts"])
                                + [[{"role": "system", "content": probe_system}, {"role": "user", "content": prompt}]])
        drafts = [c for c in (self.to_code(env, r, draft_messages, entry, declared) for r in replies[:-1]) if c]
        probe_code = extract_fenced_code(replies[-1])
        result = self.execute(env, drafts, entry, examples, probe_code) if drafts else None
        anchor_index = self.rank(result["rows"], [], stated) if result else 0
        anchor = drafts[anchor_index] if drafts else fallback
        if result and anchor_index:
            # Put the anchor first so that its column leads the probe table.
            order = [anchor_index] + [i for i in range(len(drafts)) if i != anchor_index]
            result = {"inputs": result["inputs"], "rows": [result["rows"][i] for i in order]}
        anchor_row = result["rows"][0] if result else None

        # 2. One question.
        question = answer = None
        example_check = None
        if env.can_ask():
            asked, example_check = self.contract_question(env, prompt, entry, anchor, result, anchor_row)
            if asked:
                try:
                    answer = env.ask_human(asked)
                    question = asked
                except TooManyQuestionException:
                    answer = None
                except Exception:
                    answer = None

        if not answer:
            code = anchor
            if result and len(drafts) > 1:
                code = [anchor] + [d for i, d in enumerate(drafts) if i != anchor_index]
                code = code[self.rank(result["rows"], stated, [])]
            return "```python\n" + self.finish(code, entry) + "\n```"

        # 3. Finals, compiled answer and fresh probes, in parallel.
        final_messages = self.code_messages(FINAL.format(prompt=prompt, entry=entry, question=question, answer=answer, obligations=""))
        numbered = "\n".join(f"{i + 1}. {call_text(entry, repr(tuple(a)))} -> {e!r}" for i, (a, e) in enumerate(examples)) or "(none)"
        compile_messages = [{"role": "system", "content": COMPILE},
                            {"role": "user", "content": f"Requirement:\n{prompt}\n\nQuestion: {question}\nAnswer: {answer}\n\nStated examples:\n{numbered}"}]
        probe_messages = [{"role": "system", "content": probe_system},
                          {"role": "user", "content": prompt + "\n\nThe user was asked: " + question + "\nThe user answered: " + answer}]
        replies = self.parallel(env, [final_messages] * int(self.config["finals"]) + [compile_messages, probe_messages])
        finals = [c for c in (self.to_code(env, r, final_messages, entry, declared) for r in replies[:-2]) if c] or [anchor]
        values = read_block_values(replies[-2], {"CONFIRMS", "CALLS", "PROPERTIES", "EXAMPLES"})
        probe_code = extract_fenced_code(replies[-1]) or probe_code

        # Binding (hard) checks: the confirmed example and calls the answer itself writes out.
        # Everything else is provisional (soft): compilers copy stated examples into CALLS and
        # label corrupted examples CONFIRMED, so those must not overrule the answer.
        hard, soft = [], []
        relations = answer_relations(answer, [entry] + ([declared] if declared else []))
        if example_check and values.get("CONFIRMS") is True:
            hard.append(example_check)
        for item in values.get("CALLS") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                args = item[0] if isinstance(item[0], tuple) else (tuple(item[0]) if isinstance(item[0], list) else (item[0],))
                if repr((args, item[1])) not in {repr(c) for c in hard + soft}:
                    (hard if stated_in(relations, args, item[1]) else soft).append((args, item[1]))
        verdicts = values.get("EXAMPLES") if isinstance(values.get("EXAMPLES"), list) else []
        for i, example in enumerate(examples):
            verdict = str(verdicts[i]).upper() if i < len(verdicts) else "KEEP"
            if verdict.startswith(("CONFIRMED", "KEEP")) and repr(example) not in {repr(c) for c in hard + soft}:
                (hard if verdict.startswith("CONFIRMED") and stated_in(relations, *example) else soft).append(example)
        props = [p for p in (safe_property(x) for x in (values.get("PROPERTIES") or [])[:3]) if p]
        checks = (hard + soft)[:12]
        hard_idx = list(range(min(len(hard), len(checks))))
        soft_idx = list(range(len(hard_idx), len(checks)))

        # 4. Select; one rescue candidate if no final meets every hard check.
        result = self.execute(env, finals, entry, checks, probe_code, props)
        if result and hard_idx and not any(not r["load"] and self.passed(r, hard_idx) == len(hard_idx) for r in result["rows"]):
            lines = "\n".join(f"- {call_text(entry, repr(tuple(checks[i][0])))} == {checks[i][1]!r}" for i in hard_idx)
            rescue_messages = self.code_messages(FINAL.format(prompt=prompt, entry=entry, question=question, answer=answer,
                                                              obligations=OBLIGATIONS.format(lines=lines)))
            rescue = self.to_code(env, self.complete(env, rescue_messages), rescue_messages, entry, declared)
            if rescue:
                finals = finals + [rescue]
                result = self.execute(env, finals, entry, checks, probe_code, props) or result
        if result:
            # Every final was written after the answer: a provisional check none of them
            # satisfies is one the answer overrides (or a corrupted example); drop it.
            live = [r for r in result["rows"] if not r["load"]]
            keep = [i for i in range(len(checks)) if i in hard_idx or any(i < len(r["checks"]) and r["checks"][i][0] for r in live)]
            if len(keep) < len(checks):
                checks = [checks[i] for i in keep]
                for r in live:
                    r["checks"] = [r["checks"][i] for i in keep]
                soft_idx = list(range(len(hard_idx), len(checks)))
        code, status = finals[0], None
        if result:
            chosen = self.rank(result["rows"], hard_idx, soft_idx)
            code, status = finals[chosen], result["rows"][chosen]["checks"]

        def keeps_hard(new):
            return all(new[i][0] for i in hard_idx if i < len(status) and status[i][0])

        # 5. Repairs: kept only if they gain checks without losing a satisfied hard check.
        for _ in range(int(self.config["repairs"]) if checks and status else 0):
            failing = [i for i in range(len(checks)) if i < len(status) and not status[i][0]]
            if not failing:
                break
            failures = "\n".join(f"- {call_text(entry, repr(tuple(checks[i][0])))} should return {checks[i][1]!r}, got {status[i][1]}"
                                 for i in failing[:4])
            messages = final_messages + [{"role": "assistant", "content": "```python\n" + code + "\n```"},
                                         {"role": "user", "content": REPAIR.format(failures=failures)}]
            repaired = self.to_code(env, self.complete(env, messages), messages, entry, declared)
            if not repaired:
                break
            trial = self.execute(env, [repaired], entry, checks, None)
            if not trial or trial["rows"][0]["load"]:
                break
            new = trial["rows"][0]["checks"]
            if not keeps_hard(new) or sum(c[0] for c in new) <= sum(c[0] for c in status):
                break
            code, status = repaired, new

        # 6. Answer-compliance review under the same rule; without checks a cited contradiction is required.
        if self.config["review"]:
            reply = self.complete(env, [{"role": "user", "content": REVIEW.format(prompt=prompt, question=question, answer=answer, entry=entry, code=code)}])
            if reply and not reply.strip().upper().startswith("OK"):
                fixed = extract_code(reply, entry, declared)
                if fixed:
                    fixed = sanitize(fixed)
                    if checks and status:
                        trial = self.execute(env, [fixed], entry, checks, None)
                        if trial and not trial["rows"][0]["load"]:
                            new = trial["rows"][0]["checks"]
                            if keeps_hard(new) and sum(c[0] for c in new) >= sum(c[0] for c in status):
                                code = fixed
                    elif "CONTRADICTION:" in reply.upper():
                        code = fixed
        return "```python\n" + self.finish(code, entry) + "\n```"
