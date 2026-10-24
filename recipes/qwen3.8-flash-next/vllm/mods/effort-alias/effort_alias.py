#!/usr/bin/env python3
"""effort-alias: maps OpenAI reasoning_effort values to Qwen3.8 levels.

Per the OpenAI API reference, reasoning_effort accepts:
    none, minimal, low, medium, high, xhigh, max
(https://developers.openai.com/api/reference — "Constrains effort on reasoning")

Qwen3.8 exposes only three native levels: low, medium, xhigh (xhigh is the
default). This script translates the full OpenAI ladder to those three:

    sent       native     handled by
    ---------  ---------  ------------------------------
    none       low        this script (approximation, see below)
    minimal    low        this script (rounded up to floor)
    low        low        native
    medium     medium     native
    high       xhigh      this script (rounded up)
    xhigh      xhigh      native (default)
    max        xhigh      this script (ceiling)

Note on 'none': OpenAI defines 'none' as "no effort — no reasoning". Qwen3.8
has no native 'none' level on the reasoning_effort ladder; the quietest
native level is 'low'. This script maps 'none' -> 'low' as the closest
available approximation. For true "no reasoning", callers must use
enable_thinking=false, which is orthogonal to reasoning_effort and outside
this script's scope.

Note on invalid input: any value that is not a string, or is a string not
in the ladder above (including '' and 'null'), is passed through as-is to
the template, which rejects it with HTTP 400. Non-string types (null, int,
bool, list, dict) are normalized to none before reaching .get() so that
unhashable values raise 400 rather than 500.

The patch applies only if the template still carries the expected validation
line. If a future model accepts more levels natively, that line disappears
and this script attempts to write the original template unchanged; it exits
0 only if that write succeeds.

Contract: after a successful run (exit 0), /tmp/chat_template.jinja reflects
the resolved snapshot's template, patched or not. If the snapshot directory
is not provided, the template cannot be read, or the output cannot be
written, the script exits non-zero so sparkrun's fail-fast pre_exec aborts
the launch before vLLM starts.

Template resolution: the recipe exports
``VLLM_PLE_MMAP_DIR: "{launch.model_path}"``, and sparkrun resolves the
placeholder to the exact snapshot directory inside the container. The
template is read from ``<dir>/chat_template.jinja``. This path does not
depend on where the HuggingFace cache is mounted, so the script works under
any HF_HOME layout.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# VLLM_PLE_MMAP_DIR is the recipe's PLE table directory. In this recipe its
# value is "{launch.model_path}", the resolved snapshot root, so it also
# contains chat_template.jinja. Reusing it avoids a second env var that would
# duplicate the same path.
MODEL_PATH_ENV = "VLLM_PLE_MMAP_DIR"
OUT = Path("/tmp/chat_template.jinja")

OLD = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
NEW = (
    "{%- set _effort = reasoning_effort if reasoning_effort is string "
    "else ('xhigh' if reasoning_effort is undefined else none) -%}"
    "{%- set resolved_reasoning_effort = "
    "{'high': 'xhigh', 'max': 'xhigh', 'minimal': 'low', 'none': 'low'}"
    ".get(_effort, _effort) %}"
)
GUARD = "Supported types are xhigh (default), medium, and low."


def warn_skip(msg: str) -> None:
    """Graceful degradation: vLLM will start with the unpatched template."""
    print(f"EFFORT_ALIAS: {msg}; serving as-is", file=sys.stderr)


def warn_fail(msg: str) -> None:
    """Fatal: sparkrun's pre_exec will abort the launch."""
    print(f"EFFORT_ALIAS: {msg}; aborting launch", file=sys.stderr)


def write_out(text: str) -> bool:
    """Atomic write; returns False on OSError."""
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".chat_template.", dir=OUT.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, OUT)
        return True
    except OSError as e:
        warn_fail(f"cannot write {OUT} ({e})")
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def resolve_template() -> Path | None:
    """Return the source chat_template.jinja, or None with a fatal message.

    The source is ``<VLLM_PLE_MMAP_DIR>/chat_template.jinja``. The variable
    must be set by the recipe's env: block and resolved by sparkrun; a
    missing, empty, or unresolved value is a configuration error, not a
    runtime race, so it is reported and the launch is aborted.
    """
    model_path = os.environ.get(MODEL_PATH_ENV, "").strip()

    if not model_path:
        warn_fail(f"{MODEL_PATH_ENV} is not set")
        return None
    if "{launch.model_path}" in model_path:
        warn_fail(f"{MODEL_PATH_ENV} contains an unresolved placeholder: {model_path!r}")
        return None

    candidate = Path(model_path) / "chat_template.jinja"
    if not candidate.is_file():
        warn_fail(f"chat_template.jinja not found at {candidate}")
        return None
    return candidate


def jinja_ok(text: str) -> bool:
    try:
        from jinja2 import Environment
    except ImportError:
        return True
    try:
        Environment().parse(text)
        return True
    except Exception:
        return False


def main() -> int:
    # Never let a stale file from a previous boot mask this run's failure.
    try:
        OUT.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        warn_fail(f"cannot remove stale {OUT} ({e})")
        return 1

    src = resolve_template()
    if src is None:
        return 1

    try:
        text = src.read_text(encoding="utf-8")
    except OSError as e:
        warn_fail(f"cannot read {src} ({e})")
        return 1

    if GUARD not in text:
        if not write_out(text):
            return 1
        warn_skip("template already accepts other efforts")
        return 0

    count = text.count(OLD)
    if count != 1:
        if not write_out(text):
            return 1
        warn_skip(f"OLD line not found exactly once (found {count})")
        return 0

    patched = text.replace(OLD, NEW, 1)
    if not jinja_ok(patched):
        if not write_out(text):
            return 1
        warn_skip("patched template failed jinja parse")
        return 0

    if not write_out(patched):
        return 1
    print(f"EFFORT_ALIAS applied to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())