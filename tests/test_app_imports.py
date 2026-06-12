"""Import-thread regression guard for the frame-loop perf fix.

Measured on this hardware (M3 Ultra, macOS 25.5): when mlx + the model
libraries are first imported on the MAIN thread and the engine is then
constructed on its worker thread, engine.step pays a ~80ms/frame penalty
(~183ms vs ~104ms co-resident; profile shows the extra time in malloc/free
churn inside mlx eval). Importing the MLX stack ON the engine worker first
removes the penalty entirely. QoS, MLX stream ids, and executor overhead were
all ruled out by measurement (see docs/decisions/0002, known limitation 6).

Therefore moneypenny.app must NOT import any mlx-touching module at module
level: the engine worker has to be the first thread in the process to import
mlx. This test pins that property; if someone re-adds a top-level
`from moneypenny.engine import ...` (or asr/router), it fails.
"""
import subprocess
import sys

FORBIDDEN_PREFIXES = ("mlx", "rustymimi", "mlx_lm", "mlx_audio", "parakeet_mlx", "personaplex_mlx")

_PROBE = """
import sys
import %%s
bad = sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in %r})
print("LOADED:" + ",".join(bad))
""" % (FORBIDDEN_PREFIXES,)


def _assert_no_mlx_on_import(module: str) -> None:
    # Subprocess: a clean interpreter is the only honest sys.modules check.
    out = subprocess.run(
        [sys.executable, "-c", _PROBE % module],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    loaded = out.stdout.strip().removeprefix("LOADED:")
    assert loaded == "", (
        f"{module} transitively imported {loaded} at module level; "
        "model modules must be imported on the engine worker (perf fix, "
        "docs/decisions/0002 known limitation 6)"
    )


def test_importing_app_does_not_import_mlx_stack():
    _assert_no_mlx_on_import("moneypenny.app")


def test_importing_web_does_not_import_mlx_stack():
    # The web server imports moneypenny.app (for Session) and must inherit
    # the same guarantee: moneypenny-web loads models on the engine worker.
    _assert_no_mlx_on_import("moneypenny.web")
