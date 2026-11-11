"""Compatibility layer for environments where Numba C-extensions are blocked by security policies."""
import sys
import types


def ensure_numba_compat():
    """Ensure numba can be imported or shinned if blocked by OS security policy."""
    if "numba" in sys.modules:
        return
    try:
        import numba  # noqa: F401
    except Exception:
        def dummy_jit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

        n = types.ModuleType("numba")
        n.njit = dummy_jit
        n.jit = dummy_jit
        n.__version__ = "0.68.0"

        nt = types.ModuleType("numba.typed")
        nt.List = list
        nt.Dict = dict

        sys.modules["numba"] = n
        sys.modules["numba.typed"] = nt


ensure_numba_compat()
