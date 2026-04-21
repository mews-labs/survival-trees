"""Phase-1 test-only shim.

``survival_trees/__init__.py`` still imports the legacy R backend
(``_base.py``), which does ``import rpy2.robjects as ro`` at module
load. rpy2 in turn dlopens ``libR.so`` during import. Machines without
an R install cannot import the package at all, even to reach the new
Rust backend.

In Phase 1 the R backend is NOT under test; only the Rust backend is.
We replace ``rpy2.robjects`` and its friends with stub modules in
``sys.modules`` before collection, so the legacy top-level imports
succeed without touching any R runtime. Any attempt to actually USE
the R backend would still fail loudly — as intended.
"""

from __future__ import annotations

import sys
import types


def _install_rpy2_stubs() -> None:
    if "rpy2.robjects" in sys.modules and not isinstance(
        sys.modules["rpy2.robjects"], _StubModule
    ):
        return  # real rpy2 already imported successfully; leave it alone

    class _StubR:
        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                "R runtime is not available in this environment; "
                "the Rust backend should be used instead."
            )

    rpy2_pkg = _StubModule("rpy2")
    rinterface = _StubModule("rpy2.rinterface")
    rinterface_lib = _StubModule("rpy2.rinterface_lib")
    rinterface_lib_embedded = _StubModule("rpy2.rinterface_lib.embedded")
    rinterface_lib_embedded.RRuntimeError = RuntimeError

    robjects = _StubModule("rpy2.robjects")
    robjects.r = _StubR()
    robjects.globalenv = {}
    robjects.default_converter = object()

    packages = _StubModule("rpy2.robjects.packages")

    def _importr(*_args, **_kwargs):
        return object()

    packages.importr = _importr

    pandas2ri = _StubModule("rpy2.robjects.pandas2ri")
    pandas2ri.converter = object()

    conversion = _StubModule("rpy2.robjects.conversion")

    class _LocalConverter:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    conversion.localconverter = _LocalConverter
    conversion.py2rpy = lambda x: x
    conversion.rpy2py = lambda x: x

    robjects.conversion = conversion
    robjects.pandas2ri = pandas2ri

    sys.modules["rpy2"] = rpy2_pkg
    sys.modules["rpy2.robjects"] = robjects
    sys.modules["rpy2.robjects.packages"] = packages
    sys.modules["rpy2.robjects.pandas2ri"] = pandas2ri
    sys.modules["rpy2.robjects.conversion"] = conversion
    sys.modules["rpy2.rinterface"] = rinterface
    sys.modules["rpy2.rinterface_lib"] = rinterface_lib
    sys.modules["rpy2.rinterface_lib.embedded"] = rinterface_lib_embedded


class _StubModule(types.ModuleType):
    pass


try:
    import rpy2.robjects  # noqa: F401
except Exception:
    _install_rpy2_stubs()
