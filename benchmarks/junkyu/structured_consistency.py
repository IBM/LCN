"""Deprecated shim: use ``benchmarks/check_consistency.py`` instead.

The structure-exploiting consistency checker is now a general tool. Its core is
the library function
:func:`lcn.inference.utils.structured_consistency.check_consistency_structured`,
and the standalone CLI is ``benchmarks/check_consistency.py`` (applies to any
LCN under ``benchmarks/``, not just this directory).

This module remains as a thin re-export so existing imports keep working.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Re-export the library API.
from lcn.inference.utils.structured_consistency import (  # noqa: F401,E402
    check_consistency_structured,
    is_consistent_structured,
    StructuredConsistencyResult,
    CONSISTENT, INCONSISTENT, UNDETERMINED, ERROR,
)

if __name__ == "__main__":
    # Delegate to the general CLI.
    from benchmarks.check_consistency import main
    raise SystemExit(main())
