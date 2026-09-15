"""Vendor-neutral durable-archive prototype for AAA-134.

**PROTOTYPE_ONLY_NOT_DURABLE.** Nothing in this package is a resolution of
AAA-134. The bundled backend writes to the local filesystem, which is exactly
the thing the protocol already refuses to accept as durable evidence. The
package exists to pin down a backend contract and to prove that the contract's
failure modes are detectable.
"""

from __future__ import annotations

PROTOTYPE_LABEL = "PROTOTYPE_ONLY_NOT_DURABLE"

__all__ = ["PROTOTYPE_LABEL"]
