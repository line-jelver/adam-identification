"""Optional CLI hook for a one-line retrieval status.

Library callers leave the hook unset, so identification stays quiet.
The ``adam-identify`` CLI binds a callback that updates its spinner.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token

from adam_identification.models import MaterialSource

_STATUS: ContextVar[Callable[[str], None] | None] = ContextVar("retrieval_status", default=None)

_SOURCE_MESSAGE = {
    MaterialSource.MC3D: "Searching MC3D…",
    MaterialSource.MATERIALS_PROJECT: "Searching Materials Project…",
    MaterialSource.PUBCHEM: "Searching PubChem…",
}


def bind_retrieval_status(callback: Callable[[str], None]) -> Token[Callable[[str], None] | None]:
    """Install ``callback`` for the current context. Reset the returned token."""
    return _STATUS.set(callback)


def reset_retrieval_status(token: Token[Callable[[str], None] | None]) -> None:
    """Remove the callback installed by :func:`bind_retrieval_status`."""
    _STATUS.reset(token)


def announce_search(source: MaterialSource) -> None:
    """Tell the bound callback that a search of ``source`` is starting."""
    callback = _STATUS.get()
    if callback is None:
        return
    callback(_SOURCE_MESSAGE.get(source, f"Searching {source.value}…"))
