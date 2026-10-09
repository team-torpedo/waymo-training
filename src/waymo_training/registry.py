"""Tiny name -> factory registry used to make components swappable from YAML."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(obj: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} is already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError:
            options = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.kind} {name!r}. Available: {options}") from None

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
