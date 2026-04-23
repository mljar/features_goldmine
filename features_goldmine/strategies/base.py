from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    name: str

    @abstractmethod
    def run(self, X, y, task: str, random_state: int):
        raise NotImplementedError
