"""
Base adapter interface — all sources must implement this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from matchmind.schema.match_state import MatchState


class BaseAdapter(ABC):
    """
    Abstract base class for all data source adapters.

    Contract: every adapter must return MatchState objects.
    The agent never knows which adapter produced the data.
    """

    @abstractmethod
    def load_snapshot(self, *args, **kwargs) -> MatchState:
        """Load a single match snapshot and return as MatchState."""
        ...

    @abstractmethod
    def load_test_cases(self, n: int = 25) -> list[dict]:
        """
        Load N test cases for evaluation.
        Each dict has keys: 'snapshot', 'focus_player_id', 'question',
        'ground_truth_action', 'ground_truth_event_type'.
        """
        ...
