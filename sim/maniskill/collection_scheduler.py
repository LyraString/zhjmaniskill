"""Process-safe scheduling for multi-GPU ManiSkill data collection."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")


class SharedCollectionScheduler:
    """Assign unique seeds and successful episode indices across workers."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)

    @classmethod
    def initialize(
        cls,
        state_path: str | Path,
        *,
        seed: int,
        episode_index_offset: int,
        episode_count: int,
        max_attempts: int | None,
    ) -> "SharedCollectionScheduler":
        scheduler = cls(state_path)
        scheduler.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "next_seed": int(seed),
            "attempt_count": 0,
            "max_attempts": None if max_attempts is None else int(max_attempts),
            "next_episode_index": int(episode_index_offset),
            "end_episode_index": int(episode_index_offset + episode_count),
        }
        scheduler.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return scheduler

    @contextmanager
    def _locked_state(self, *, write: bool) -> Iterator[dict]:
        with self.state_path.open("r+", encoding="utf-8") as state_file:
            fcntl.flock(
                state_file.fileno(),
                fcntl.LOCK_EX if write else fcntl.LOCK_SH,
            )
            try:
                state_file.seek(0)
                state = json.load(state_file)
                yield state
                if write:
                    state_file.seek(0)
                    json.dump(state, state_file, ensure_ascii=False, indent=2)
                    state_file.truncate()
                    state_file.flush()
                    os.fsync(state_file.fileno())
            finally:
                fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)

    def claim_seed(self) -> int | None:
        """Return the next unclaimed seed, or None when collection should stop."""
        with self._locked_state(write=True) as state:
            if state["next_episode_index"] >= state["end_episode_index"]:
                return None
            max_attempts = state["max_attempts"]
            if max_attempts is not None and state["attempt_count"] >= max_attempts:
                return None
            seed = int(state["next_seed"])
            state["next_seed"] = seed + 1
            state["attempt_count"] += 1
            return seed

    def save_next_episode(self, save_fn: Callable[[int], T]) -> tuple[int, T] | None:
        """Atomically assign and save the next successful episode."""
        with self._locked_state(write=True) as state:
            episode_idx = int(state["next_episode_index"])
            if episode_idx >= int(state["end_episode_index"]):
                return None
            result = save_fn(episode_idx)
            state["next_episode_index"] = episode_idx + 1
            return episode_idx, result

    def snapshot(self) -> dict:
        with self._locked_state(write=False) as state:
            return dict(state)
