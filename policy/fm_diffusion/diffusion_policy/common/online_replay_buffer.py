from __future__ import annotations

import torch


class OnlineReplayBuffer:
    """Fixed-capacity tensor ring buffer for ResFiT online transitions."""

    _FIELD_DIMS = {
        "obs_feature": "feature_dim",
        "base_action": "action_dim",
        "action": "action_dim",
        "reward": 1,
        "next_obs_feature": "feature_dim",
        "next_base_action": "action_dim",
        "terminated": 1,
    }

    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        action_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if capacity <= 0 or feature_dim <= 0 or action_dim <= 0:
            raise ValueError("capacity, feature_dim, and action_dim must be positive.")
        if not dtype.is_floating_point:
            raise ValueError("OnlineReplayBuffer requires a floating-point dtype.")

        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self._position = 0
        self._size = 0
        self._storage = {
            name: torch.empty(
                (self.capacity, self._resolve_dim(dim)),
                device=self.device,
                dtype=self.dtype,
            )
            for name, dim in self._FIELD_DIMS.items()
        }

    def __len__(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._position

    def add_batch(self, **transition: torch.Tensor) -> None:
        missing = self._storage.keys() - transition.keys()
        extra = transition.keys() - self._storage.keys()
        if missing or extra:
            raise KeyError(
                f"Transition fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}."
            )

        prepared = {
            name: self._prepare_field(name, value)
            for name, value in transition.items()
        }
        batch_sizes = {value.shape[0] for value in prepared.values()}
        if len(batch_sizes) != 1:
            raise ValueError("All transition fields must have the same batch size.")
        batch_size = batch_sizes.pop()
        if batch_size == 0:
            return

        if batch_size >= self.capacity:
            prepared = {
                name: value[-self.capacity:]
                for name, value in prepared.items()
            }
            batch_size = self.capacity

        first_count = min(batch_size, self.capacity - self._position)
        second_count = batch_size - first_count
        for name, value in prepared.items():
            self._storage[name][self._position:self._position + first_count].copy_(
                value[:first_count]
            )
            if second_count:
                self._storage[name][:second_count].copy_(value[first_count:])

        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self.capacity, self._size + batch_size)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = torch.randint(
            self._size,
            (int(batch_size),),
            device=self.device,
            generator=generator,
        )
        destination = self.device if device is None else torch.device(device)
        return {
            name: values.index_select(0, indices).to(destination)
            for name, values in self._storage.items()
        }

    def _prepare_field(self, name: str, value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        expected_dim = self._storage[name].shape[1]
        if value.ndim == 1 and expected_dim == 1:
            value = value.unsqueeze(-1)
        if value.ndim != 2 or value.shape[1] != expected_dim:
            raise ValueError(
                f"Field `{name}` must have shape [B, {expected_dim}], got "
                f"{tuple(value.shape)}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Field `{name}` contains NaN or Inf values.")
        return value

    def _resolve_dim(self, dim: int | str) -> int:
        return int(getattr(self, dim)) if isinstance(dim, str) else int(dim)
