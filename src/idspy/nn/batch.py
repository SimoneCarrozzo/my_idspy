from dataclasses import dataclass
from typing import Mapping, Optional, Union, Any, Dict
import torch
from torch import Tensor


Features = Union[Tensor, Mapping[str, Tensor]]


def _map_tensors(x: Any, fn) -> Any:
    """
    Apply a function to all tensors in a nested structure.
    """
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, Mapping):
        return {k: _map_tensors(v, fn) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_map_tensors(v, fn) for v in x)
    return x


@dataclass(frozen=True)
class Batch:
    """
    Container for model features and targets.
    """

    features: Features
    target: Optional[Tensor] = None

    def to(self, device: torch.device, non_blocking: bool = True) -> "Batch":
        """
        Move all tensors in the batch to the specified device.
        """
        return Batch(
            features=_map_tensors(
                self.features, lambda t: t.to(device, non_blocking=non_blocking)
            ),
            target=(
                None
                if self.target is None
                else self.target.to(device, non_blocking=non_blocking)
            ),
        )

    def detach(self) -> "Batch":
        """
        Detach all tensors in the batch from the computation graph.
        """
        return Batch(
            features=_map_tensors(self.features, lambda t: t.detach()),
            target=None if self.target is None else self.target.detach(),
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the batch as a dictionary.
        """
        return {"features": self.features, "target": self.target}

    def __getitem__(self, key: str) -> Any:
        """
        Get an item from the batch by key ('features' or 'target').
        """
        if key == "features":
            return self.features
        if key == "target":
            return self.target
        raise KeyError(key)

    def __iter__(self):
        """
        Iterate over batch keys ('features', 'target').
        """
        yield "features"
        yield "target"


def ensure_batch(x: Batch | Mapping[str, Any]) -> Batch:
    """
    Ensure the input is a Batch instance.
    """
    if isinstance(x, Batch):
        return x

    return Batch(features=x["features"], target=x.get("target"))


def default_collate(samples: list[Mapping[str, Any]]) -> Batch:
    """
    Collate a list of supervised samples into a Batch.
    """
    first_features = samples[0]["features"]

    def stack(ts: list[Tensor]) -> Tensor:
        return torch.stack(ts, 0)

    if isinstance(first_features, Mapping):
        keys = list(first_features.keys())
        feature_lists = {k: [] for k in keys}
        for s in samples:
            for k in keys:
                feature_lists[k].append(s["features"][k])
        features = {k: stack(feature_lists[k]) for k in keys}
    else:
        feature_list = [s["features"] for s in samples]
        features = stack(feature_list)

    target = None
    if "target" in samples[0] and samples[0]["target"] is not None:
        target = stack([s["target"] for s in samples]).view(-1)

    return Batch(features=features, target=target)
