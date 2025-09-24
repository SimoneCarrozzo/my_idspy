from typing import Optional

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from ...core.step import ContextualStep, Step
from ...core.state import State


class TorchProfiler(ContextualStep):
    """Wrap a step inside a torch.profiler profile that writes TensorBoard traces."""

    def __init__(
        self,
        step: Step,
        log_dir: str,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        in_scope: Optional[str] = None,
        out_scope: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(step=step, in_scope=in_scope, out_scope=out_scope, name=name)

        # store config
        self.log_dir = log_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops

    def context(self, state: State) -> Optional[any]:
        # Decide activities
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        elif torch.backends.mps.is_available():
            acts.append(ProfilerActivity.MPS)

        # Standard TB schedule: wait -> warmup -> active (repeat)
        sched = schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )

        profiler = profile(
            activities=acts,
            schedule=sched,
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )

        return profiler
