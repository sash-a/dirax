from typing import ClassVar, Protocol, TypedDict


class Behaviour(TypedDict, total=False):
    """The Ray options a Component's own correctness rests on.

    Closed, and declared by the class rather than reachable from a config file, because the
    options it holds are load-bearing where they are set at all. See ADR-0011.
    """

    max_concurrency: int
    max_restarts: int
    max_task_retries: int
    allow_out_of_order_execution: bool
    max_pending_calls: int


class Component(Protocol):
    """A class Ray places as its own process.

    One member, so ``type[Component]`` rejects a class that never decided what its correctness
    requires.
    """

    RAY_OPTIONS: ClassVar[Behaviour]
