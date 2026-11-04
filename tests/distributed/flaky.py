"""Actors the monitor tests place, in their own module because a worker imports it.

A worker's environment carries dirax's runtime dependencies and nothing else, so this imports
nothing of the suite.
"""

import os
import time
import uuid
from pathlib import Path

POLL = 0.01


class Boom(Exception):
    """What a run raises when a test wants an actor to fail on purpose."""


class Flaky:
    """An actor that fails its first ``fail_first`` runs, then runs until it is closed.

    A restarted actor is a new process with no memory, so how many times ``run`` has been
    started is kept in ``ledger``, which outlives the process. Failing is either raising, which
    leaves the process up, or ``crash``, which takes it down.
    """

    def __init__(self, ledger: str, fail_first: int = 0, crash: bool = False, fail_on_stop: bool = False):
        self.ledger = Path(ledger)
        self.fail_first = fail_first
        self.crash = crash
        self.fail_on_stop = fail_on_stop
        self.stop = False
        self.instance = uuid.uuid4().hex

    def ping(self) -> str:
        return self.instance

    def run(self) -> None:
        with self.ledger.open("a") as file:
            file.write("started\n")
        if len(self.ledger.read_text().splitlines()) <= self.fail_first:
            if self.crash:
                os._exit(1)
            raise Boom("while running")
        while not self.stop:
            time.sleep(POLL)
        if self.fail_on_stop:
            raise Boom("while stopping")

    def close(self) -> None:
        self.stop = True
