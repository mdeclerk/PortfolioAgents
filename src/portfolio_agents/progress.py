"""A Rich Progress stage tracker driven by the pipeline's two callbacks.

The pipeline announces phase boundaries via `stage(name)` and status lines via
`log(msg)`. Started phases appear in execution order: the current phase spins,
completed phases retain a ✔ (a failed one a ✖), and each phase shows elapsed time
and its latest detail. Rendering stays on stderr so stdout keeps the report path.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

# The pipeline's six observable phases, in order.
PHASES = ("connect", "fetch", "metrics", "positions", "portfolio", "report")


class StageTracker:
    """Drives native Rich progress tasks from the pipeline's log/stage callbacks."""

    def __init__(self, phases: tuple[str, ...] = PHASES, console: Console | None = None) -> None:
        self._phases = phases
        self._progress = Progress(
            SpinnerColumn(finished_text=""),
            TextColumn("{task.description}"),
            TextColumn("{task.elapsed:5.1f}s", style="progress.elapsed"),
            TextColumn("{task.fields[detail]}", style="dim"),
            console=console or Console(stderr=True),
            refresh_per_second=12,
            transient=False,
        )
        self._current: tuple[str, TaskID] | None = None

    def stage(self, name: str) -> None:
        """Mark the current phase done and start `name` spinning."""
        assert name in self._phases, f"unknown phase {name!r}, expected one of {self._phases}"
        self._finish("✔", "green")
        self._current = name, self._progress.add_task(name, total=1, detail="")

    def log(self, msg: str) -> None:
        """Attach the latest status line to the current phase as its detail text."""
        if self._current is None:
            self.stage(self._phases[0])
        assert self._current is not None
        self._progress.update(self._current[1], detail=msg)

    @contextmanager
    def live(self) -> Iterator[StageTracker]:
        """Render the stepper live to stderr for the duration of the block."""
        with self._progress:
            try:
                yield self
            except BaseException:
                self._finish("✖", "red")
                raise
            else:
                self._finish("✔", "green")

    def _finish(self, marker: str, style: str) -> None:
        if self._current is None:
            return
        name, task_id = self._current
        self._progress.stop_task(task_id)
        self._progress.update(
            task_id, completed=1, description=f"[{style}]{marker} {name}[/{style}]"
        )
        self._current = None
