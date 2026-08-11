from __future__ import annotations

from types import SimpleNamespace

from tools import monitor_training


def test_read_scalars_merges_all_event_files(monkeypatch, tmp_path) -> None:
    first = tmp_path / "events.out.tfevents.first"
    second = tmp_path / "events.out.tfevents.second"
    first.touch()
    second.touch()

    values = {
        str(first): [SimpleNamespace(step=0, value=1.0, wall_time=1.0)],
        str(second): [
            SimpleNamespace(step=0, value=2.0, wall_time=2.0),
            SimpleNamespace(step=1, value=3.0, wall_time=3.0),
        ],
    }

    class FakeAccumulator:
        def __init__(self, path, size_guidance):
            self.path = path

        def Reload(self):
            return None

        def Tags(self):
            return {"scalars": ["train/loss"]}

        def Scalars(self, tag):
            return values[self.path]

    monkeypatch.setattr(monitor_training, "EventAccumulator", FakeAccumulator)
    scalars, event_names = monitor_training.read_scalars(tmp_path)

    assert [item.value for item in scalars["train/loss"]] == [2.0, 3.0]
    assert event_names is not None
    assert "first" in event_names and "second" in event_names
