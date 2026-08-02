from types import SimpleNamespace

import pytest

from sdanalysis_kun.app_sd import SDApp


class ScrollCanvasStub:
    def __init__(self):
        self.calls = []

    def yview_scroll(self, units, mode):
        self.calls.append((units, mode))


@pytest.mark.parametrize(
    ("event", "expected_units"),
    [
        (SimpleNamespace(num=None, delta=120), -1),
        (SimpleNamespace(num=None, delta=-240), 2),
        (SimpleNamespace(num=None, delta=1), -1),
        (SimpleNamespace(num=4, delta=0), -1),
        (SimpleNamespace(num=5, delta=0), 1),
    ],
)
def test_scroll_adjective_pairs_supports_mousewheel_events(event, expected_units):
    app = SDApp.__new__(SDApp)
    app.adjective_canvas = ScrollCanvasStub()

    result = app._scroll_adjective_pairs(event)

    assert app.adjective_canvas.calls == [(expected_units, "units")]
    assert result == "break"


def test_scroll_adjective_pairs_ignores_empty_mousewheel_event():
    app = SDApp.__new__(SDApp)
    app.adjective_canvas = ScrollCanvasStub()

    result = app._scroll_adjective_pairs(SimpleNamespace(num=None, delta=0))

    assert app.adjective_canvas.calls == []
    assert result is None
