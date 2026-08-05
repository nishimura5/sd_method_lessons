from types import SimpleNamespace

import pytest

from sdanalysis_kun.app_sd import SDApp, _read_sd_csv


class ScrollCanvasStub:
    def __init__(self):
        self.calls = []

    def yview_scroll(self, units, mode):
        self.calls.append((units, mode))


class StringVarStub:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class BindWidgetStub:
    def __init__(self):
        self.bindings = {}

    def bind(self, event_name, callback):
        self.bindings[event_name] = callback


class ComboStub:
    def __init__(self):
        self.values = None
        self.current_index = None

    def __setitem__(self, key, value):
        if key == "values":
            self.values = value

    def current(self, index):
        self.current_index = index


def test_read_sd_csv_preserves_stimulus_id_as_text(tmp_path):
    csv_path = tmp_path / "stimuli.csv"
    csv_path.write_text("stimulus_id,score\n001,5\n0010,6\n,7\n", encoding="utf-8")

    df = _read_sd_csv(csv_path, encoding="utf-8")

    assert df["stimulus_id"].tolist() == ["001", "0010", ""]
    assert all(isinstance(value, str) for value in df["stimulus_id"])
    assert df["score"].tolist() == [5, 6, 7]


def test_select_file_sets_sibling_thumbnail_folder_with_png(monkeypatch, tmp_path):
    csv_path = tmp_path / "stimuli.csv"
    csv_path.write_text("stimulus_id,score\n001,5\n", encoding="utf-8")
    thumbnail_folder = tmp_path / "thumb"
    thumbnail_folder.mkdir()
    (thumbnail_folder / "001.png").touch()

    app = SDApp.__new__(SDApp)
    app.file_path_var = StringVarStub()
    app.png_folder_var = StringVarStub()
    app.stimulus_col_combo = ComboStub()
    app.resp_col_combo = ComboStub()
    app.resp_col_var = StringVarStub()
    app._populate_checkboxes = lambda: None
    monkeypatch.setattr("sdanalysis_kun.app_sd.filedialog.askopenfilename", lambda **_options: str(csv_path))

    app._select_file()

    assert app.png_folder_var.get() == str(thumbnail_folder)


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


def test_select_png_folder_reuses_and_updates_session_folder(monkeypatch, tmp_path):
    current_folder = tmp_path / "current"
    selected_folder = tmp_path / "selected"
    current_folder.mkdir()
    selected_folder.mkdir()
    root = object()
    app = SDApp.__new__(SDApp)
    app.root = root
    app.png_folder_var = StringVarStub(str(current_folder))
    received_options = {}

    def askdirectory(**options):
        received_options.update(options)
        return str(selected_folder)

    monkeypatch.setattr("sdanalysis_kun.app_sd.filedialog.askdirectory", askdirectory)

    app._select_png_folder()

    assert received_options["parent"] is root
    assert received_options["initialdir"] == str(current_folder)
    assert app.png_folder_var.get() == str(selected_folder)


def test_select_png_folder_keeps_session_folder_when_cancelled(monkeypatch, tmp_path):
    current_folder = tmp_path / "current"
    current_folder.mkdir()
    app = SDApp.__new__(SDApp)
    app.root = object()
    app.png_folder_var = StringVarStub(str(current_folder))
    monkeypatch.setattr("sdanalysis_kun.app_sd.filedialog.askdirectory", lambda **_options: "")

    app._select_png_folder()

    assert app.png_folder_var.get() == str(current_folder)


def test_show_stimulus_png_prompts_for_folder_when_session_folder_is_empty():
    app = SDApp.__new__(SDApp)
    app.png_folder_var = StringVarStub()
    messages = []
    app._show_png_preview_message = lambda canvas, message: messages.append((canvas, message))
    preview_canvas = object()

    app._show_stimulus_png(preview_canvas, object(), "A")

    assert messages == [(preview_canvas, "Select a thumbnail PNG folder in the main window first.")]


def test_show_stimulus_png_uses_shared_lookup_and_preview(monkeypatch, tmp_path):
    png_path = tmp_path / "001.png"
    png_path.touch()
    app = SDApp.__new__(SDApp)
    app.png_folder_var = StringVarStub(str(tmp_path))
    shown = []
    app._show_png_preview = lambda canvas, dialog, path, stimulus: shown.append(
        (canvas, dialog, path, stimulus)
    )
    monkeypatch.setattr(
        "sdanalysis_kun.app_sd.find_stimulus_png",
        lambda folder, stimulus: png_path if (folder, stimulus) == (str(tmp_path), 1) else None,
    )
    preview_canvas = object()
    dialog = object()

    app._show_stimulus_png(preview_canvas, dialog, 1)

    assert shown == [(preview_canvas, dialog, png_path, 1)]


def test_bind_stimulus_png_hover_displays_the_bound_stimulus():
    app = SDApp.__new__(SDApp)
    shown = []
    app._show_stimulus_png = lambda canvas, dialog, stimulus: shown.append((canvas, dialog, stimulus))
    widget = BindWidgetStub()
    preview_canvas = object()
    dialog = object()

    app._bind_stimulus_png_hover(widget, preview_canvas, dialog, "stimulus-2")
    widget.bindings["<Enter>"](SimpleNamespace())

    assert shown == [(preview_canvas, dialog, "stimulus-2")]
