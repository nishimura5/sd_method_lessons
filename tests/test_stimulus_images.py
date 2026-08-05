import pytest

from sdanalysis_kun.stimulus_images import find_stimulus_png, find_thumbnail_png_folder


@pytest.mark.parametrize("folder_name", ["thumb", "thumbnail"])
def test_find_thumbnail_png_folder_finds_sibling_with_png(tmp_path, folder_name):
    csv_path = tmp_path / "data.csv"
    csv_path.touch()
    thumbnail_folder = tmp_path / folder_name
    thumbnail_folder.mkdir()
    (thumbnail_folder / "001.PNG").touch()

    assert find_thumbnail_png_folder(csv_path) == thumbnail_folder


def test_find_thumbnail_png_folder_ignores_empty_and_nested_png_folders(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.touch()
    thumb_folder = tmp_path / "thumb"
    thumb_folder.mkdir()
    nested_folder = thumb_folder / "nested"
    nested_folder.mkdir()
    (nested_folder / "001.png").touch()

    assert find_thumbnail_png_folder(csv_path) is None


def test_find_thumbnail_png_folder_uses_first_candidate_that_contains_png(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.touch()
    (tmp_path / "thumb").mkdir()
    thumbnail_folder = tmp_path / "thumbnail"
    thumbnail_folder.mkdir()
    (thumbnail_folder / "001.png").touch()

    assert find_thumbnail_png_folder(csv_path) == thumbnail_folder


def test_find_stimulus_png_uses_exact_text_match(tmp_path):
    exact = tmp_path / "1.png"
    padded = tmp_path / "001.png"
    exact.touch()
    padded.touch()

    assert find_stimulus_png(tmp_path, 1) == exact
    assert find_stimulus_png(tmp_path, "001") == padded


def test_find_stimulus_png_does_not_infer_zero_padding(tmp_path):
    padded = tmp_path / "0001.png"
    padded.touch()

    assert find_stimulus_png(tmp_path, 1) is None
    assert find_stimulus_png(tmp_path, "0001") == padded


def test_find_stimulus_png_does_not_apply_numeric_matching_to_alphanumeric_ids(tmp_path):
    image = tmp_path / "A001.png"
    image.touch()

    assert find_stimulus_png(tmp_path, "A001") == image
    assert find_stimulus_png(tmp_path, "a001") is None
