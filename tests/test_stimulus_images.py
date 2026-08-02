from sdanalysis_kun.stimulus_images import find_stimulus_png


def test_find_stimulus_png_uses_exact_text_match_first(tmp_path):
    exact = tmp_path / "1.png"
    padded = tmp_path / "001.png"
    exact.touch()
    padded.touch()

    assert find_stimulus_png(tmp_path, 1) == exact
    assert find_stimulus_png(tmp_path, "001") == padded


def test_find_stimulus_png_matches_zero_padded_integer_after_csv_conversion(tmp_path):
    padded = tmp_path / "001.png"
    padded.touch()

    assert find_stimulus_png(tmp_path, 1) == padded
    assert find_stimulus_png(tmp_path, 1.0) == padded


def test_find_stimulus_png_does_not_apply_numeric_matching_to_alphanumeric_ids(tmp_path):
    image = tmp_path / "A001.png"
    image.touch()

    assert find_stimulus_png(tmp_path, "A001") == image
    assert find_stimulus_png(tmp_path, "a001") is None
