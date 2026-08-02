import re
from pathlib import Path


_INTEGER_ID_PATTERN = re.compile(r"^[+-]?\d+(?:\.0+)?$")
_INTEGER_STEM_PATTERN = re.compile(r"^[+-]?\d+$")


def _as_integer_id(value):
    text = str(value).strip()
    if not _INTEGER_ID_PATTERN.fullmatch(text):
        return None
    return int(text.split(".", maxsplit=1)[0])


def find_stimulus_png(folder, stimulus_id):
    """Find a PNG whose stem matches a stimulus ID.

    Exact text matching takes priority. If an ID was converted to a number by
    CSV parsing, a unique integer-equivalent stem such as ``001`` is accepted
    for the value ``1``. A three-digit stem is preferred when multiple padded
    spellings of the same integer exist.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return None

    try:
        png_paths = sorted(
            (path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        return None
    stimulus_text = str(stimulus_id).strip()

    for path in png_paths:
        if path.stem == stimulus_text:
            return path

    integer_id = _as_integer_id(stimulus_id)
    if integer_id is None:
        return None

    numeric_matches = [
        path
        for path in png_paths
        if _INTEGER_STEM_PATTERN.fullmatch(path.stem) and int(path.stem) == integer_id
    ]
    if len(numeric_matches) == 1:
        return numeric_matches[0]

    recommended_stem = f"{integer_id:03d}"
    for path in numeric_matches:
        if path.stem == recommended_stem:
            return path

    return None
