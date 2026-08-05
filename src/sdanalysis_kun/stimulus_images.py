from pathlib import Path


def find_thumbnail_png_folder(csv_path):
    """Find a sibling ``thumb`` or ``thumbnail`` folder containing PNG files."""
    csv_folder = Path(csv_path).parent

    for folder_name in ("thumb", "thumbnail"):
        folder_path = csv_folder / folder_name
        if not folder_path.is_dir():
            continue

        try:
            if any(path.is_file() and path.suffix.lower() == ".png" for path in folder_path.iterdir()):
                return folder_path
        except OSError:
            continue

    return None


def find_stimulus_png(folder, stimulus_id):
    """Find the PNG named ``<stimulus_id>.png`` in a folder."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return None

    stimulus_text = str(stimulus_id)
    try:
        for path in folder_path.iterdir():
            if path.is_file() and path.suffix.lower() == ".png" and path.stem == stimulus_text:
                return path
    except OSError:
        return None

    return None
