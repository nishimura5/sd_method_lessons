from pathlib import Path


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
