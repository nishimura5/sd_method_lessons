import os
import sys

import matplotlib


def configure_matplotlib_backend():
    if "matplotlib.pyplot" in sys.modules:
        return

    if os.environ.get("MPLBACKEND"):
        return

    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        pass


configure_matplotlib_backend()
