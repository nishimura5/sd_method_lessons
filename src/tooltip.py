import tkinter as tk


class ToolTip:
    """ウィジェットにマウスホバーで表示されるツールチップ。"""

    def __init__(self, widget, text, position="bottom"):
        self.widget = widget
        self.text = text
        self.position = position
        self.tipwindow = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("TkDefaultFont", 11),
            padx=8,
            pady=6,
        )
        label.pack()
        tw.update_idletasks()

        if self.position == "top":
            y = self.widget.winfo_rooty() - tw.winfo_height() - 5
        else:
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None
