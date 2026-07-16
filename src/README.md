# SDAnalysis-kun

![SDAnalysis-kun main window](https://www.design.kyushu-u.ac.jp/~eigo/image/sdanalysis-kun/main_window.webp)

`src/sdanalysis_kun/` contains a Tkinter-based GUI application. It is a companion tool for trying the analyses from the lesson code through CSV selection and button-based operations.

## Downloads

- [Download SDAnalysis-kun for Windows](https://github.com/nishimura5/sd_method_lessons/releases/download/v0.2.0/SDAnalysisKun020_win.zip)
- [Download SDAnalysis-kun for macOS](https://github.com/nishimura5/sd_method_lessons/releases/download/v0.2.0/SDAnalysisKun020_mac.zip)

Launch command:

```bash
uv run sdanalysis-kun
```

Alternatively:

```bash
uv run sd-method-lessons
```

The GUI mainly supports the following operations:

- Select a CSV file
- Select the stimulus column
- Select the respondent column (optional)
- Filter the stimuli to analyze
- Select adjective-pair columns
- Specify a regular expression for formatting adjective-pair names
- Select a 5-point or 7-point rating scale
- Run parallel analysis with Pearson correlation or polychoric correlation
- Specify the number of factors, or use the number suggested by the parallel analysis (PA) result
- Choose Promax rotation, Varimax rotation, or no rotation
- Display, plot, and export the factor loading matrix as CSV
- Display and export the mean and standard deviation (SD) of factor scores by stimulus as CSV
- Display a stimulus map with PCA based on mean factor scores
