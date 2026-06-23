# SDAnalysis-kun

`src/sdanalysis_kun/` contains a Tkinter-based GUI application. It is a companion tool for trying the analyses from the lesson code through CSV selection and button-based operations.

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
- Display and export factor scores by stimulus as CSV
- Display a stimulus map with PCA
