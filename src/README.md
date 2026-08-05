# SDAnalysis-kun

![SDAnalysis-kun main window](https://www.design.kyushu-u.ac.jp/~eigo/image/sdanalysis-kun/main_window.webp)

`src/sdanalysis_kun/` contains a Tkinter-based GUI application. It is a companion tool for trying the analyses from the lesson code through CSV selection and button-based operations.

## Downloads

- [Download SDAnalysis-kun for Windows](https://github.com/nishimura5/sd_method_lessons/releases/download/v0.3.0/SDAnalysisKun030_win.zip)
- [Download SDAnalysis-kun for macOS](https://github.com/nishimura5/sd_method_lessons/releases/download/v0.3.0/SDAnalysisKun030_mac.zip)

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
- Display a stimulus map using either PCA or two selected factor-score axes; when a respondent column is selected, each stimulus is summarized by its centroid and a within-stimulus 1-SD covariance ellipse representing respondent variability

## Expected CSV Structure

SDAnalysis-kun assumes a table with the following layout.

- The data is in long format for observations: each row is one response record (typically one respondent rating one stimulus).
- A `stimulus_id` column exists and stores the stimulus identifier.
- A `respondent_id` column exists and stores the respondent identifier.
- Adjective-pair ratings are in wide format: each adjective pair is a separate numeric column.

Example:

```csv
id,respondent_id,stimulus_id,fast-slow,bright-dark,soft-hard
1,r_001,s_001,5,2,4
2,r_001,s_002,3,4,2
3,r_002,s_001,6,3,5
```

In short: observation keys (`stimulus_id`, `respondent_id`) are row-wise, and adjective pairs are column-wise.
