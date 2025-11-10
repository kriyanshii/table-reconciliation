## Table Reconciliation Solution

### How to Run
- Ensure you have Python 3 installed (standard library only).
- From the repository root, run `python3 starter_code.py`.
- The script processes `basic`, `intermediate`, and `advanced` test cases, printing summary stats and writing detailed results to `output_basic.json`, `output_intermediate.json`, and `output_advanced.json`.

### Approach and Design Decisions
- **Text normalization**: Implemented a canonical numeric string formatter that removes currency symbols, whitespace variants, and resolves negative indicators (parentheses or minus signs). Formatting trims redundant decimal zeros while preserving integer magnitude, producing stable keys such as `"-500"` and `"1234.56"`. Non-numeric text falls back to lowercased, whitespace-normalized strings.
- **Lookup structures**: OCR lines are indexed two ways—by normalized text and by absolute numeric magnitude. The latter enables tolerant matching when table extraction omits a negative sign but OCR preserves it in parentheses.
- **Spatial reasoning**: Each OCR line’s center coordinates are precomputed. Lightweight 1D k-means initializes baseline centers for rows and columns, using only numeric OCR lines to avoid outliers. Table items are processed row-major; candidates receive a score combining distance from expected row/column baselines with adaptive penalties that incorporate previously matched positions.
- **Ambiguity resolution**: Candidates are sorted deterministically; once matched, lines are removed from all indexes to prevent reuse. Tie-breaking uses tiny id-based offsets for reproducibility.

### Assumptions
- Numeric table items correspond to numeric OCR lines; non-numeric content (e.g., headers) is out of scope for matching.
- Parentheses always indicate negative amounts; if table data lacks the sign, magnitude matching is still desirable.
- Table structure is consistent (rows/columns form a grid) so positional heuristics based on evenly spaced centers are reasonable.

### Results
- `basic.json`: 100% match rate.
- `intermediate.json`: 100% match rate.
- `advanced.json`: 93.3% match rate (only the intentionally empty cell remains unmatched).

### Future Improvements
- Incorporate full-blown assignment (Hungarian) optimization using combined text and spatial costs to further reduce edge-case conflicts.
- Introduce probabilistic confidence scores per match and expose richer diagnostics (e.g., scoring components, fallback reasons).
- Add configurable tolerances and heuristics for headers or non-numeric cells, potentially using NLP to classify them.
- Expand automated testing with unit tests for normalization and scoring utilities, plus regression coverage for new document formats.

