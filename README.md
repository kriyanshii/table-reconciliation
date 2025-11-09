# Table Reconciliation Coding Challenge

## Background

You're building a document processing system that extracts financial data from scanned documents. The system uses two different technologies:

1. **Table extraction model**: Identifies table structures and extracts data in a grid format (rows, columns, cell values)
2. **OCR (Optical Character Recognition)**: Reads all text from the document with precise position coordinates

The problem? These two systems work independently and don't communicate. You have table data with no position information, and OCR text with positions but no table structure. Your task is to **reconcile** them - match each table cell to its corresponding OCR text box to get both structured data AND positions.

## Why This Matters

Knowing the exact position of each table cell enables:
- Highlighting/annotating specific values in the UI
- Extracting data with visual context
- Validating that table extraction matches what OCR sees
- Building interactive document viewers

## The Challenge

Design and implement an algorithm that matches table cells to OCR text boxes.

Your solution should:
1. **Normalize** text to handle formatting differences (currency symbols, decimals, negatives)
2. **Match** table cells to their corresponding OCR text boxes
3. **Handle ambiguity** - when multiple OCR texts have the same content, use position or other heuristics to pick the right one
4. **Maximize success rate** - aim to correctly match as many table items as possible

**Note:** The algorithm design is up to you! Think about what information you have available (text content, positions, row/column structure) and how you can use it effectively.

## Input Format

### Table Items
```python
{
  "table_items": [
    {
      "id": 1,
      "row": 0,
      "column": 0,
      "amount_text": "1,234.50"
    },
    {
      "id": 2,
      "row": 0,
      "column": 1,
      "amount_text": "(500.00)"
    }
    # ... more items
  ]
}
```

- `row`: 0-indexed row number in the table
- `column`: 0-indexed column number in the table
- `amount_text`: Raw text extracted from table cell

### OCR Lines
```python
{
  "ocr_lines": [
    {
      "id": 101,
      "text": "$1,234.50",
      "h_min": 100.0,
      "h_max": 115.0,
      "w_min": 50.0,
      "w_max": 120.0
    },
    {
      "id": 102,
      "text": "(500.00)",
      "h_min": 100.0,
      "h_max": 115.0,
      "w_min": 200.0,
      "w_max": 260.0
    }
    # ... more lines
  ]
}
```

- `h_min/h_max`: Vertical position (top/bottom edge)
- `w_min/w_max`: Horizontal position (left/right edge)
- Coordinate system: origin at top-left, h increases downward, w increases rightward

## Output Format

```python
{
  "matches": [
    {
      "table_item_id": 1,
      "ocr_line_id": 101
    },
    {
      "table_item_id": 2,
      "ocr_line_id": 102
    }
    # ... more matches
  ],
  "stats": {
    "total_items": 10,
    "matched_items": 9,
    "unmatched_items": 1,
    "success_rate": 90.0
  }
}
```

## Core Requirements

### 1. Text Normalization (Required)
Implement a function to normalize text for matching:
- Handle currency symbols, spaces, non-breaking spaces
- Handle negative notations: `(500)` and `-500` should match
- Handle decimal and comma variations

Examples:
- `"$1,234.50"` should match `"1234.50"`
- `"(500.00)"` should match `"-500"`
- `" 100.00 "` should match `"100"`

**Hint:** Converting to a canonical format (e.g., digit-only string) can simplify matching.

### 2. Matching Algorithm (Required)
Design an algorithm to match table items to OCR lines. Consider:
- **Text matching:** How will you find OCR lines with matching content?
- **Ambiguity resolution:** What if multiple OCR lines have the same text?
- **Position information:** How can you use `h_min/h_max/w_min/w_max` coordinates?
- **Table structure:** How can you use row/column information?

**Things to think about:**
- Should you process rows in a specific order?
- Can you use successfully matched items to help match others?
- How do you avoid matching the same OCR line to multiple table items?
- What heuristics can improve accuracy when there are multiple candidates?

### 3. Success Rate (Goal)
Aim for high success rates on the test cases:
- Basic: Target 100%
- Intermediate: Target 100%
- Advanced: Target 85%+ (some items may be unmatchable)

Your algorithm will be evaluated on both correctness and approach quality.

## What We're Looking For

### Code Quality
- Clean, readable code with meaningful variable names
- Proper data structure choices
- Comments explaining non-obvious logic
- Modular design (separate functions for each step)

### Algorithm Design
- Well-reasoned approach to the matching problem
- Efficient use of data structures (e.g., dictionaries for fast lookup)
- Smart use of available information (text, positions, structure)
- Logical separation of concerns

### Edge Case Handling
- Empty cells or empty OCR text
- Duplicate values (same text appearing multiple times)
- Missing or extra OCR lines
- Ambiguous matches (multiple valid candidates)

### Testing
- Demonstrate your solution works on provided test cases
- Add any additional test cases you think are important

## Test Cases Provided

1. **basic.json**: Simple 3x3 table with all unique values
2. **intermediate.json**: Table with some duplicate values requiring position logic
3. **advanced.json**: Complex scenarios with missing values and high ambiguity

## Bonus Challenges (Optional)

If you finish early and want to showcase additional skills:

1. **Confidence Scores**: Add confidence scores to matches (e.g., based on uniqueness, position distance)
2. **Statistics/Logging**: Add detailed logging showing the matching process and decision points
3. **Performance Optimization**: Optimize for large tables (100+ rows, 1000+ OCR lines)
4. **Unit Tests**: Write comprehensive unit tests for your functions
5. **Algorithm Analysis**: Document the time/space complexity of your approach

## Submission Guidelines

Submit:
1. Your implementation (Python preferred, but any language is fine)
2. A brief README explaining:
   - How to run your code
   - **Your algorithm and approach** (this is important!)
   - Why you chose this approach
   - Any assumptions you made
   - Trade-offs and what you would improve with more time

## Evaluation Criteria

- **Correctness (40%)**: Passes test cases, handles edge cases
- **Code Quality (30%)**: Readability, organization, documentation
- **Algorithm Design (20%)**: Sound approach, proper data structures, efficient solution
- **Edge Cases (10%)**: Robustness and defensive coding

**Note:** There are multiple valid approaches to this problem. We're evaluating the quality of your solution, not whether it matches a specific algorithm.

## Questions?

If anything is unclear, make reasonable assumptions and document them in your README. We value seeing your thought process!

Good luck!
