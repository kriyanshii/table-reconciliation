# Quick Start Guide

Get up and running with the Table Reconciliation challenge in 5 minutes!

## Setup

### Option 1: Python (Recommended)

```bash
# No dependencies needed! Uses only Python standard library
python starter_code.py
```

### Option 2: Your Preferred Language

- Read `README.md` for the problem description
- Look at `test_cases/*.json` for input format
- Implement your solution in any language

## What to Do

### Step 1: Understand the Problem (15 min)

Read through `README.md` and examine the test case files:

```bash
# Look at the basic test case to understand the data format
cat test_cases/basic.json
```

**Key observations:**

- Table items have: id, row, column, amount_text
- OCR lines have: id, text, h_min/h_max (vertical), w_min/w_max (horizontal)
- You need to match them and return Match objects

### Step 2: Implement Text Normalization

Open `starter_code.py` and implement the `normalize_text` function:

```python
def normalize_text(text: str) -> str:
    # Your implementation here
    pass
```

**Goal:** Make different formats match the same way

- `"$1,234.50"` and `"1234.50"` should normalize to the same thing
- `"(500.00)"` and `"-500"` should be equivalent
- Think about what canonical format is easiest for matching

**Test it:**

```python
# Add these at the bottom of starter_code.py temporarily
print(normalize_text("$1,234.50"))
print(normalize_text("(500.00)"))
print(normalize_text(" 100.00 "))
```

### Step 3: Design Your Algorithm

**This is the creative part!** Think about your approach:

**Questions to consider:**

1. **Text matching:** How will you find OCR lines with matching text?
    - Build a dictionary mapping normalized text to OCR lines?
    - Search linearly through all OCR lines for each item?

2. **Handling duplicates:** What if the same amount appears multiple times?
    - Use position to pick the right one?
    - Process in a specific order to reduce ambiguity?

3. **Using table structure:** Can row/column information help?
    - Match row by row?
    - Find a "reliable" row first and use it as a reference?

4. **Position matching:** How will you use coordinates when text alone isn't enough?
    - Calculate distances between positions?
    - Look for proximity patterns?

**Sketch out your algorithm before coding!**

### Step 4: Implement Your Algorithm

Fill in the `TableReconciliationEngine.reconcile()` method:

```python
def reconcile(self) -> Tuple[List[Match], ReconciliationStats]:
    # Your algorithm here

    # Remember to:
    # 1. Match table items to OCR lines
    # 2. Store matches in self.matches
    # 3. Avoid reusing OCR lines
    # 4. Update stats

    return self.matches, self.stats
```

**Hints:**

- Add helper methods as needed (the TODO comments suggest some ideas)
- Test incrementally - get basic.json working first
- Print debug information to understand what's happening

**Example helper method structure:**

```python
def _build_text_map(self):
    """Map normalized text to OCR lines."""
    # Could be useful for fast lookup
    pass


def _find_candidates(self, table_item):
    """Find OCR lines that match this item's text."""
    pass


def _select_best_match(self, table_item, candidates):
    """Pick the best OCR line from multiple candidates."""
    # This is where position logic might go
    pass
```

### Step 5: Test Your Solution

```bash
python starter_code.py
```

**Expected output:**

```
Running test case: basic.json
...
Success rate: ???%
```

### Step 6: Refine & Document

- Clean up your code
- Add comments explaining your approach
- Write a brief README explaining:
    - Your algorithm/approach
    - Why you chose this method
    - Trade-offs you considered
    - What you'd improve with more time

## Submission Checklist

Before submitting:

- [ ] All three test cases run without errors
- [ ] Basic test case: 90%+ success rate
- [ ] Intermediate test case: 85%+ success rate
- [ ] Advanced test case: 75%+ success rate
- [ ] Code is clean and commented
- [ ] No debug print statements left
- [ ] README.md included explaining your approach
- [ ] Edge cases handled gracefully
