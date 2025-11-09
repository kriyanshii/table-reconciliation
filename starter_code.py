"""
Table Reconciliation Starter Code

This file provides data structures and a basic skeleton.
Feel free to modify this structure as needed for your approach!
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class TableItem:
    """Represents a cell in the extracted table."""
    id: int
    row: int
    column: int
    amount_text: str


@dataclass
class OCRLine:
    """Represents a text box detected by OCR with position coordinates."""
    id: int
    text: str
    h_min: float  # Top edge (vertical position)
    h_max: float  # Bottom edge
    w_min: float  # Left edge (horizontal position)
    w_max: float  # Right edge


@dataclass
class Match:
    """Represents a successful match between a table item and OCR line."""
    table_item_id: int
    ocr_line_id: int


@dataclass
class ReconciliationStats:
    """Statistics about the reconciliation process."""
    total_items: int = 0
    matched_items: int = 0
    unmatched_items: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.matched_items / self.total_items) * 100


def normalize_text(text: str) -> str:
    """
    TODO: Implement text normalization.

    Normalize text for matching. Think about:
    - How to handle currency symbols ($, €, etc.)
    - How to handle negative notations: (500) vs -500
    - How to handle decimals and commas: 1,234.50
    - What canonical format makes matching easiest?

    Examples of desired behavior:
        "$1,234.50" should match "1234.50"
        "(500.00)" should match "-500"
        " 100.00 " should match "100"

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text for matching
    """
    # TODO: Your implementation here
    pass


class TableReconciliationEngine:
    """
    Main reconciliation engine.

    TODO: Design and implement your matching algorithm.

    You have access to:
    - self.table_items: List of table cells to match
    - self.ocr_lines: List of OCR text boxes with positions
    - self.matches: Where you should store successful matches
    - self.stats: Track statistics

    Think about:
    - How will you match table items to OCR lines?
    - How will you handle duplicate text values?
    - How can you use position information?
    - What order should you process items?
    - How can you avoid matching the same OCR line twice?
    """

    def __init__(self, table_items: List[TableItem], ocr_lines: List[OCRLine]):
        self.table_items = table_items
        self.ocr_lines = ocr_lines
        self.matches: List[Match] = []
        self.stats = ReconciliationStats()

        # TODO: Add any data structures you need
        # Hint: A dictionary mapping normalized text to OCR lines could be useful

    def reconcile(self) -> Tuple[List[Match], ReconciliationStats]:
        """
        Main reconciliation method.

        TODO: Implement your matching algorithm here.

        Returns:
            Tuple of (matches, stats)
        """
        # TODO: Your algorithm implementation

        # Update stats before returning
        self.stats.total_items = len(self.table_items)
        self.stats.matched_items = len(self.matches)
        self.stats.unmatched_items = self.stats.total_items - self.stats.matched_items

        return self.matches, self.stats

    # TODO: Add any helper methods you need
    # For example:
    # - Method to group items by row
    # - Method to find matching OCR lines for a table item
    # - Method to calculate distance between positions
    # - Method to select best match from multiple candidates
    # - etc.


def load_test_case(filepath: str) -> Tuple[List[TableItem], List[OCRLine]]:
    """
    Load test case from JSON file.

    Args:
        filepath: Path to JSON test case file

    Returns:
        Tuple of (table_items, ocr_lines)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    table_items = [TableItem(**item) for item in data['table_items']]
    ocr_lines = [OCRLine(**line) for line in data['ocr_lines']]

    return table_items, ocr_lines


def save_results(matches: List[Match], stats: ReconciliationStats, filepath: str):
    """Save reconciliation results to JSON file."""
    output = {
        'matches': [{'table_item_id': m.table_item_id, 'ocr_line_id': m.ocr_line_id}
                    for m in matches],
        'stats': {
            'total_items': stats.total_items,
            'matched_items': stats.matched_items,
            'unmatched_items': stats.unmatched_items,
            'success_rate': stats.success_rate
        }
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    """Main function to run reconciliation on test cases."""
    test_cases = ['basic.json', 'intermediate.json', 'advanced.json']

    for test_case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Running test case: {test_case}")
        print('=' * 60)

        # Load test case
        table_items, ocr_lines = load_test_case(f'test_cases/{test_case}')

        print(f"Loaded {len(table_items)} table items and {len(ocr_lines)} OCR lines")

        # Run reconciliation
        engine = TableReconciliationEngine(table_items, ocr_lines)
        matches, stats = engine.reconcile()

        # Print results
        print(f"\nResults:")
        print(f"  Total items: {stats.total_items}")
        print(f"  Matched: {stats.matched_items}")
        print(f"  Unmatched: {stats.unmatched_items}")
        print(f"  Success rate: {stats.success_rate:.1f}%")

        # Save results
        output_file = f'output_{test_case}'
        save_results(matches, stats, output_file)
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
