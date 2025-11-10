"""
Table Reconciliation Starter Code

This file provides data structures and a basic skeleton.
Feel free to modify this structure as needed for your approach!
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple


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
    if text is None:
        return ""

    # Normalize whitespace and minus characters
    cleaned = text.replace("\u00a0", " ").replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = cleaned.strip()
    if not cleaned:
        return ""

    # Collapse internal whitespace for consistent comparisons
    cleaned = re.sub(r"\s+", " ", cleaned)

    negative = False
    candidate = cleaned

    # Treat wrapping parentheses as negative indicator
    paren_match = re.fullmatch(r"\(\s*(.+?)\s*\)", candidate)
    if paren_match:
        negative = True
        candidate = paren_match.group(1)

    candidate = candidate.strip()

    # Track explicit negative indicators
    if candidate.startswith("-"):
        negative = True
        candidate = candidate[1:].strip()
    if candidate.endswith("-"):
        negative = True
        candidate = candidate[:-1].strip()

    # Remove currency symbols and other non-numeric characters, preserving decimal separators
    numeric_portion = re.sub(r"[^\d,\.]", "", candidate)
    numeric_portion = numeric_portion.replace(",", "")

    if numeric_portion and re.search(r"\d", numeric_portion):
        try:
            decimal_value = Decimal(numeric_portion)
        except (InvalidOperation, ValueError):
            decimal_value = None

        if decimal_value is not None:
            normalized = format(decimal_value, "f")
            if "." in normalized:
                normalized = normalized.rstrip("0").rstrip(".")
            if not normalized:
                normalized = "0"
            if negative and normalized != "0":
                normalized = f"-{normalized}"
            return normalized

    # Fallback for non-numeric strings: lowercase trimmed text
    return cleaned.lower()


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

        self._ocr_by_text: Dict[str, List[OCRLine]] = defaultdict(list)
        self._line_centers: Dict[int, Tuple[float, float]] = {}
        self._line_normalized_text: Dict[int, str] = {}
        self._line_unsigned_key: Dict[int, Optional[str]] = {}
        self._matched_ocr_ids: set[int] = set()
        self._row_center_estimates: Dict[int, float] = {}
        self._row_match_counts: Dict[int, int] = defaultdict(int)
        self._column_center_estimates: Dict[int, float] = {}
        self._column_match_counts: Dict[int, int] = defaultdict(int)
        self._ocr_by_unsigned_text: Dict[str, List[OCRLine]] = defaultdict(list)

        self.row_count = (max((item.row for item in self.table_items), default=-1) + 1)
        self.column_count = (max((item.column for item in self.table_items), default=-1) + 1)

        self._min_center_h: Optional[float] = None
        self._max_center_h: Optional[float] = None
        self._min_center_w: Optional[float] = None
        self._max_center_w: Optional[float] = None
        self._row_baseline_centers: List[float] = []
        self._column_baseline_centers: List[float] = []

        self._prepare_lookup_structures()

    def reconcile(self) -> Tuple[List[Match], ReconciliationStats]:
        """
        Main reconciliation method.

        TODO: Implement your matching algorithm here.

        Returns:
            Tuple of (matches, stats)
        """
        for table_item in sorted(self.table_items, key=lambda item: (item.row, item.column, item.id)):
            normalized_text = normalize_text(table_item.amount_text)
            if not normalized_text:
                continue

            candidates = self._find_candidates(normalized_text)
            if not candidates:
                continue

            best_candidate = self._select_best_candidate(table_item, candidates)
            if best_candidate is None:
                continue

            self._register_match(table_item, best_candidate)

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

    def _prepare_lookup_structures(self) -> None:
        if not self.ocr_lines:
            self._min_center_h = self._max_center_h = 0.0
            self._min_center_w = self._max_center_w = 0.0
            return

        centers_h: List[float] = []
        centers_w: List[float] = []
        numeric_centers_h: List[float] = []
        numeric_centers_w: List[float] = []

        for line in self.ocr_lines:
            center = self._compute_center(line)
            self._line_centers[line.id] = center
            centers_h.append(center[0])
            centers_w.append(center[1])

            normalized_text = normalize_text(line.text)
            self._ocr_by_text[normalized_text].append(line)
            self._line_normalized_text[line.id] = normalized_text

            unsigned_key = self._unsigned_key(normalized_text)
            self._line_unsigned_key[line.id] = unsigned_key
            if unsigned_key is not None:
                self._ocr_by_unsigned_text[unsigned_key].append(line)
                numeric_centers_h.append(center[0])
                numeric_centers_w.append(center[1])

        ref_h = numeric_centers_h if numeric_centers_h else centers_h
        ref_w = numeric_centers_w if numeric_centers_w else centers_w

        self._min_center_h = min(ref_h)
        self._max_center_h = max(ref_h)
        self._min_center_w = min(ref_w)
        self._max_center_w = max(ref_w)

        self._row_baseline_centers = self._compute_axis_baselines(ref_h, self.row_count)
        self._column_baseline_centers = self._compute_axis_baselines(ref_w, self.column_count)

        for lines in self._ocr_by_text.values():
            lines.sort(key=lambda l: (self._line_centers[l.id][0], self._line_centers[l.id][1], l.id))

    @staticmethod
    def _compute_center(line: OCRLine) -> Tuple[float, float]:
        return (
            (line.h_min + line.h_max) / 2.0,
            (line.w_min + line.w_max) / 2.0,
        )

    def _find_candidates(self, normalized_text: str) -> List[OCRLine]:
        direct_candidates = [
            line
            for line in self._ocr_by_text.get(normalized_text, [])
            if line.id not in self._matched_ocr_ids
        ]
        if direct_candidates:
            return direct_candidates

        unsigned_key = self._unsigned_key(normalized_text)
        if unsigned_key is None:
            return []

        return [
            line
            for line in self._ocr_by_unsigned_text.get(unsigned_key, [])
            if line.id not in self._matched_ocr_ids
        ]

    def _select_best_candidate(self, table_item: TableItem, candidates: List[OCRLine]) -> Optional[OCRLine]:
        best_line: Optional[OCRLine] = None
        best_score: Optional[float] = None

        for line in candidates:
            score = self._score_candidate(table_item, line)
            if best_score is None or score < best_score:
                best_score = score
                best_line = line

        return best_line

    def _score_candidate(self, table_item: TableItem, line: OCRLine) -> float:
        center_h, center_w = self._line_centers[line.id]

        expected_h = self._expected_row_center(table_item.row)
        expected_w = self._expected_column_center(table_item.column)

        score = abs(center_h - expected_h) * 1.0 + abs(center_w - expected_w) * 0.6

        if table_item.row in self._row_center_estimates:
            row_estimate = self._row_center_estimates[table_item.row]
            score += abs(center_h - row_estimate) * 0.3

        if table_item.column in self._column_center_estimates:
            column_estimate = self._column_center_estimates[table_item.column]
            score += abs(center_w - column_estimate) * 0.3

        score += line.id * 1e-6  # Deterministic tie-breaker
        return score

    def _expected_row_center(self, row_index: int) -> float:
        if self._min_center_h is None or self._max_center_h is None:
            return 0.0
        if 0 <= row_index < len(self._row_baseline_centers):
            return self._row_baseline_centers[row_index]
        if self.row_count <= 1 or self._min_center_h == self._max_center_h:
            return (self._min_center_h + self._max_center_h) / 2.0
        spacing = (self._max_center_h - self._min_center_h) / max(self.row_count - 1, 1)
        return self._min_center_h + spacing * row_index

    def _expected_column_center(self, column_index: int) -> float:
        if self._min_center_w is None or self._max_center_w is None:
            return 0.0
        if 0 <= column_index < len(self._column_baseline_centers):
            return self._column_baseline_centers[column_index]
        if self.column_count <= 1 or self._min_center_w == self._max_center_w:
            return (self._min_center_w + self._max_center_w) / 2.0
        spacing = (self._max_center_w - self._min_center_w) / max(self.column_count - 1, 1)
        return self._min_center_w + spacing * column_index

    def _register_match(self, table_item: TableItem, line: OCRLine) -> None:
        self.matches.append(Match(table_item_id=table_item.id, ocr_line_id=line.id))
        self._matched_ocr_ids.add(line.id)

        center_h, center_w = self._line_centers[line.id]
        self._update_row_estimate(table_item.row, center_h)
        self._update_column_estimate(table_item.column, center_w)

        # Remove matched line from candidate pool to keep future lookups efficient
        normalized_key = self._line_normalized_text.get(line.id)
        if normalized_key is not None:
            candidate_list = self._ocr_by_text.get(normalized_key)
            if candidate_list is not None:
                candidate_list[:] = [candidate for candidate in candidate_list if candidate.id != line.id]

        unsigned_key = self._line_unsigned_key.get(line.id)
        if unsigned_key is not None:
            unsigned_list = self._ocr_by_unsigned_text.get(unsigned_key)
            if unsigned_list is not None:
                unsigned_list[:] = [candidate for candidate in unsigned_list if candidate.id != line.id]

    def _update_row_estimate(self, row_index: int, value: float) -> None:
        self._row_match_counts[row_index] += 1
        count = self._row_match_counts[row_index]
        previous = self._row_center_estimates.get(row_index, 0.0)
        self._row_center_estimates[row_index] = previous + (value - previous) / count

    def _update_column_estimate(self, column_index: int, value: float) -> None:
        self._column_match_counts[column_index] += 1
        count = self._column_match_counts[column_index]
        previous = self._column_center_estimates.get(column_index, 0.0)
        self._column_center_estimates[column_index] = previous + (value - previous) / count

    @staticmethod
    def _is_numeric_key(value: str) -> bool:
        return bool(value) and re.fullmatch(r"-?\d+(?:\.\d+)?", value) is not None

    def _unsigned_key(self, normalized_text: str) -> Optional[str]:
        if not self._is_numeric_key(normalized_text):
            return None
        unsigned = normalized_text.lstrip("-")
        return unsigned or "0"

    @staticmethod
    def _compute_axis_baselines(values: List[float], group_count: int) -> List[float]:
        if group_count <= 0 or not values:
            return []
        sorted_values = sorted(values)
        if group_count == 1:
            return [sum(sorted_values) / len(sorted_values)]

        start = sorted_values[0]
        end = sorted_values[-1]
        if start == end:
            return [start for _ in range(group_count)]

        centroids = [
            start + (end - start) * i / max(group_count - 1, 1)
            for i in range(group_count)
        ]

        for _ in range(12):
            clusters: List[List[float]] = [[] for _ in range(group_count)]
            for value in sorted_values:
                idx = min(range(group_count), key=lambda j: abs(value - centroids[j]))
                clusters[idx].append(value)

            new_centroids = []
            for cluster, centroid in zip(clusters, centroids):
                if cluster:
                    new_centroids.append(sum(cluster) / len(cluster))
                else:
                    new_centroids.append(centroid)

            if all(abs(nc - oc) < 1e-3 for nc, oc in zip(new_centroids, centroids)):
                centroids = new_centroids
                break

            centroids = new_centroids

        return sorted(centroids)


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
