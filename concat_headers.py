

import csv
import json
import re
import io
import sys
from pathlib import Path


def _is_data_value(cell: str) -> bool:
    cell = cell.strip()
    if not cell or cell in ("-", "—", "–", "+", "±"):
        return False
    if re.fullmatch(r"[\d,.\s\-–±≤≥<>×]+", cell):
        return True
    return False


def detect_header_rows(rows: list[list[str]]) -> int:
    if len(rows) < 2:
        return 1

    def data_ratio(row: list[str]) -> float:
        non_empty = [c for c in row if c.strip() and c.strip() not in ("Unnamed: " + str(i) for i in range(100))]
        if not non_empty:
            return 0.0
        data_cells = sum(1 for c in non_empty if _is_data_value(c))
        return data_cells / len(non_empty)

    if data_ratio(rows[1]) < 0.4:
        return 2
    return 1


def _is_unnamed(cell: str) -> bool:
    return not cell.strip() or re.match(r"^Unnamed:\s*\d+$", cell.strip()) is not None


def flatten_headers(header_rows: list[list[str]]) -> list[str]:
    if len(header_rows) == 1:
        return [c.strip() for c in header_rows[0]]

    top, bot = header_rows[0], header_rows[1]
    n = max(len(top), len(bot))

    top = top + [""] * (n - len(top))
    bot = bot + [""] * (n - len(bot))

    filled_top: list[str] = []
    last_top = ""
    for cell in top:
        if _is_unnamed(cell):
            filled_top.append(last_top)
        else:
            last_top = cell.strip()
            filled_top.append(last_top)

    result = []
    for t, b in zip(filled_top, bot):
        b = b.strip()
        if b and not _is_unnamed(b):
            result.append(f"{b} {t}" if t else b)
        else:
            result.append(t)

    return result


def parse_csv(text: str, delimiter: str = ";") -> tuple[list[str], list[list[str]]]:

    reader = csv.reader(io.StringIO(text.strip()), delimiter=delimiter)
    rows = [row for row in reader]

    if not rows:
        return [], []

    n_header = detect_header_rows(rows)
    header_rows = rows[:n_header]
    data_rows = rows[n_header:]

    headers = flatten_headers(header_rows)
    return headers, data_rows



def to_flat_csv(headers: list[str], data_rows: list[list[str]], delimiter: str = ";") -> str:

    lines = [delimiter.join(headers)]
    for row in data_rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append(delimiter.join(padded[:len(headers)]))
    return "\n".join(lines)

