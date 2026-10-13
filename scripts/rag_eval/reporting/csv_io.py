"""CSV parsing, validation and semicolon-conversion helpers for the rag_eval runner."""

import json
import math
from pathlib import Path

import pandas as pd


def _parse_json_column(value, col_name: str, row_idx: int) -> tuple[list, bool]:
    """Parse JSON array string from CSV cell.

    Returns (parsed_list, had_parse_error).
    - Empty/NaN → ([], False) — not an error
    - Valid JSON array → (parsed_list, False)
    - Pipe-separated text → split by " | " → (parsed_list, False)
    - Malformed JSON / not a list and not pipe-separated → ([], True) — WARNING emitted
    """
    if isinstance(value, float) and math.isnan(value):
        return [], False
    if not isinstance(value, str) or value.strip() == "":
        return [], False
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed, False
        # Not a JSON array — treat as malformed
        print(f"  WARNING: Row {row_idx}: {col_name} is not a JSON array — {str(value)[:100]}")
        return [], True
    except json.JSONDecodeError:
        # FALLBACK 1: pipe-separated plain text (output format uses " | ".join)
        if " | " in value:
            parts = [p.strip() for p in value.split(" | ") if p.strip()]
            if parts:
                return parts, False
        # FALLBACK 2: single plain-text value → return as single-element list
        # (handles citations like ["text"](chunk_N), grading notes, etc.)
        stripped = value.strip()
        if stripped:
            return [stripped], False
        return [], False


def _validate_csv_columns(df: pd.DataFrame) -> None:
    """Validate that required columns are present in the CSV DataFrame.

    Raises ValueError naming missing columns if any required column is absent.
    Required columns: question, response, retrieved_contexts.
    """
    required = {"question", "response", "retrieved_contexts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def _is_missing_value(value) -> bool:
    """Return True if value is None, NaN, or pd.NA.
    
    Used by _run_csv_mode guards for citations, latency_ms, and grading_notes.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _move_model_id_last(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with ``model_id`` re-appended as the last column.

    No-op when the column is absent. Post-hoc columns (e.g. ``parametric_suspect``)
    append after the per-row ``model_id`` stamp, so every CSV rewrite puts the
    evaluated-model column back at the end.
    """
    if "model_id" not in df.columns:
        return df
    return df[[c for c in df.columns if c != "model_id"] + ["model_id"]]


# Minimum number of columns of a plausible results table. A comma parse that
# yields fewer is a misparse of a non-comma file, never a rewrite candidate.
_RESULTS_TABLE_MIN_COLUMNS = 10


def _is_already_semicolon_delimited(raw: bytes) -> bool:
    """True when the first line is already semicolon-delimited.

    Headers are simple, unquoted column names, so the first physical line is a
    reliable delimiter signal even when ``csv.Sniffer`` fails on the ragged
    RAGAS rows (embedded newlines/quotes) and would otherwise misparse the
    file and rewrite it lossily.
    """
    first_line = raw.split(b"\n", 1)[0]
    return first_line.count(b";") > first_line.count(b",")


def _is_results_table(df: pd.DataFrame) -> bool:
    """True when *df* looks like a results table this pass may rewrite.

    Requires the ``question`` column and at least
    ``_RESULTS_TABLE_MIN_COLUMNS`` columns, so a comma parse of a quoted
    single-column blob is skipped instead of persisted as a lossy rewrite.
    """
    return "question" in df.columns and len(df.columns) >= _RESULTS_TABLE_MIN_COLUMNS


def _csv_to_semicolon(directory: Path) -> None:
    """Rewrite all CSVs in directory using semicolon separator (Excel-friendly).

    Safety invariants:
    - Files whose first line already uses semicolons (more ``;`` than ``,``)
      are skipped before any parse, so converted files stay byte-identical.
    - Phase 1 only rewrites a comma parse that looks like a results table
      (``question`` column, at least ``_RESULTS_TABLE_MIN_COLUMNS`` columns);
      anything else is skipped with a visible warning and left untouched.
    - Phase 2 keeps the csv-module fallback for files with malformed quoting
      and writes to a temp file first, then replaces atomically to avoid data
      loss.
    """
    import csv as csv_module

    for csv_path in directory.glob("*.csv"):
        # REQ-009: Use csv.Sniffer on first 8KB to detect the actual delimiter
        try:
            raw = csv_path.open("rb").read(8192)
        except OSError:
            continue
        if not raw:
            continue

        # Deterministic already-converted detector (REQ-PIPELINE-HARDENING-009):
        # the first line decides before any parse, so a Sniffer misdetection on
        # ragged rows can never rewrite an already-semicolon file.
        if _is_already_semicolon_delimited(raw):
            continue

        # If the file is already semicolon-delimited, skip conversion
        try:
            dialect = csv_module.Sniffer().sniff(raw[:8192].decode("utf-8-sig", errors="replace"))
            if dialect.delimiter == ";":
                continue  # already converted
        except csv_module.Error:
            pass  # sniffer failed — fall through to Phase 1

        # Phase 1: pandas with python engine (handles more edge cases than C engine)
        try:
            df = pd.read_csv(csv_path, sep=",", engine="python", encoding="utf-8-sig")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            pass  # parsing failed — fall through to Phase 2
        else:
            # Shape guard: a comma parse that is not a results table is a
            # misparse of a non-comma file. Skip it instead of rewriting.
            if not _is_results_table(df):
                print(
                    f"Warning: Skipping {csv_path.name}: comma parse yielded "
                    f"{len(df.columns)} column(s) without a 'question' column — "
                    f"not a results table; file left untouched"
                )
                continue
            tmp_path = csv_path.with_name(csv_path.name + ".tmp")
            try:
                df.to_csv(tmp_path, sep=";", index=False)
                tmp_path.replace(csv_path)
            except OSError:
                print(f"Warning: Could not convert {csv_path.name} to semicolon format: write/replace failed")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            continue

        # Phase 2: csv module fallback for files with embedded commas
        #           (e.g. error trace cells that break fixed-width parsing)
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv_module.reader(f)
                rows = list(reader)
            if rows:
                # Write to temp file first, then atomically replace (avoid truncation loss)
                tmp_path = csv_path.with_name(csv_path.name + ".tmp")
                try:
                    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                        csv_module.writer(f, delimiter=";").writerows(rows)
                    tmp_path.replace(csv_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
                continue
        except Exception as phase2_exc:
            print(f"Warning: Could not convert {csv_path.name} to semicolon format: {phase2_exc}")
            continue

        print(f"Warning: Could not convert {csv_path.name} to semicolon format: unparseable content — both pandas and csv module failed to read the file")
