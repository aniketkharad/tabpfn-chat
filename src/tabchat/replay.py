"""CLI replay and audit trail utility for tabchat sessions.

Usage:
    uv run python -m tabchat.replay <session_id> [--data-dir PATH]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO


def _format_header(title: str, width: int = 72) -> str:
    """Format section header banner."""
    pad = (width - len(title) - 4) // 2
    return f"\n{'=' * pad} [ {title} ] {'=' * pad}"


def _format_sub_header(title: str, width: int = 72) -> str:
    """Format subsection header."""
    return f"\n--- {title} {'-' * max(0, width - len(title) - 5)}"


def replay_session(
    session_id: str,
    data_dir: Path | str | None = None,
    out: TextIO = sys.stdout,
) -> int:
    """Read session artifacts from disk and output chronological audit trail.

    Args:
        session_id: The 32-character session ID.
        data_dir: Base storage directory (defaults to "data" or Settings.DATA_DIR).
        out: Text stream to print audit report to.

    Returns:
        0 on success, 1 on critical session directory failure.
    """
    if data_dir is None:
        try:
            from tabchat.config import get_settings
            base_dir = get_settings().DATA_DIR
        except Exception:
            base_dir = Path("data")
    else:
        base_dir = Path(data_dir)

    session_dir = base_dir / "sessions" / session_id

    out.write(_format_header("TABCHAT SESSION AUDIT TRAIL") + "\n")
    out.write(f"Session ID : {session_id}\n")
    out.write(f"Disk Path  : {session_dir.resolve()}\n")

    if not session_dir.exists() or not session_dir.is_dir():
        out.write(f"\n[ERROR] Session directory does not exist: {session_dir}\n")
        out.write("=" * 72 + "\n")
        return 1

    # -------------------------------------------------------------------------
    # 1. Session Initialization
    # -------------------------------------------------------------------------
    out.write(_format_sub_header("1. Session Initialization"))
    history_file = session_dir / "history.jsonl"
    init_timestamp: str | None = None

    if history_file.exists():
        try:
            for line in history_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    first_record = json.loads(line)
                    init_timestamp = first_record.get("timestamp")
                    break
        except Exception:
            pass

    if not init_timestamp:
        try:
            mtime = os.path.getctime(session_dir)
            init_timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception:
            init_timestamp = "Unknown (no timestamps found)"

    out.write(f"\nInitialized At : {init_timestamp}\n")
    out.write(f"Session Status : Active on disk\n")

    # -------------------------------------------------------------------------
    # 2. Dataset Shape and Column Metadata
    # -------------------------------------------------------------------------
    out.write(_format_sub_header("2. Dataset Shape & Column Metadata"))
    card_file = session_dir / "dataset_card.json"
    raw_file = session_dir / "raw.csv"

    if not card_file.exists():
        out.write("\n[WARNING] dataset_card.json missing: No dataset uploaded for this session.\n")
    else:
        try:
            card_data = json.loads(card_file.read_text(encoding="utf-8"))
            row_cnt = card_data.get("row_count", "N/A")
            col_cnt = card_data.get("col_count", "N/A")
            raw_size = f"{raw_file.stat().st_size:,} bytes" if raw_file.exists() else "missing raw.csv"
            out.write(f"\nDimensions : {row_cnt} rows × {col_cnt} columns ({raw_size})\n")

            col_names = card_data.get("column_names", [])
            col_types = card_data.get("inferred_types", {})
            null_counts = card_data.get("null_counts", {})
            unique_counts = card_data.get("unique_counts", {})

            out.write("\n  #  | Column Name                    | Type         | Nulls | Uniques\n")
            out.write("  ---|--------------------------------|--------------|-------|--------\n")
            for idx, cname in enumerate(col_names, start=1):
                ctype = col_types.get(cname, "unknown")
                cnull = str(null_counts.get(cname, 0))
                cuniq = str(unique_counts.get(cname, 0))
                out.write(f"  {idx:02d} | {cname:<30} | {ctype:<12} | {cnull:<5} | {cuniq:<7}\n")
        except json.JSONDecodeError as exc:
            out.write(f"\n[WARNING] Corrupted JSON in dataset_card.json: {exc}\n")
        except Exception as exc:
            out.write(f"\n[WARNING] Failed reading dataset card: {exc}\n")

    # -------------------------------------------------------------------------
    # 3. Chat Turns
    # -------------------------------------------------------------------------
    out.write(_format_sub_header("3. Conversational Chat Turns"))
    if not history_file.exists():
        out.write("\n[WARNING] history.jsonl missing: No chat history recorded for this session.\n")
    else:
        try:
            history_lines = [l for l in history_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not history_lines:
                out.write("\n(No chat messages recorded)\n")
            else:
                user_turn = 1
                for line_idx, line in enumerate(history_lines, start=1):
                    try:
                        record = json.loads(line)
                        role = record.get("role", "unknown").upper()
                        ts = record.get("timestamp", "")
                        content = record.get("content", "").strip()

                        if role == "USER":
                            out.write(f"\n[Turn {user_turn} - USER] ({ts})\n")
                            out.write(f"  > {content}\n")
                            user_turn += 1
                        elif role == "ASSISTANT":
                            out.write(f"[ASSISTANT]\n")
                            # Indent content
                            indented = "\n".join(f"  {l}" for l in content.splitlines())
                            out.write(f"{indented}\n")
                        else:
                            out.write(f"[{role}]: {content}\n")
                    except json.JSONDecodeError as exc:
                        out.write(f"\n[WARNING] Corrupted record at history.jsonl line {line_idx}: {exc}\n")
        except Exception as exc:
            out.write(f"\n[WARNING] Failed reading history.jsonl: {exc}\n")

    # -------------------------------------------------------------------------
    # 4. Generated Job Spec
    # -------------------------------------------------------------------------
    out.write(_format_sub_header("4. Analytical Job Spec"))
    spec_file = session_dir / "job_spec.json"

    if not spec_file.exists():
        out.write("\n[WARNING] job_spec.json missing: JobSpec has not been generated for this session.\n")
    else:
        try:
            spec_data = json.loads(spec_file.read_text(encoding="utf-8"))
            out.write(f"\nTask Type       : {spec_data.get('task_type')}\n")
            out.write(f"Target Column   : {spec_data.get('target_column')}\n")
            out.write(f"Feature Columns : {', '.join(spec_data.get('feature_columns', []))}\n")
            excluded = spec_data.get("excluded_columns", [])
            out.write(f"Excluded Columns: {', '.join(excluded) if excluded else '(None)'}\n")
            out.write(f"Split Strategy  : {spec_data.get('split_strategy')} (holdout={spec_data.get('holdout_fraction', 0.2):.0%}, seed={spec_data.get('random_seed', 42)})\n")
            if spec_data.get("split_column"):
                out.write(f"Split Column    : {spec_data.get('split_column')}\n")
            out.write(f"Evaluation Metric: {spec_data.get('eval_metric')}\n")
            out.write(f"Rationale       : {spec_data.get('rationale')}\n")
            out.write("\nCanonical JobSpec JSON:\n")
            out.write(json.dumps(spec_data, indent=2) + "\n")
        except json.JSONDecodeError as exc:
            out.write(f"\n[WARNING] Corrupted JSON in job_spec.json: {exc}\n")
        except Exception as exc:
            out.write(f"\n[WARNING] Failed reading job spec: {exc}\n")

    # -------------------------------------------------------------------------
    # 5. TabPFN Execution Metrics vs Baseline
    # -------------------------------------------------------------------------
    out.write(_format_sub_header("5. Execution Metrics vs Baseline"))
    results_file = session_dir / "results.json"
    latest_narration: str | None = None

    if not results_file.exists():
        out.write("\n[WARNING] results.json missing: TabPFN job has not been executed yet.\n")
    else:
        try:
            results_data = json.loads(results_file.read_text(encoding="utf-8"))
            train_sz = results_data.get("train_size", "N/A")
            hold_sz = results_data.get("holdout_size", "N/A")
            out.write(f"\nSample Partitioning: Train = {train_sz} rows | Holdout = {hold_sz} rows\n")

            base_m = results_data.get("baseline_metrics", {})
            model_m = results_data.get("model_metrics", {})
            delta_m = results_data.get("delta", {})

            all_metrics = sorted(set(list(base_m.keys()) + list(model_m.keys())))
            out.write("\n  Metric        | Baseline   | TabPFN-3.5 | Delta (Lift)\n")
            out.write("  --------------|------------|------------|-------------\n")
            for m in all_metrics:
                b_val = f"{base_m.get(m):.4f}" if isinstance(base_m.get(m), (int, float)) else "N/A"
                m_val = f"{model_m.get(m):.4f}" if isinstance(model_m.get(m), (int, float)) else "N/A"
                d_num = delta_m.get(m)
                if isinstance(d_num, (int, float)):
                    sign = "+" if d_num > 0 else ""
                    d_val = f"{sign}{d_num:.4f}"
                else:
                    d_val = "N/A"
                out.write(f"  {m:<13} | {b_val:<10} | {m_val:<10} | {d_val:<11}\n")

            warnings = results_data.get("warnings", [])
            if warnings:
                out.write("\nExecution Warnings:\n")
                for w in warnings:
                    out.write(f"  ⚠️  {w}\n")

            preds = results_data.get("sample_predictions", [])
            if preds:
                out.write(f"\nSample Predictions Preview (First {len(preds)} holdout rows):\n")
                out.write("  Row Index | Actual     | Predicted  | Confidence / Quantiles\n")
                out.write("  ----------|------------|------------|------------------------\n")
                for p in preds:
                    rid = p.get("row_index", "N/A")
                    act = str(p.get("actual", "N/A"))
                    prd = str(p.get("predicted", "N/A"))
                    if "probabilities" in p:
                        conf = f"Probs: {p['probabilities']}"
                    elif "quantiles" in p:
                        q = p["quantiles"]
                        conf = f"Q10: {q.get('q10')}, Q50: {q.get('q50')}, Q90: {q.get('q90')}"
                    else:
                        conf = "-"
                    out.write(f"  {rid:<9} | {act:<10} | {prd:<10} | {conf}\n")

        except json.JSONDecodeError as exc:
            out.write(f"\n[WARNING] Corrupted JSON in results.json: {exc}\n")
        except Exception as exc:
            out.write(f"\n[WARNING] Failed reading results.json: {exc}\n")

    # -------------------------------------------------------------------------
    # 6. Final Narration
    # -------------------------------------------------------------------------
    out.write(_format_sub_header("6. Final Calibrated Narration"))
    # Extract narration from last assistant message or history
    narration_found = False
    if history_file.exists():
        try:
            records = [json.loads(l) for l in history_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            for r in reversed(records):
                if r.get("role") == "assistant":
                    c = r.get("content", "").strip()
                    # Check if this is results narration (mentions accuracy, baseline, lift, holdout, or TabPFN)
                    if any(term in c.lower() for term in ("accuracy", "baseline", "lift", "holdout", "tabpfn", "rmse", "r2")):
                        out.write(f"\n{c}\n")
                        narration_found = True
                        break
        except Exception:
            pass

    if not narration_found:
        out.write("\n(No final execution narration recorded for this session)\n")

    out.write("\n" + "=" * 72 + "\n")
    return 0


def main() -> None:
    """CLI entrypoint for tabchat replay utility."""
    parser = argparse.ArgumentParser(
        prog="tabchat.replay",
        description="Print structured chronological audit trail of a tabchat session from disk.",
    )
    parser.add_argument("session_id", help="Session ID to inspect and replay")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Base data directory (defaults to DATA_DIR from settings or 'data')",
    )

    args = parser.parse_args()
    code = replay_session(session_id=args.session_id, data_dir=args.data_dir)
    sys.exit(code)


if __name__ == "__main__":
    main()
