#!/usr/bin/env python3
"""
Simple Parquet viewer.

Examples:
  python3 scripts/view_parquet.py --path my_data/sft_40k_mix/train.parquet
  python3 scripts/view_parquet.py --path data/a.parquet --head 3 --columns prompt,response
  python3 scripts/view_parquet.py --path data/a.parquet --out /tmp/preview.jsonl --head 20
  python3 scripts/view_parquet.py --path big.parquet --out all.jsonl --all
  python3 scripts/view_parquet.py --path my_data/sft_40k_v2/train.parquet -o preview.txt --format txt --head 10
"""

import argparse
import ast
import json
import textwrap
from pathlib import Path
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print(
        "ERROR: missing dependency 'pyarrow'. Install it with:\n"
        "  python3 -m pip install --user pyarrow",
        file=sys.stderr,
    )
    sys.exit(1)


def normalize_value(value):
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def build_records(df, head_n, export_all):
    if export_all:
        slice_df = df
    else:
        slice_df = df.head(head_n)
    preview_records = slice_df.to_dict(orient="records")
    return [{k: normalize_value(v) for k, v in r.items()} for r in preview_records]


def infer_format(out_path: Path) -> str:
    suf = out_path.suffix.lower()
    if suf == ".jsonl":
        return "jsonl"
    if suf == ".json":
        return "json"
    if suf == ".txt":
        return "txt"
    return "jsonl"


def _write_messages_txt_block(
    f,
    sample_idx: int,
    messages,
    sep: str,
    sub: str,
    wrap_width: int,
) -> None:
    if not isinstance(messages, list):
        f.write(f"[Sample {sample_idx}] (invalid messages, not a list)\n{sep}\n\n")
        return
    f.write(f"[Sample {sample_idx}] turns={len(messages)}\n")
    f.write(sub + "\n")
    for t, m in enumerate(messages, 1):
        if not isinstance(m, dict):
            f.write(f"  (Turn {t}) [INVALID MESSAGE]\n\n")
            continue
        role = str(m.get("role", "unknown")).strip()
        content = str(m.get("content", "")).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        f.write(f"  (Turn {t}) role={role}\n")
        f.write("  content:\n")
        if not content:
            f.write("    [EMPTY]\n\n")
            continue
        for line in content.split("\n"):
            wrapped = textwrap.wrap(
                line, width=wrap_width, break_long_words=False, break_on_hyphens=False
            )
            if not wrapped:
                f.write("    \n")
            else:
                for w in wrapped:
                    f.write(f"    {w}\n")
        f.write("\n")
    f.write(sep + "\n\n")


def write_records(path: Path, records, fmt: str, txt_wrap_width: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    elif fmt == "txt":
        sep = "=" * 110
        sub = "-" * 110
        with path.open("w", encoding="utf-8") as f:
            f.write(f"rows: {len(records)}\n{sep}\n\n")
            for i, rec in enumerate(records, 1):
                msgs = rec.get("messages")
                if msgs is not None:
                    _write_messages_txt_block(f, i, msgs, sep, sub, txt_wrap_width)
                else:
                    f.write(f"[Sample {i}] (no messages column; raw row)\n{sub}\n")
                    blob = json.dumps(rec, ensure_ascii=False, indent=2, default=str)
                    for line in blob.split("\n"):
                        for w in textwrap.wrap(
                            line,
                            width=txt_wrap_width,
                            break_long_words=False,
                            break_on_hyphens=False,
                        ) or [""]:
                            f.write(f"  {w}\n")
                    f.write(f"{sep}\n\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(records, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a parquet file quickly.")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to parquet file, e.g. my_data/sft_40k_mix/train.parquet",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="How many rows to preview/export (default: 5). Ignored if --all.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="",
        help="Comma-separated column names to display, e.g. prompt,response",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Hide schema output.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="",
        help="Write preview rows to this file. .jsonl / .json / .txt (readable messages).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl", "txt", "auto"],
        default="auto",
        help="Output format when using --out (default: infer from extension; .txt => readable messages).",
    )
    parser.add_argument(
        "--txt-wrap",
        type=int,
        default=100,
        metavar="N",
        help="Line width for --format txt (default: 100).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="export_all",
        help="Export/read all rows (only with --out; can be large).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="With --out: only print one line (row count + path), no schema.",
    )
    parser.add_argument(
        "--also-print",
        action="store_true",
        help="With --out: also print rows to terminal (default: only write file).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.export_all and not args.out:
        print("ERROR: --all requires --out (refusing to dump entire table to terminal).", file=sys.stderr)
        return 5

    file_path = Path(args.path).expanduser()
    if not file_path.exists():
        print(f"ERROR: file not found: {file_path}", file=sys.stderr)
        return 2
    if file_path.suffix.lower() != ".parquet":
        print(f"WARNING: file suffix is not .parquet: {file_path}")

    selected_columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    try:
        parquet_file = pq.ParquetFile(file_path)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: failed to open parquet: {exc}", file=sys.stderr)
        return 3

    if not args.quiet or not args.out:
        print(f"path: {file_path}")
        print(f"rows: {parquet_file.metadata.num_rows}")
        print(f"columns: {parquet_file.metadata.num_columns}")

    if not args.no_schema and not (args.quiet and args.out):
        print("\n=== schema ===")
        print(parquet_file.schema_arrow)

    try:
        table = pq.read_table(file_path, columns=selected_columns or None)
    except Exception as exc:
        print(f"ERROR: failed to read table: {exc}", file=sys.stderr)
        return 4

    df = table.to_pandas()

    if not args.quiet or not args.out:
        if selected_columns:
            print(f"\nselected columns: {selected_columns}")
        else:
            print("\nselected columns: ALL")

    n = max(args.head, 0)
    if args.export_all:
        n_display = len(df)
    else:
        n_display = min(n, len(df)) if n > 0 else 0

    records = build_records(df, n, args.export_all)

    out_path = Path(args.out).expanduser() if args.out else None
    if out_path:
        fmt = args.format if args.format != "auto" else infer_format(out_path)
        write_records(out_path, records, fmt, txt_wrap_width=args.txt_wrap)
        if args.quiet:
            print(f"wrote {len(records)} rows -> {out_path} ({fmt})")
        else:
            print(f"\n=== wrote {len(records)} rows -> {out_path} ({fmt}) ===")

    # 默认：写了文件就不再往终端刷大段 JSON，用编辑器打开 --out 即可
    if out_path and not args.also_print:
        return 0

    print(f"\n=== head({n_display}) ===")
    if n == 0 and not args.export_all:
        print("(skip row preview because --head 0)")
    elif df.empty:
        print("(empty parquet)")
    else:
        for i, normalized in enumerate(records, 1):
            print(f"\n--- row {i} ---")
            print(json.dumps(normalized, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
