#!/usr/bin/env python3
"""Validate that all three official part_05 sources opened and exported samples."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--expected-source-indices", nargs="+", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.source_report)
    if not report_path.is_file():
        raise FileNotFoundError(f"Source report not found: {report_path}")

    reports = []
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number} in the source report is not valid JSON") from exc

    by_index = {int(report["source_index"]): report for report in reports}
    expected = list(args.expected_source_indices)
    missing = [index for index in expected if index not in by_index]
    if missing:
        raise ValueError(f"Source report is missing source_index: {missing}")

    failed = []
    for index in expected:
        report = by_index[index]
        status = report.get("status")
        exported = int(report.get("exported", 0))
        resume_skipped = int(report.get("resume_skipped", 0))
        print(
            f"source_index={index} status={status} exported={exported} "
            f"resume_skipped={resume_skipped} "
            f"lmdb={report.get('lmdb_spec')} message={report.get('message', '')}"
        )
        if status != "ok" or exported + resume_skipped <= 0:
            failed.append(index)

    if failed:
        raise ValueError(f"The following official sources did not complete successfully: {failed}")

    print("part_05 three-source status validation passed")


if __name__ == "__main__":
    main()
