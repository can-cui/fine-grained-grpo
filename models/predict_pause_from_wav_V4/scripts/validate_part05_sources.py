#!/usr/bin/env python3
"""校验 part_05 的三个正式来源均已成功打开并导出样本。"""

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
        raise FileNotFoundError(f"找不到来源报告：{report_path}")

    reports = []
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"来源报告第 {line_number} 行不是合法 JSON") from exc

    by_index = {int(report["source_index"]): report for report in reports}
    expected = list(args.expected_source_indices)
    missing = [index for index in expected if index not in by_index]
    if missing:
        raise ValueError(f"来源报告缺少 source_index：{missing}")

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
        raise ValueError(f"以下正式来源没有成功完成：{failed}")

    print("part_05 三来源状态校验通过")


if __name__ == "__main__":
    main()
