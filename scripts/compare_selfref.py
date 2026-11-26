"""Compare Native one-shot runs to frozen Dice references."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathology_trainfree.selfref_config import branch_name


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row["dataset"], row["support_id"], row["query_id"], branch_name(row["branch"].lower())


def reference_rows(path, kind, actual):
    if kind in ("kcenter", "train-oof"):
        name = "per_pair.csv" if kind == "kcenter" else "oof_per_pair.csv"
        rows = [row for row in read_csv(path / name) if row["config"] == "S1" and float(row["rho"]) == .125]
        return {key(row): float(row["dice"]) for row in rows}
    if kind == "fixed-budget":
        rows = [row for ds in ("GlaS", "CRAG") for row in read_csv(path / f"per_pair_{ds}.csv")
                if row["method"] == "C1-2048"]
        return {key(row): float(row["dice"]) for row in rows}
    result = {}
    for row in actual:
        ds, sid, qid, branch = key(row)
        source = path / ds / sid / f"{qid}.json"
        if not source.is_file():
            raise FileNotFoundError(f"No historical N3 result: {source}")
        result[ds, sid, qid, branch] = float(json.loads(source.read_text())["dice"])
    return result


def stats(rows, field):
    supports = defaultdict(list)
    for row in rows:
        supports[row["support_id"]].append(row[field])
    values = sorted(map(mean, supports.values()))
    return mean(r[field] for r in rows), mean(values[:math.ceil(.1 * len(values))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    old_gates = {"s1": "kcenter", "c1": "fixed-budget", "n3": "final-layer", "oof": "train-oof"}
    expected_counts = {"kcenter": 27440, "fixed-budget": 1200, "final-layer": 600, "train-oof": 516}
    parser.add_argument("--gate", choices=tuple(expected_counts), type=lambda name: old_gates.get(name, name),
                        required=True)
    args = parser.parse_args()
    actual = read_csv(args.run / "per_pair.csv")
    lookup = reference_rows(args.reference, args.gate, actual)
    actual_keys = {key(r) for r in actual}
    if len(actual_keys) != len(actual) or len(actual) != expected_counts[args.gate]:
        raise ValueError(f"Incorrect pair coverage for {args.gate}: {len(actual)} rows")
    if args.gate in ("kcenter", "train-oof") and actual_keys != lookup.keys():
        raise ValueError("Pair coverage differs from reference")
    rows = []
    groups = defaultdict(list)
    for row in actual:
        ds, sid, qid, branch = key(row)
        dice, ref = float(row["dice"]), lookup[ds, sid, qid, branch]
        item = dict(dataset=ds, support_id=sid, query_id=qid, branch=branch, dice=dice,
                    reference_dice=ref, absolute_error=abs(dice - ref),
                    fg_bank_size=row["fg_bank_size"], bg_bank_size=row["bg_bank_size"])
        rows.append(item)
        groups[ds, branch].append(item)
    with (args.run / "parity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if args.gate == "kcenter":
        index = lambda row: (row["dataset"], branch_name(row["branch"].lower()))
        summaries = {index(r): r for r in json.loads((args.run / "summary.json").read_text())}
        reference_summaries = {index(r): r for r in read_csv(args.reference / "T3_test.csv")
                               if r["config"] == "S1" and float(r["rho"]) == .125}
        if summaries.keys() != groups.keys() or reference_summaries.keys() != groups.keys():
            raise ValueError("Summary branch coverage differs from reference")
    results = []
    for (ds, branch), group in groups.items():
        if args.gate == "kcenter":
            got, tail = (float(summaries[ds, branch][field]) for field in ("dice", "support_worst10"))
            ref, ref_tail = (float(reference_summaries[ds, branch][field])
                             for field in ("dice", "support_worst10"))
        else:
            got, tail = stats(group, "dice")
            ref, ref_tail = stats(group, "reference_dice")
        results.append(dict(dataset=ds, branch=branch, rows=len(group),
                            max_pair_difference=max(row["absolute_error"] for row in group),
                            dice=got, reference_dice=ref, dice_difference=abs(got - ref),
                            support_worst10=tail, reference_support_worst10=ref_tail,
                            support_worst10_difference=abs(tail - ref_tail)))
    (args.run / "parity.json").write_text(json.dumps(results, indent=2) + "\n")
    for item in results:
        print(f"{item['dataset']} {item['branch']}: {item['rows']} pairs, max Dice error "
              f"{item['max_pair_difference']:.8g}")
    failed = any(r["max_pair_difference"] > 1e-4 for r in results
                 if args.gate != "train-oof" or r["branch"] == "query_refined")
    if args.gate == "kcenter":
        failed |= any(max(r["dice_difference"], r["support_worst10_difference"]) > 1e-5 for r in results)
    if failed:
        raise SystemExit("Dice parity gate failed. See parity.csv and parity.json.")


if __name__ == "__main__":
    main()
