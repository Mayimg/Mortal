#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch TensorBoard logs (.tfevents) から特定の scalar を抽出しCSVに保存する
pandas 不使用版
"""

import argparse
import os
import csv
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

# 取得対象のタグ
TARGET_TAGS = [
    "ach/entropy",
    "ach/policy_loss",
    "ach/ratio_mean",
    "ach/value_loss",
    "ach/ratio_clip_rate",
]

def load_scalars(logdir, tags):
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 10**6}
    )
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag not in available:
            print(f"[WARN] tag {tag} が見つかりません")
            continue
        scalars = ea.Scalars(tag)
        # list of (wall_time, step, value)
        data = []
        seen = {}
        for s in scalars:
            # 同じ step が複数ある場合は最新を採用
            if s.step not in seen or s.wall_time > seen[s.step][0]:
                seen[s.step] = (s.wall_time, s.value)
        for step, (t, v) in sorted(seen.items()):
            data.append((step, t, v))
        result[tag] = data
    return result

def export_csv(data_dict, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 個別CSV
    for tag, data in data_dict.items():
        filename = outdir / f"{tag.replace('/', '_')}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "wall_time", "value"])
            for step, t, v in data:
                writer.writerow([step, t, v])
        print(f"[OK] {filename} に出力しました")

    # ワイド形式（stepごとにまとめる）
    steps = sorted(set(step for d in data_dict.values() for step, _, _ in d))
    wide_filename = outdir / "metrics_wide.csv"
    with open(wide_filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["step"] + list(data_dict.keys())
        writer.writerow(header)
        for step in steps:
            row = [step]
            for tag in data_dict.keys():
                val = next((v for s, _, v in data_dict[tag] if s == step), "")
                row.append(val)
            writer.writerow(row)
    print(f"[OK] {wide_filename} に出力しました")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="TensorBoardのログディレクトリ")
    parser.add_argument("--outdir", default="tb_export", help="CSV出力ディレクトリ")
    args = parser.parse_args()

    data_dict = load_scalars(args.logdir, TARGET_TAGS)
    if not data_dict:
        print("[ERROR] 指定タグのデータが見つかりませんでした")
        return
    export_csv(data_dict, args.outdir)

if __name__ == "__main__":
    main()
