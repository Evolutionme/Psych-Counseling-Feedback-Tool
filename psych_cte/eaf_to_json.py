"""Convert ELAN .eaf annotations to structured JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .utils import dump_json, ensure_dir


def parse_time_slots(root):
    slots = {}
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        return slots
    for slot in time_order.findall("TIME_SLOT"):
        slot_id = slot.attrib.get("TIME_SLOT_ID")
        value = slot.attrib.get("TIME_VALUE")
        if slot_id and value is not None:
            try:
                slots[slot_id] = int(float(value))
            except ValueError:
                slots[slot_id] = 0
    return slots


def parse_tiers(root, time_slots):
    tiers = {}
    for tier in root.findall("TIER"):
        tier_name = tier.attrib.get("TIER_ID", "tier")
        participant = tier.attrib.get("PARTICIPANT", "")
        ling_type = tier.attrib.get("LINGUISTIC_TYPE_REF", "")
        records = []
        for ann in tier.findall("ANNOTATION"):
            align = ann.find("ALIGNABLE_ANNOTATION")
            ref = ann.find("REF_ANNOTATION")
            if align is not None:
                ann_id = align.attrib.get("ANNOTATION_ID", "")
                start_ref = align.attrib.get("TIME_SLOT_REF1")
                end_ref = align.attrib.get("TIME_SLOT_REF2")
                records.append(
                    {
                        "tier_id": tier_name,
                        "participant": participant,
                        "linguistic_type": ling_type,
                        "annotation_id": ann_id,
                        "ref": "",
                        "start_ms": time_slots.get(start_ref, 0),
                        "end_ms": time_slots.get(end_ref, 0),
                        "value": (align.findtext("ANNOTATION_VALUE") or "").strip(),
                    }
                )
            elif ref is not None:
                records.append(
                    {
                        "tier_id": tier_name,
                        "participant": participant,
                        "linguistic_type": ling_type,
                        "annotation_id": ref.attrib.get("ANNOTATION_ID", ""),
                        "ref": ref.attrib.get("ANNOTATION_REF", ""),
                        "start_ms": None,
                        "end_ms": None,
                        "value": (ref.findtext("ANNOTATION_VALUE") or "").strip(),
                    }
                )
        tiers[tier_name] = records
    return tiers


def overlap(a_start, a_end, b_start, b_end):
    if a_start is None or a_end is None:
        return False
    return max(a_start, b_start) < min(a_end, b_end)


def collect_text(records, start_ms, end_ms):
    items = []
    for rec in records:
        if overlap(rec["start_ms"], rec["end_ms"], start_ms, end_ms):
            items.append(rec)
    items.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    return "".join([item["value"] for item in items if item["value"]]).strip()


def collect_tier_value(tiers, tier_name, unit_record):
    records = tiers.get(tier_name, [])
    if not records:
        return ""
    unit_id = unit_record.get("annotation_id", "")
    start_ms = unit_record.get("start_ms", 0)
    end_ms = unit_record.get("end_ms", 0)
    matches = []
    for rec in records:
        if rec.get("ref") == unit_id:
            matches.append(rec["value"])
        elif overlap(rec.get("start_ms"), rec.get("end_ms"), start_ms, end_ms):
            matches.append(rec["value"])
    return matches[0] if matches else ""


def build_turns(tiers, tier_name, role):
    turns = []
    for idx, rec in enumerate(tiers.get(tier_name, []), 1):
        turns.append(
            {
                "turn_index": idx,
                "role": role,
                "tier_id": tier_name,
                "annotation_id": rec.get("annotation_id", ""),
                "start_ms": rec.get("start_ms"),
                "end_ms": rec.get("end_ms"),
                "text": rec.get("value", ""),
            }
        )
    return turns


def overlap_turns(unit, turns):
    matched = []
    for turn in turns:
        start = turn.get("start_ms")
        end = turn.get("end_ms")
        if start is None or end is None:
            continue
        if overlap(start, end, unit["start_ms"], unit["end_ms"]):
            matched.append(turn["turn_index"])
    return matched


def derive_units(tiers, client_tier, therapist_tier):
    client_records = [r for r in tiers.get(client_tier, []) if r.get("start_ms") is not None]
    therapist_records = [r for r in tiers.get(therapist_tier, []) if r.get("start_ms") is not None]
    turns = []
    for rec in client_records:
        turns.append(("client", rec))
    for rec in therapist_records:
        turns.append(("therapist", rec))
    turns.sort(key=lambda x: (x[1]["start_ms"], x[1]["end_ms"]))

    units = []
    current = None
    for speaker, rec in turns:
        if speaker == "client":
            if current:
                units.append(current)
            current = {
                "annotation_id": rec.get("annotation_id", ""),
                "segment_id": "",
                "start_ms": rec["start_ms"],
                "end_ms": rec["end_ms"],
                "unit_time_text": "",
                "client_text": rec["value"],
                "therapist_text": "",
                "raw_labels": {},
            }
        elif current:
            current["end_ms"] = max(current["end_ms"], rec["end_ms"])
            current["therapist_text"] = current["therapist_text"] + rec["value"]
            units.append(current)
            current = None

    if current:
        units.append(current)
    for idx, unit in enumerate(units, 1):
        unit["segment_id"] = "unit_{:04d}".format(idx)
    return units


def get_label(unit, name):
    return unit.get("raw_labels", {}).get(name, "")


def build_units(tiers, client_tier, therapist_tier, unit_tier):
    client_turns = build_turns(tiers, client_tier, "client")
    therapist_turns = build_turns(tiers, therapist_tier, "therapist")

    if unit_tier and unit_tier in tiers:
        base_units = []
        for idx, rec in enumerate(tiers[unit_tier], 1):
            base_units.append(
                {
                    "annotation_id": rec.get("annotation_id", ""),
                    "segment_id": rec.get("annotation_id", "unit_{:04d}".format(idx)) or "unit_{:04d}".format(idx),
                    "start_ms": rec["start_ms"],
                    "end_ms": rec["end_ms"],
                    "unit_time_text": rec.get("value", ""),
                    "client_text": "",
                    "therapist_text": "",
                    "raw_labels": {},
                }
            )
    else:
        base_units = derive_units(tiers, client_tier, therapist_tier)

    for unit in base_units:
        if client_tier in tiers:
            unit["client_text"] = collect_text(tiers[client_tier], unit["start_ms"], unit["end_ms"])
        if therapist_tier in tiers:
            unit["therapist_text"] = collect_text(tiers[therapist_tier], unit["start_ms"], unit["end_ms"])

        unit["client_turn_indexes"] = overlap_turns(unit, client_turns)
        unit["therapist_turn_indexes"] = overlap_turns(unit, therapist_turns)
        unit["raw_labels"] = {}
        for tier_name in tiers:
            if tier_name in (client_tier, therapist_tier, unit_tier):
                continue
            value = collect_tier_value(tiers, tier_name, unit)
            if value:
                unit["raw_labels"][tier_name] = value

        unit["共情单元起止时间"] = unit.get("unit_time_text", unit.get("segment_id", ""))
        unit["来访者表达区间"] = collect_tier_value(tiers, client_tier, unit)
        unit["咨询师回应区间"] = collect_tier_value(tiers, therapist_tier, unit)
        unit["来访文本"] = unit.get("client_text", "")
        unit["咨询师文本"] = unit.get("therapist_text", "")
        unit["单元CTE分数"] = get_label(unit, "单元CTE分数")
        unit["单元CTE分数依据"] = get_label(unit, "单元CTE分数依据")
        unit["内容情感反映"] = get_label(unit, "内容情感反映")
        unit["接纳确认"] = get_label(unit, "接纳确认")
        unit["促进探索"] = get_label(unit, "促进探索")
        unit["共情阻碍"] = get_label(unit, "共情阻碍")
        unit["共情标签依据"] = get_label(unit, "共情标签依据")
    return base_units


def first_tier_value(tiers, tier_names):
    for tier_name in tier_names:
        records = tiers.get(tier_name, [])
        for rec in records:
            value = rec.get("value", "")
            if value:
                return value
    return ""


def parse_eaf(path, client_tier, therapist_tier, unit_tier):
    tree = ET.parse(path)
    root = tree.getroot()
    time_slots = parse_time_slots(root)
    tiers = parse_tiers(root, time_slots)
    units = build_units(tiers, client_tier, therapist_tier, unit_tier)
    client_turns = build_turns(tiers, client_tier, "client")
    therapist_turns = build_turns(tiers, therapist_tier, "therapist")
    audio_id = Path(path).stem
    audio_cte_score = first_tier_value(tiers, ["CTE总评分", "整体CTE分数", "整段CTE评分"])
    audio_cte_rationale = first_tier_value(tiers, ["CTE总评分依据", "整体评分依据", "整段评分依据"])
    for unit in units:
        unit["CTE总评分"] = audio_cte_score
        unit["CTE总评分依据"] = audio_cte_rationale
    return {
        "audio_id": audio_id,
        "audio_cte_score": audio_cte_score,
        "audio_cte_rationale": audio_cte_rationale,
        "tiers": tiers,
        "client_turns": client_turns,
        "therapist_turns": therapist_turns,
        "segments": units,
        "source_file": str(Path(path).resolve()),
    }


def export_csv(json_data, csv_path):
    rows = []
    for seg in json_data.get("segments", []):
        row = {
            "audio_id": json_data.get("audio_id", ""),
            "segment_id": seg.get("segment_id", ""),
            "共情单元起止时间": seg.get("共情单元起止时间", ""),
            "start_ms": seg.get("start_ms", ""),
            "end_ms": seg.get("end_ms", ""),
            "来访者表达区间": seg.get("来访者表达区间", ""),
            "咨询师回应区间": seg.get("咨询师回应区间", ""),
            "来访文本": seg.get("来访文本", ""),
            "咨询师文本": seg.get("咨询师文本", ""),
            "单元CTE分数": seg.get("单元CTE分数", ""),
            "单元CTE分数依据": seg.get("单元CTE分数依据", ""),
            "内容情感反映": seg.get("内容情感反映", ""),
            "接纳确认": seg.get("接纳确认", ""),
            "促进探索": seg.get("促进探索", ""),
            "共情阻碍": seg.get("共情阻碍", ""),
            "共情标签依据": seg.get("共情标签依据", ""),
            "CTE总评分": seg.get("CTE总评分", json_data.get("audio_cte_score", "")),
            "CTE总评分依据": seg.get("CTE总评分依据", json_data.get("audio_cte_rationale", "")),
            "raw_labels": json.dumps(seg.get("raw_labels", {}), ensure_ascii=False),
        }
        rows.append(row)

    ensure_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert ELAN EAF to structured JSON/CSV.")
    parser.add_argument("eaf", help="Path to .eaf file")
    parser.add_argument("--client-tier", default="来访文本", help="Client speaker tier name")
    parser.add_argument("--therapist-tier", default="咨询师文本", help="Therapist speaker tier name")
    parser.add_argument("--unit-tier", default="共情单元起止时间", help="Unit tier name, optional")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--csv", default="", help="Optional CSV path")
    args = parser.parse_args()

    data = parse_eaf(args.eaf, args.client_tier, args.therapist_tier, args.unit_tier)
    dump_json(args.output, data)
    if args.csv:
        export_csv(data, args.csv)


if __name__ == "__main__":
    main()
