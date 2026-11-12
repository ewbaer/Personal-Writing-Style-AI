"""Evaluate the deployed passage detector on a held-out CSV: label,text."""
import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, roc_auc_score

from src.ai_detector import AIDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--cutoff", type=float, default=50, help="Evaluation cutoff from 0 to 100; does not change the app")
    args = parser.parse_args()
    if not 0 <= args.cutoff <= 100:
        parser.error("Cutoff must be between 0 and 100.")

    detector = AIDetector()
    labels, scores, seen = [], [], set()
    try:
        file = args.samples.open(newline="", encoding="utf-8-sig")
    except OSError as error:
        parser.error(str(error))
    with file:
        reader = csv.DictReader(file, strict=True)
        if not {"label", "text"}.issubset(reader.fieldnames or []):
            parser.error("CSV must have label and text columns.")
        for number, row in enumerate(reader, 2):
            if None in row:
                parser.error(f"Record {number}: extra CSV fields. Enclose text containing commas in quotes.")
            label = (row["label"] or "").strip().lower()
            text = row["text"] or ""
            if label not in {"human", "ai"}:
                parser.error(f"Record {number}: label must be human or ai.")
            key = " ".join(text.lower().split())
            if key in seen:
                parser.error(f"Record {number}: duplicate sample.")
            seen.add(key)
            result = detector.score(text)
            if not result["available"]:
                parser.error(f"Record {number}: {result['message']}")
            labels.append(int(label == "ai"))
            scores.append(result["score"])
    if set(labels) != {0, 1}:
        parser.error("Include both human and AI samples.")
    predictions = [int(score >= args.cutoff) for score in scores]
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    print(json.dumps({
        "samples": len(labels),
        "model_version": detector.bundle["version"],
        "cutoff": args.cutoff,
        "human_flagged": fp,
        "human_total": tn + fp,
        "ai_missed": fn,
        "ai_total": tp + fn,
        "human_false_positive_rate": round(fp / (tn + fp), 4),
        "ai_detection_rate": round(tp / (tp + fn), 4),
        "roc_auc": round(roc_auc_score(labels, scores), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
