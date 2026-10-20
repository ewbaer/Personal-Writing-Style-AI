"""Prepare Kaggle AI_Human.csv for the local detector. Run from the repo root."""
import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
import pandas as pd


def prepare(source, output, per_class=10000):
    if source.resolve() == output.resolve():
        raise ValueError('Input and output must be different files.')
    if output.exists():
        raise FileExistsError(f'{output} already exists. Rename it before running again.')
    seen = {}  # Only hashes/labels for the full file, not every full passage.
    conflicts = set()
    samples = {0: [], 1: []}
    stats = Counter()
    for chunk in pd.read_csv(source, usecols=['text', 'generated'], chunksize=5000,
                             dtype=str, keep_default_na=False):
        for text, raw_label in zip(chunk.text, chunk.generated):
            stats['rows_read'] += 1
            text = text.strip()
            try:
                numeric = float(raw_label)
            except ValueError:
                stats['invalid_label'] += 1
                continue
            if numeric not in (0.0, 1.0):
                stats['invalid_label'] += 1
                continue
            label = int(numeric)
            words = text.split()
            if not 50 <= len(words) <= 1000 or len(text) > 30000:
                stats['empty_or_outside_length_limits'] += 1
                continue
            normalized = ' '.join(words).lower()
            digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            if digest in seen:
                stats['duplicate_rows'] += 1
                if seen[digest] != label:
                    conflicts.add(digest)
                continue
            seen[digest] = label
            stats[f'unique_label_{label}'] += 1
            # Hash-based sampling covers the entire file, not just its first rows.
            priority = int(hashlib.sha256(('sample42:' + digest).encode()).hexdigest(), 16)
            item = (-priority, digest, text)
            heap = samples[label]
            if len(heap) < per_class:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        print(f"Read {stats['rows_read']:,} rows...", flush=True)
    selected = {label: sorted([item for item in items if item[1] not in conflicts], reverse=True)
                for label, items in samples.items()}
    count = min(len(selected[0]), len(selected[1]))
    if count < 100:
        raise ValueError(f'Only {count} usable rows per class. Need at least 100 for the pilot.')
    rows = []
    for label in (0, 1):
        for _, digest, text in selected[label][:count]:
            rows.append({'text': text, 'label': 'ai' if label else 'human',
                         'group_id': 'kaggle_' + digest})
    result = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding='utf-8')
    report = dict(stats)
    report.update({'conflicting_texts_excluded': len(conflicts), 'saved_human': count,
                   'saved_ai': count, 'input': str(source), 'output': str(output),
                   'label_mapping': {'0': 'human', '1': 'ai'},
                   'limitations': 'Source labels preserved, not independently verified. Exact normalized duplicates checked; no author/prompt IDs supplied. Near-duplicates and related essays may remain across train/test. Review samples and use independent LinkedIn evaluation.'})
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nSaved {output}: {count:,} human + {count:,} AI passages.')
    print('This is a mechanically cleaned pilot, not a manually reviewed or validated dataset.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'AI_Human.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'detector_labeled.csv')
    parser.add_argument('--per-class', type=int, default=10000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.input = args.input if args.input.is_absolute() else root / args.input
    args.output = args.output if args.output.is_absolute() else root / args.output
    if args.per_class < 100:
        parser.error('--per-class must be at least 100')
    prepare(args.input, args.output, args.per_class)
