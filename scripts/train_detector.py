"""Train a first detector from CSV columns text,label,group_id.
Labels: human/ai or 0/1. Related originals/rewrites must share group_id.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]

def prepare(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'text', 'label', 'group_id'}
    if not required.issubset(df.columns):
        raise ValueError('CSV needs text, label, group_id columns. See docs/SETUP.md.')
    df['label'] = df.label.str.strip().str.lower().map({'human': 0, 'ai': 1, '0': 0, '1': 1})
    if df.label.isna().any():
        raise ValueError('Every label must be human, ai, 0, or 1. Do not guess missing labels.')
    df['text'] = df.text.str.strip()
    df['group_id'] = df.group_id.str.strip()
    if (df.group_id == '').any():
        raise ValueError('Every row needs a group_id.')
    df['normalized'] = df.text.str.lower().str.replace(r'\s+', ' ', regex=True)
    conflicts = df.groupby('normalized').label.nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical texts have conflicting labels. Review the dataset.')
    # Reject ambiguous duplicate grouping rather than silently risking split leakage.
    if (df.groupby('normalized').group_id.nunique() > 1).any():
        raise ValueError('Duplicate texts have different group IDs. Merge their related groups first.')
    before = len(df)
    df = df.drop_duplicates('normalized')
    lengths = df.text.str.split().str.len()
    df = df[(lengths >= 50) & (lengths <= 1000) & (df.text.str.len() <= 30000)].reset_index(drop=True)
    print(f'Kept {len(df)} rows; removed {before-len(df)} duplicates or out-of-range lengths.')
    if set(df.label.unique()) != {0, 1} or df.groupby('label').size().min() < 100:
        raise ValueError('Use at least 100 usable rows per label for this pilot. This minimum does not establish reliability.')
    df['label'] = df.label.astype(int)
    # Five group-separated folds: three train, one validation, one final test.
    df['fold'] = -1
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, indices) in enumerate(splitter.split(df.text, df.label, df.group_id)):
        df.loc[indices, 'fold'] = fold
    df['split'] = df.fold.map({0: 'test', 1: 'validation', 2: 'train', 3: 'train', 4: 'train'})
    for name, part in df.groupby('split'):
        if part.label.nunique() != 2:
            raise ValueError(f'{name} lacks one class. Add more independent groups.')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=ROOT / 'data' / 'detector_labeled.csv')
    args = parser.parse_args()
    if not args.csv.is_absolute():
        args.csv = ROOT / args.csv
    df = prepare(args.csv)
    train = df[df.split == 'train']
    model = Pipeline([
        ('words', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)),
        ('classifier', LogisticRegression(max_iter=1500, random_state=42))
    ])
    model.fit(train.text, train.label)
    version = datetime.now(timezone.utc).isoformat()
    report = {'version': version, 'threshold': 0.5,
              'warning': 'Pilot only. Scores are not independently calibrated. Test on separate LinkedIn data before making claims.',
              'dataset_sha256': hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              'counts': {name: {str(k): int(v) for k, v in part.label.value_counts().items()}
                         for name, part in df.groupby('split')}}
    for name in ('validation', 'test'):
        part = df[df.split == name]
        scores = model.predict_proba(part.text)[:, list(model.classes_).index(1)]
        predicted = (scores >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(part.label, predicted, labels=[0, 1]).ravel()
        report[name] = {'classification': classification_report(part.label, predicted, labels=[0, 1], target_names=['human', 'ai'], output_dict=True, zero_division=0),
                        'human_false_positive_rate': float(fp / (tn + fp)),
                        'ai_recall': float(tp / (tp + fn)),
                        'roc_auc': float(roc_auc_score(part.label, scores)),
                        'brier_score': float(brier_score_loss(part.label, scores)),
                        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}
    model_dir = ROOT / 'models'
    result_dir = ROOT / 'results' / 'detector'
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    # The saved pipeline includes the vocabulary, fitted on TRAIN ONLY.
    joblib.dump({'model': model, 'version': version}, model_dir / 'ai_detector.joblib')
    (result_dir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    manifest = df[['group_id', 'label', 'split']].copy()
    manifest['text_sha256'] = df.normalized.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    manifest.to_csv(result_dir / 'split_manifest.csv', index=False)
    print(json.dumps(report, indent=2))
    print('Saved detector. Restart the app to load it. Check results/detector/metrics.json for this training run.')

if __name__ == '__main__':
    main()
