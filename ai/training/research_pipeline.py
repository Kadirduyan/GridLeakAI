"""Yeniden üretilebilir eğitim-içi araştırma: kat-içi özellik seçimi, Optuna, SMOTE.

Bu yeni araştırma uygulaması geçmiş Optuna oturumunun birebir kaydı değildir.
Final v3 dosyalarını değiştirmez ve test hedeflerini okumaz. Seçim/tuning OOF
sonuçları geliştirme skorudur; bağımsız genelleme iddiası için kullanılmaz.
"""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ai.features.prepare_data import features49
from ai.features.v3_features import load_raw_for_ids, extract_subscriber_features, FoldContextTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_folds(raw_path, split_path, output, folds=3, limit=None, seed=42):
    """Test tüketimleri/etiketleri kullanılmaz. Bağlam yalnız tr indekslerine fit edilir."""
    split = pd.read_csv(split_path, dtype={'CONS_NO': str})
    if not split.CONS_NO.is_unique or not set(split.grup).issubset({'egitim', 'test'}):
        raise ValueError('Geçersiz bölme listesi.')
    ids = split.loc[split.grup == 'egitim', 'CONS_NO'].tolist()
    wanted = set(ids)
    # Dosya taranır fakat yalnız eğitim satırlarının etiketleri tutulur.
    labels = pd.concat([c[c.CONS_NO.isin(wanted)] for c in pd.read_csv(raw_path, usecols=['CONS_NO', 'FLAG'], dtype={'CONS_NO': str}, chunksize=4000)]).set_index('CONS_NO').FLAG.loc[ids]
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError('Eğitim etiketleri geçersiz.')
    if limit and limit < len(ids):
        ids, _ = train_test_split(ids, train_size=limit, stratify=labels, random_state=seed)
        labels = labels.loc[ids]
    arr, dates = load_raw_for_ids(ids, raw_path=Path(raw_path))
    base, quality = features49(pd.DataFrame(arr, index=pd.Index(ids, name='CONS_NO'), columns=dates))
    valid = quality.dolu_gun_sayisi.to_numpy() > 0
    arr, base, labels = arr[valid], base.loc[valid], labels.loc[valid]
    sub = extract_subscriber_features(arr, dates, base, include_history=True)
    y = labels.to_numpy(int)
    if np.bincount(y).min() < folds:
        raise ValueError('Her sınıfta kat sayısı kadar örnek olmalı.')
    cache = Path(output)/'folds'
    cache.mkdir(parents=True, exist_ok=False)
    audit = []
    for k, (tr, va) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(sub, y)):
        ctx = FoldContextTransformer(include_history=True, random_state=seed).fit(arr[tr], dates, sub.iloc[tr])
        def transform(ix):
            return pd.concat([sub.iloc[ix], ctx.transform(arr[ix], sub.iloc[ix])], axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)
        train, val = transform(tr), transform(va)
        assert not set(train.index) & set(val.index)
        joblib.dump({'train': train, 'validation': val, 'y_train': y[tr], 'y_validation': y[va], 'validation_positions': va}, cache/f'{k}.joblib', compress=3)
        audit.append({'fold': k+1, 'train_ids': train.index.tolist(), 'validation_ids': val.index.tolist(), 'context_fit_ids': train.index.tolist()})
        print(f'Kat {k+1}/{folds}: {len(tr)} öğrenme, {len(va)} doğrulama, {train.shape[1]} özellik', flush=True)
    pd.DataFrame({'CONS_NO': base.index, 'FLAG': y}).to_csv(Path(output)/'training_rows.csv', index=False)
    save_json(Path(output)/'fold_audit.json', audit)
    save_json(Path(output)/'manifest.json', {'seed': seed, 'folds': folds, 'sample_limit': limit, 'rows': len(y), 'features': train.columns.tolist(), 'split_sha256': hashlib.sha256(Path(split_path).read_bytes()).hexdigest(), 'test_used': False, 'note': 'Yeni geliştirme deneyi; final v3 sonuçlarının yeniden üretildiği iddia edilmez.', 'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'optuna', 'imbalanced-learn']}})


def fold_data(output):
    for file in sorted((Path(output)/'folds').glob('*.joblib')):
        yield joblib.load(file)


def metrics(y, score):
    p, r, t = precision_recall_curve(y, score)
    threshold = float(t[np.argmax(2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12))])
    pred = score >= threshold
    return {'AP': float(average_precision_score(y, score)), 'F1': float(f1_score(y, pred)), 'Precision': float(precision_score(y, pred, zero_division=0)), 'Recall': float(recall_score(y, pred)), 'FP': int(((y == 0)&pred).sum()), 'FN': int(((y == 1)&~pred).sum()), 'threshold': threshold, 'threshold_scope': 'Aynı geliştirme OOF üzerinde seçildi; bağımsız F1 değildir.'}


def build_model(family, params=None, smoke=False, smote=False):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from sklearn.impute import SimpleImputer
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    trees = 12 if smoke else 400
    defaults = {
        'lgbm': dict(n_estimators=trees, learning_rate=.04, num_leaves=31, min_child_samples=20, subsample_freq=1, random_state=42, n_jobs=2, verbosity=-1),
        'xgboost': dict(n_estimators=trees, learning_rate=.04, max_depth=5, tree_method='hist', random_state=42, n_jobs=2, eval_metric='logloss'),
        'catboost': dict(iterations=trees, learning_rate=.04, depth=5, random_seed=42, thread_count=2, verbose=False, allow_writing_files=False),
        'mlp': dict(hidden_layer_sizes=(32,) if smoke else (128,64), max_iter=3 if smoke else 100, early_stopping=True, random_state=42),
        'lr': dict(C=.5, max_iter=500, random_state=42),
    }
    constructors = {'lgbm': LGBMClassifier, 'xgboost': XGBClassifier, 'catboost': CatBoostClassifier, 'mlp': MLPClassifier, 'lr': LogisticRegression}
    model = constructors[family](**(defaults[family] | (params or {})))
    # Tüm ailelerde aynı doldurma, SMOTE için aynı ölçekleme: karşılaştırmada
    # yalnız SMOTE adımı değişir. Dönüşümler fit sırasında yalnız öğrenme katındadır.
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    steps += [('scale', QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42) if family == 'mlp' else StandardScaler())]
    if smote:
        steps += [('smote', SMOTE(sampling_strategy=.5, k_neighbors=3, random_state=42))]
    return Pipeline(steps+[('model', model)])


def evaluate(output, family, params, columns, smoke=False, smote=False):
    y = pd.read_csv(Path(output)/'training_rows.csv').FLAG.to_numpy(int)
    score = np.full(len(y), np.nan)
    for f in fold_data(output):
        model = build_model(family, params, smoke, smote)
        model.fit(f['train'][columns], f['y_train'])
        score[f['validation_positions']] = model.predict_proba(f['validation'][columns])[:, 1]
    if not np.isfinite(score).all():
        raise ValueError('Eksik/sonsuz OOF tahmini.')
    return metrics(y, score), score


def select_features(output, top_k=160, repeats=2, smoke=False):
    """Doğrulama katında LightGBM TreeSHAP ve AP permutation; yalnız eğitim OOF."""
    from sklearn.inspection import permutation_importance
    from lightgbm import LGBMClassifier
    rows = []
    for k, f in enumerate(fold_data(output)):
        model = LGBMClassifier(n_estimators=12 if smoke else 400, learning_rate=.04, num_leaves=31, random_state=42, n_jobs=2, verbosity=-1)
        model.fit(f['train'], f['y_train'])
        val = f['validation']
        contrib = model.booster_.predict(val, pred_contrib=True)[:, :-1]
        perm = permutation_importance(model, val, f['y_validation'], scoring='average_precision', n_repeats=repeats, random_state=42, n_jobs=1)
        rows.append(pd.DataFrame({'feature': val.columns, 'fold': k+1, 'mean_abs_shap': np.abs(contrib).mean(axis=0), 'permutation_ap': perm.importances_mean}))
    pd.concat(rows).to_csv(Path(output)/'selection_by_fold.csv', index=False)
    ranking = pd.concat(rows).groupby('feature')[['mean_abs_shap', 'permutation_ap']].mean()
    ranking['eligible'] = ranking.permutation_ap >= 0
    ranking['rank_score'] = ranking.mean_abs_shap.rank(pct=True)+ranking.permutation_ap.rank(pct=True)
    ranking = ranking.sort_values(['eligible','rank_score'], ascending=False)
    # Negatif katkılı özellikleri sırf 160 sayısına ulaşmak için geri ekleme.
    selected = ranking[ranking.eligible].head(top_k).index.tolist()
    if not selected:
        raise ValueError('Pozitif/sıfır permutation katkılı özellik bulunamadı.')
    ranking.to_csv(Path(output)/'feature_ranking.csv')
    save_json(Path(output)/'selected_features.json', {'features': selected, 'requested': top_k, 'actual': len(selected), 'note': 'Yeni seçim; eski kilitli 160 listesi değiştirilmez.'})
    rows_eval = []
    for label, cols in [('all_candidates', ranking.index.tolist()), ('selected', selected)]:
        m, _ = evaluate(output, 'lgbm', {}, cols, smoke)
        rows_eval.append({'experiment': label, 'features': len(cols), **m})
    pd.DataFrame(rows_eval).to_csv(Path(output)/'selection_comparison.csv', index=False)
    return selected


def tune(output, columns, trials=20, smoke=False):
    import optuna
    rows = []
    best = {}
    for family in ['lgbm', 'xgboost', 'catboost', 'mlp']:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        def objective(trial):
            if family == 'mlp':
                params = {'hidden_layer_sizes': (trial.suggest_categorical('width', [16,32] if smoke else [64,128,256]),), 'alpha': trial.suggest_float('alpha', 1e-4, .1, log=True), 'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, .01, log=True)}
            else:
                params = {'learning_rate': trial.suggest_float('learning_rate', .015, .12, log=True)}
                params['iterations' if family == 'catboost' else 'n_estimators'] = trial.suggest_categorical('trees', [8,12] if smoke else [400,800,1400])
                if family == 'lgbm':
                    params.update(num_leaves=trial.suggest_int('num_leaves', 16,64), min_child_samples=trial.suggest_int('min_child_samples', 10,80), reg_lambda=trial.suggest_float('reg_lambda', .01,10,log=True))
                elif family == 'xgboost':
                    params.update(max_depth=trial.suggest_int('max_depth',3,8), min_child_weight=trial.suggest_float('min_child_weight',1,15), reg_lambda=trial.suggest_float('reg_lambda',.1,10,log=True))
                else:
                    params.update(depth=trial.suggest_int('depth',3,8), l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1,10))
            m, score = evaluate(output, family, params, columns, smoke)
            trial.set_user_attr('model_params', params)
            trial.set_user_attr('metrics', m)
            np.save(Path(output)/f'{family}_trial_{trial.number}_oof.npy', score)
            rows.append({'model': family, 'trial': trial.number, **m})
            pd.DataFrame(rows).to_csv(Path(output)/'tuning_metrics.csv', index=False)
            return m['AP']
        study.optimize(objective, n_trials=trials)
        study.trials_dataframe().to_csv(Path(output)/f'{family}_optuna_trials.csv', index=False)
        best[family] = study.best_trial.user_attrs['model_params']
        save_json(Path(output)/'best_params.json', best)
    return best


def compare_smote(output, columns, smoke=False):
    rows = []
    for family in ['lr','lgbm','xgboost']:
        for enabled in [False, True]:
            m, scores = evaluate(output, family, {}, columns, smoke, enabled)
            rows.append({'model': family, 'SMOTE': enabled, **m})
            np.save(Path(output)/f'{family}_smote_{enabled}_oof.npy', scores)
    pd.DataFrame(rows).to_csv(Path(output)/'smote_comparison.csv', index=False)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=ROOT/'data/raw/data.csv')
    p.add_argument('--split', type=Path, default=ROOT/'data/splits/abone_bolunmesi.csv')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['all','select','tune','smote'], default='all')
    p.add_argument('--features', type=Path, help='Sabit özellik listesi JSON; mevcut kilit dosyası da okunabilir.')
    p.add_argument('--limit', type=int)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--top-k', type=int, default=160)
    p.add_argument('--smoke', action='store_true', help='Kısa yazılım testi; model başarısı olarak raporlanmaz.')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Var olan deney üzerine yazılmaz; yeni --output seçin.')
    a.output.mkdir(parents=True)
    prepare_folds(a.raw, a.split, a.output, a.folds, a.limit)
    columns = json.loads(a.features.read_text(encoding='utf-8'))['features'] if a.features else json.loads((a.output/'manifest.json').read_text(encoding='utf-8'))['features']
    if a.mode in ['all','select']:
        columns = select_features(a.output, a.top_k, 1 if a.smoke else 2, a.smoke)
    if a.mode in ['all','tune']:
        tune(a.output, columns, a.trials, a.smoke)
    if a.mode in ['all','smote']:
        print(compare_smote(a.output, columns, a.smoke))
    print('Tamamlandı:', a.output, '| Final model ve test kümesi değiştirilmedi.')


if __name__ == '__main__':
    main()
