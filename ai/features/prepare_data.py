"""Ham SGCC CSV -> sabit abone bölmesi -> 7/49 satır-bazlı özellik.

Mevcut çıktılar üzerine yazmaz. Varsayılan olarak yalnız eğitim grubunu üretir;
test hazırlığı --include-test ile ayrıca istenir, başarı metriği hesaplanmaz.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def features49(consumption):
    """Eksik takvim günleri NaN kalır; sıfır serisini eksik gün keser."""
    x = consumption.copy()
    x.columns = pd.to_datetime(x.columns)
    if x.columns.duplicated().any():
        raise ValueError('Tekrarlanan tarih sütunu.')
    x = x.reindex(columns=pd.date_range('2016-01-01', '2016-10-31'))
    count = x.count(axis=1)
    denominator = count.where(count > 0)
    f = pd.DataFrame(index=x.index)
    f['ortalama_tuketim'] = x.mean(axis=1)
    f['ortanca_tuketim'] = x.median(axis=1)
    f['tuketim_standart_sapma'] = x.std(axis=1)
    f['en_yuksek_tuketim'] = x.max(axis=1)
    f['sifir_gun_orani'] = x.eq(0).sum(axis=1) / denominator
    first, last = x.loc[:, '2016-01-01':'2016-03-31'], x.loc[:, '2016-08-01':'2016-10-31']
    valid = (first.notna().mean(axis=1) >= .8) & (last.notna().mean(axis=1) >= .8)
    early = first.mean(axis=1)
    f['donemsel_degisim_yuzdesi'] = ((last.mean(axis=1)-early)/early.where(early > 0)*100).where(valid)
    run = np.zeros(len(x), dtype=int)
    longest = run.copy()
    for col in x.eq(0).to_numpy().T:
        run = np.where(col, run+1, 0)
        longest = np.maximum(longest, run)
    f['en_uzun_sifir_serisi'] = pd.Series(longest, index=x.index).where(count > 0)
    f['missing_ratio'] = x.isna().mean(axis=1)
    f['variation'] = x.std(axis=1)/(x.mean(axis=1)+1)
    for q in [.1, .25, .75, .9]:
        f[f'quantile_{q}'] = x.quantile(q, axis=1)
    f['iqr'] = f['quantile_0.75']-f['quantile_0.25']
    f['peak_ratio'] = x.max(axis=1)/(x.mean(axis=1)+1)
    diff = x.diff(axis=1)
    f['daily_change_mean'] = diff.abs().mean(axis=1)
    f['daily_change_std'] = diff.std(axis=1)
    f['weekend_ratio'] = x.loc[:, x.columns.dayofweek >= 5].mean(axis=1)/(x.loc[:, x.columns.dayofweek < 5].mean(axis=1)+1)
    for month in range(1, 11):
        m = x.loc[:, x.columns.month == month]
        f[f'month_{month}_mean'] = m.mean(axis=1)
        f[f'month_{month}_missing'] = m.isna().mean(axis=1)
        f[f'month_{month}_zero'] = m.eq(0).sum(axis=1)/m.count(axis=1).replace(0, np.nan)
    f['late_early_ratio'] = x.iloc[:, -90:].mean(axis=1)/(x.iloc[:, :90].mean(axis=1)+1)
    f = f.replace([np.inf, -np.inf], np.nan)
    quality = pd.DataFrame({'dolu_gun_sayisi': count, 'doluluk_yuzdesi': count/305*100}, index=x.index)
    return f, quality


def prepare(raw_path, output, split_path=None, include_test=False):
    raw_path, output = Path(raw_path), Path(output)
    if output.exists():
        raise FileExistsError(f'Yeni bir çıktı klasörü seçin: {output}')
    # Bölme için yalnız kimlik/etiket okunur; tüketim daha sonra seçilen gruptan yüklenir.
    labels = pd.read_csv(raw_path, usecols=['CONS_NO', 'FLAG'], dtype={'CONS_NO': str})
    if labels.CONS_NO.isna().any() or not labels.CONS_NO.is_unique or labels.FLAG.isna().any() or set(labels.FLAG.unique()) != {0, 1}:
        raise ValueError('Kimlikler benzersiz, etiketler 0/1 olmalıdır.')
    if split_path and Path(split_path).exists():
        split = pd.read_csv(split_path, dtype={'CONS_NO': str})
        if not split.CONS_NO.is_unique or set(split.CONS_NO) != set(labels.CONS_NO) or set(split.grup) != {'egitim', 'test'}:
            raise ValueError('Bölme listesi ham veriyle uyuşmuyor.')
    else:
        tr, te = train_test_split(labels.CONS_NO, test_size=.2, random_state=42, stratify=labels.FLAG)
        split = pd.concat([pd.DataFrame({'CONS_NO': tr, 'grup': 'egitim'}), pd.DataFrame({'CONS_NO': te, 'grup': 'test'})], ignore_index=True)
    groups = ['egitim', 'test'] if include_test else ['egitim']
    ids = set(split.loc[split.grup.isin(groups), 'CONS_NO'])
    chunks = [ch[ch.CONS_NO.isin(ids)] for ch in pd.read_csv(raw_path, dtype={'CONS_NO': str}, chunksize=4000)]
    data = pd.concat(chunks).set_index('CONS_NO').drop(columns='FLAG')
    data.columns = pd.to_datetime(data.columns, format='%Y/%m/%d')
    output.mkdir(parents=True, exist_ok=False)
    split.to_csv(output/'abone_bolunmesi.csv', index=False)
    summary = {}
    for group in groups:
        ordered = split.loc[split.grup == group, 'CONS_NO']
        f, quality = features49(data.loc[ordered])
        scored = quality.dolu_gun_sayisi > 0
        kept = f.loc[scored].copy()
        if group == 'egitim':
            kept['FLAG'] = labels.set_index('CONS_NO').FLAG.loc[kept.index]
        kept.to_csv(output/f'{group}_ozellikleri_2016_v1.csv', index_label='CONS_NO')
        quality.to_csv(output/f'{group}_veri_kalitesi_2016.csv', index_label='CONS_NO')
        pd.DataFrame({'CONS_NO': f.index[~scored], 'neden': '2016 döneminde tüketim kaydı bulunmuyor'}).to_csv(output/f'{group}_veri_yetersiz_2016.csv', index=False)
        if group == 'test':
            targets = labels.set_index('CONS_NO').loc[ordered].copy()
            targets['skorlanabilir'] = scored
            targets.to_csv(output/'test_hedefleri_2016_v1.csv', index_label='CONS_NO')
        summary[group] = {'toplam': len(f), 'skorlanabilir': int(scored.sum()), 'yetersiz': int((~scored).sum())}
    config = {'surum': '49_reconstructed', 'donem_baslangic': '2016-01-01', 'donem_bitis': '2016-10-31', 'kimlik': 'CONS_NO', 'hedef': 'FLAG', 'ozellikler': f.columns.tolist(), 'random_state': 42, 'test_orani': .2, 'donemsel_doluluk_esigi': .8, 'eleme_kurali': '2016 dolu gün sayısı == 0', 'ogrenilmis_donusum_uygulandi': False, 'summary': summary}
    (output/'ozellik_tanimi_2016_v1.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=ROOT/'data/raw/data.csv')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', type=Path, default=ROOT/'data/splits/abone_bolunmesi.csv')
    p.add_argument('--include-test', action='store_true')
    args = p.parse_args()
    print(prepare(args.raw, args.output, args.split, args.include_test))
