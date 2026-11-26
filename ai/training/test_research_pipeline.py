"""Çalıştırma: python -m pytest ai/training/test_research_pipeline.py -q"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ai.features.prepare_data import features49, prepare
from ai.training import research_pipeline as research


def test_missing_day_breaks_zero_run_and_is_not_zero():
    x = pd.DataFrame([[0., 0., 0.], [np.nan]*3], index=['a','b'], columns=pd.to_datetime(['2016-09-16','2016-09-17','2016-09-19']))
    f, q = features49(x)
    assert len(f.columns) == 49
    assert f.loc['a','en_uzun_sifir_serisi'] == 2
    assert f.loc['a','sifir_gun_orani'] == 1
    assert q.loc['a','dolu_gun_sayisi'] == 3
    assert np.isnan(f.loc['b','en_uzun_sifir_serisi'])
    assert np.isnan(f.loc['b','sifir_gun_orani'])


def test_prepare_preserves_split_and_refuses_overwrite(tmp_path):
    ids = [f'abone_{i:03d}' for i in range(20)]
    raw = pd.DataFrame({'CONS_NO': ids, 'FLAG': [0,1]*10, '2016/1/1': np.arange(20,dtype=float), '2016/10/31': np.arange(20,dtype=float)})
    raw.loc[0,['2016/1/1','2016/10/31']] = np.nan
    raw_path = tmp_path/'raw.csv'; raw.to_csv(raw_path,index=False)
    split = pd.DataFrame({'CONS_NO':ids,'grup':['egitim']*16+['test']*4})
    split_path=tmp_path/'split.csv';split.to_csv(split_path,index=False)
    out=tmp_path/'prepared'
    summary=prepare(raw_path,out,split_path,include_test=True)
    assert summary['egitim']['yetersiz']==1
    assert summary['test']['skorlanabilir']==4
    tr=pd.read_csv(out/'egitim_ozellikleri_2016_v1.csv')
    te=pd.read_csv(out/'test_ozellikleri_2016_v1.csv')
    assert not set(tr.CONS_NO)&set(te.CONS_NO)
    assert 'FLAG' not in te
    pd.testing.assert_frame_equal(pd.read_csv(out/'abone_bolunmesi.csv'),split)
    with pytest.raises(FileExistsError):prepare(raw_path,out,split_path)


def test_context_fit_never_sees_validation_or_test(tmp_path,monkeypatch):
    ids=[f'id_{i}' for i in range(60)]
    raw_path=tmp_path/'raw.csv'
    pd.DataFrame({'CONS_NO':ids,'FLAG':[0,1]*30}).to_csv(raw_path,index=False)
    split_path=tmp_path/'split.csv'
    pd.DataFrame({'CONS_NO':ids,'grup':['egitim']*50+['test']*10}).to_csv(split_path,index=False)
    dates=pd.date_range('2014-01-01','2016-10-31')
    def load(requested,raw_path):
        assert not set(requested)&set(ids[50:])
        arr=np.vstack([np.full(len(dates),int(x.split('_')[1])+1.) for x in requested])
        return arr,dates
    fitted=[]
    class Context:
        def __init__(self,**kwargs):pass
        def fit(self,arr,dates,frame):
            self.ids=set(frame.index);fitted.append(self.ids);return self
        def transform(self,arr,frame):return pd.DataFrame({'ctx':np.ones(len(frame))},index=frame.index)
    monkeypatch.setattr(research,'load_raw_for_ids',load)
    monkeypatch.setattr(research,'extract_subscriber_features',lambda arr,dates,base,**kw:base)
    monkeypatch.setattr(research,'FoldContextTransformer',Context)
    out=tmp_path/'research'
    research.prepare_folds(raw_path,split_path,out,folds=2)
    audit=json.loads((out/'fold_audit.json').read_text())
    seen=[]
    for fit,fold in zip(fitted,audit):
        assert fit==set(fold['train_ids'])==set(fold['context_fit_ids'])
        assert not fit&set(fold['validation_ids'])
        seen.extend(fold['validation_ids'])
    assert len(seen)==50 and set(seen)==set(ids[:50])


def test_smote_pair_uses_same_preprocessing():
    for family in ['lr','lgbm','xgboost']:
        natural=research.build_model(family,smoke=True)
        balanced=research.build_model(family,smoke=True,smote=True)
        for step in ['impute','scale','model']:
            assert repr(natural.named_steps[step])==repr(balanced.named_steps[step])
        assert 'smote' not in natural.named_steps
        assert balanced.named_steps['smote'].sampling_strategy==.5
