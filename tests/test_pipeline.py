"""Tests for timestamp handling, feature leakage and error metrics."""
import pandas as pd
import numpy as np
import pytest
from src.pipeline import hourly_states, make_features, chronological_split, evaluate


def test_hourly_states_do_not_fill_long_outages():
    t = pd.date_range('2024-01-01', periods=60, freq='h', tz='UTC')
    t = t.delete(slice(1,10))
    events = pd.DataFrame({'space':['A']*len(t),'timestamp':t,'headcount':[1]*len(t)})
    result=hourly_states(events,max_age_hours=2)
    assert not ((result['timestamp']==pd.Timestamp('2024-01-01T06:00Z')).any())


def test_features_require_consecutive_hours():
    times = pd.date_range('2024-01-01',periods=70,freq='h',tz='UTC').delete(35)
    states=pd.DataFrame({'space':'A','timestamp':times,'headcount':np.arange(len(times))})
    features=make_features(states)
    assert pd.Timestamp('2024-01-02T10:00Z') not in set(features['timestamp'])
    assert len(features)>0


def test_split_chronological():
    times=pd.date_range('2024-01-01',periods=300,freq='h',tz='UTC')
    f=pd.DataFrame({'timestamp':times,'target':1})
    a,b,c=chronological_split(f)
    assert a['timestamp'].max()<b['timestamp'].min()<c['timestamp'].min()


def test_metrics_nonnegative_predictions():
    result=evaluate([0,1,2],[-1,1,2])
    assert result['MAE']==pytest.approx(0)
    assert result['RMSE']==pytest.approx(0)
