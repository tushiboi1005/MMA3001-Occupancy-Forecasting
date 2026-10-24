"""Leakage-aware building occupancy forecasting from event-log measurements.

Usage: python -m src.pipeline --zip path/to/occupancy.zip
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_events(zip_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Stream the compressed CSV, keeping only required event fields.

    Events are *not* treated as equally spaced observations. A later step
    reconstructs hourly states with a maximum freshness threshold.
    """
    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(csv_names) != 1:
            raise ValueError('Expected exactly one CSV in the ZIP')
        with z.open(csv_names[0]) as raw:
            # Chunked CSV reading avoids constructing millions of Python dictionaries.
            chunks = []
            seen = 0
            for chunk in pd.read_csv(raw, usecols=['floorspaceid','collecteddate','headcount'],
                                     chunksize=250_000, low_memory=False):
                if limit is not None:
                    chunk = chunk.iloc[:max(0, limit-seen)]
                seen += len(chunk)
                chunk = chunk.rename(columns={'floorspaceid':'space','collecteddate':'timestamp'})
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'],utc=True,errors='coerce')
                chunk['headcount'] = pd.to_numeric(chunk['headcount'],errors='coerce')
                chunk = chunk.dropna(subset=['space','timestamp','headcount'])
                chunk = chunk[chunk['headcount']>=0]
                chunks.append(chunk)
                if limit is not None and seen >= limit:
                    break
    if not chunks:
        raise ValueError('No events found')
    return pd.concat(chunks,ignore_index=True)


def hourly_states(events: pd.DataFrame, max_age_hours: int = 2) -> pd.DataFrame:
    """Construct hourly occupancy snapshots without filling long sensor outages.

    For each hour, use the most recent *previous* event, if at most
    max_age_hours old. Missing hours remain missing, not zero.
    """
    outputs = []
    for space, group in events.groupby('space'):
        g = group.sort_values('timestamp').drop_duplicates('timestamp', keep='last')
        if len(g) < 50:
            continue
        start = g['timestamp'].iloc[0].ceil('h')
        end = g['timestamp'].iloc[-1].floor('h')
        if end <= start:
            continue
        hours = pd.DataFrame({'timestamp': pd.date_range(start, end, freq='h', tz='UTC')})
        snapshot = pd.merge_asof(hours, g[['timestamp', 'headcount']].rename(columns={'timestamp':'event_time'}),
                                 left_on='timestamp', right_on='event_time', direction='backward',
                                 tolerance=pd.Timedelta(hours=max_age_hours))
        snapshot['space'] = space
        snapshot = snapshot.dropna(subset=['headcount'])
        outputs.append(snapshot[['space','timestamp','headcount']])
    if not outputs:
        raise ValueError('Insufficient events to construct hourly snapshots')
    return pd.concat(outputs, ignore_index=True).sort_values(['space','timestamp'])


def make_features(states: pd.DataFrame) -> pd.DataFrame:
    """Predict headcount one hour ahead from known history and calendar time.

    Features at time t are used to predict t+1. Consecutive-hour checks
    prevent lags and targets from bridging missing periods.
    """
    frames = []
    for space, g in states.groupby('space'):
        g = g.sort_values('timestamp').copy()
        for lag in [0, 1, 2, 24]:
            g[f'lag_{lag}'] = g['headcount'].shift(lag)
            age = (g['timestamp'] - g['timestamp'].shift(lag)).dt.total_seconds() / 3600
            g.loc[age != lag, f'lag_{lag}'] = np.nan
        g['target'] = g['headcount'].shift(-1)
        next_gap = (g['timestamp'].shift(-1) - g['timestamp']).dt.total_seconds()/3600
        g.loc[next_gap != 1, 'target'] = np.nan
        g['hour'] = g['timestamp'].dt.hour
        g['weekday'] = g['timestamp'].dt.dayofweek
        g['is_weekend'] = (g['weekday'] >= 5).astype(int)
        frames.append(g)
    return pd.concat(frames).dropna(subset=['lag_0','lag_1','lag_2','lag_24','target']).reset_index(drop=True)


def chronological_split(frame: pd.DataFrame):
    """Use chronological 60/20/20 partitions, avoiding random temporal leakage."""
    unique_times = np.sort(frame['timestamp'].unique())
    if len(unique_times) < 100:
        raise ValueError('Insufficient unique hours for temporal validation')
    a, b = unique_times[int(len(unique_times)*0.6)], unique_times[int(len(unique_times)*0.8)]
    train = frame[frame['timestamp'] < a].copy()
    valid = frame[(frame['timestamp'] >= a)&(frame['timestamp'] < b)].copy()
    test = frame[frame['timestamp'] >= b].copy()
    if min(len(train),len(valid),len(test)) < 30:
        raise ValueError('One temporal partition has fewer than 30 observations')
    return train, valid, test


def evaluate(actual, predicted):
    """Return MAE and RMSE in persons, clipping impossible negative forecasts."""
    predicted = np.maximum(np.asarray(predicted), 0)
    return {'MAE':float(mean_absolute_error(actual, predicted)),
            'RMSE':float(np.sqrt(mean_squared_error(actual, predicted)))}


def run(zip_path, output='results', limit=None):
    """Execute the auditable experiment and write numerical/graphical outputs."""
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    events = load_events(zip_path, limit)
    states = hourly_states(events)
    frame = make_features(states)
    if len(frame) < 300:
        raise ValueError('Not enough consecutive hourly samples for modelling')
    train, valid, test = chronological_split(frame)
    cols = ['lag_0','lag_1','lag_2','lag_24','hour','weekday','is_weekend']
    # Space-specific models avoid interpreting arbitrary UUIDs as ordered numbers.
    all_predictions = []
    space_metrics = {}
    for space in sorted(frame['space'].unique()):
        tr, va, te = (part[part['space']==space] for part in (train,valid,test))
        if min(len(tr),len(va),len(te)) < 30:
            continue
        candidates = []
        for leaf in (20, 50, 100):
            model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                                  min_samples_leaf=leaf, random_state=42)
            model.fit(tr[cols], tr['target'])
            score = evaluate(va['target'], model.predict(va[cols]))['MAE']
            candidates.append((score, leaf))
        best_leaf = min(candidates)[1]
        combined = pd.concat([tr,va])
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                              min_samples_leaf=best_leaf, random_state=42)
        model.fit(combined[cols],combined['target'])
        predictions = {'persistence':te['lag_0'].to_numpy(),
                       'previous_day':te['lag_24'].to_numpy(),
                       'ML':model.predict(te[cols])}
        space_metrics[space] = {'train':len(tr),'validation':len(va),'test':len(te),
                                'selected_min_leaf':best_leaf,
                                'validation_MAE_by_leaf':{str(leaf):score for score,leaf in candidates},
                                'test':{name:evaluate(te['target'],p) for name,p in predictions.items()}}
        tmp = te[['space','timestamp','target']].copy()
        for name,p in predictions.items():tmp[name]=np.maximum(p,0)
        all_predictions.append(tmp)
    if not all_predictions:
        raise ValueError('No spaces have enough data in all three time partitions')
    pred = pd.concat(all_predictions).sort_values('timestamp')
    pred.to_csv(out/'test_predictions.csv',index=False)
    overall = {name:evaluate(pred['target'],pred[name]) for name in ('persistence','previous_day','ML')}
    report = {'source':str(zip_path),'events_clean':len(events),'hourly_states':len(states),
              'feature_rows':len(frame),'spaces_modelled':len(space_metrics),
              'date_min':str(events['timestamp'].min()),'date_max':str(events['timestamp'].max()),
              'test_overall':overall,'per_space':space_metrics,
              'limitations':['Event-driven records; hourly snapshots use most recent state up to 2h old',
                             'Missing hours excluded, not assumed vacant',
                             'Predictions evaluated on observed hours only',
                             'Space identifiers not mapped to rooms due to unverified location mapping']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8,4))
    labels = list(overall)
    ax.bar(labels,[overall[k]['MAE'] for k in labels]);ax.set_ylabel('Test MAE (persons)')
    ax.set_title('One-hour-ahead occupancy prediction');fig.tight_layout()
    fig.savefig(out/'model_comparison.png',dpi=170);plt.close(fig)
    # Plot one space to avoid joining unrelated room observations.
    selected_space = pred['space'].value_counts().idxmax()
    sample = (pred[pred['space'] == selected_space]
              .sort_values('timestamp')
              .tail(168))
    fig, ax = plt.subplots(figsize=(11,4.5))
    ax.plot(sample['timestamp'],sample['target'],label='Actual occupancy',linewidth=1.5)
    ax.plot(sample['timestamp'],sample['ML'],label='ML prediction',linewidth=1.5)
    ax.set_xlabel('Date and time')
    ax.set_ylabel('Headcount (people)')
    ax.set_title('One-hour-ahead occupancy forecasting — single space')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out/'test_predictions.png',dpi=170)
    plt.close(fig)
    print(json.dumps({'overall':overall,'spaces_modelled':len(space_metrics),'feature_rows':len(frame)},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip',required=True,help='Path to occupancy ZIP file')
    parser.add_argument('--output',default='results')
    parser.add_argument('--limit',type=int,default=None,help='Development only: first N rows')
    args = parser.parse_args()
    run(args.zip,args.output,args.limit)
