import argparse, os, json
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize timestamps (UTC)
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'],  utc=True, errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], utc=True, errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime','tpep_dropoff_datetime'])

    # ISO 'Z' event time (matches spec)
    event_iso = df['tpep_pickup_datetime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    out = pd.DataFrame()
    out['event_ts'] = event_iso

    # entity_key uses the ISO timestamp (matches spec)
    out['entity_key'] = (
        df['VendorID'].astype(str) + "_" +
        event_iso + "_" +
        df['PULocationID'].astype(str) + "_" +
        df['DOLocationID'].astype(str)
    )

    # required base columns & derived features
    for c in ['trip_distance','fare_amount','total_amount','payment_type']:
        if c not in df.columns: df[c] = 0

    dur = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60.0
    out['duration_min'] = dur.clip(lower=0, upper=12*60)

    dist  = pd.to_numeric(df['trip_distance'], errors='coerce').fillna(0.0).clip(lower=0)
    total = pd.to_numeric(df['total_amount'],  errors='coerce').fillna(0.0)
    out['trip_distance']   = dist
    out['fare_amount']     = pd.to_numeric(df['fare_amount'], errors='coerce').fillna(0.0)
    out['total_amount']    = total
    out['amount_per_mile'] = (total / dist.clip(lower=0.1)).round(4)

    out['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour.astype('int64')
    out['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek.astype('int64')
    out['is_weekend']  = out['day_of_week'].isin([5,6]).astype('int64')

    pt = pd.to_numeric(df['payment_type'], errors='coerce').fillna(-1).astype('int64')
    out['payment_type_credit'] = (pt==1).astype('int64')
    out['payment_type_cash']   = (pt==2).astype('int64')

    cols = ['entity_key','event_ts','trip_distance','fare_amount','total_amount','duration_min',
            'amount_per_mile','hour_of_day','day_of_week','is_weekend',
            'payment_type_credit','payment_type_cash']
    return out[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV path, e.g. data/nyc_taxi_2025-03.csv')
    ap.add_argument('--month', required=True, help='YYYY-MM (for output names)')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    feats = build_features(df)

    os.makedirs('features/offline', exist_ok=True)
    os.makedirs('features/online',  exist_ok=True)

    offline = f'features/offline/nyc_features_{args.month}.parquet'
    online  = f'features/online/online_{args.month}.json'

    feats.to_parquet(offline, index=False)   # needs pyarrow
    sample = feats.sample(n=min(5, len(feats)), random_state=42)
    records = { r['entity_key']: {k: r[k] for k in feats.columns if k not in ('entity_key','event_ts')}
               for _, r in sample.iterrows() }
    with open(online, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    print('WROTE', offline)
    print('WROTE', online)

if __name__ == '__main__':
    main()