import json
from phase_3_predictive_analytics.preprocess_data import fetch_and_preprocess_data

# Run preprocessing with defaults
try:
    df = fetch_and_preprocess_data()
except Exception as e:
    print(json.dumps({"error": str(e)}))
    raise

# Collect numeric feature stats
stats = {}
for col in df.columns:
    try:
        ser = df[col]
        if ser.dtype.kind in 'biufc':
            stats[col] = {
                'mean': None if ser.isna().all() else float(ser.mean()),
                'std': None if ser.isna().all() else float(ser.std()),
                'min': None if ser.isna().all() else float(ser.min()),
                'max': None if ser.isna().all() else float(ser.max()),
                'non_null_count': int(ser.count()),
                'dtype': str(ser.dtype)
            }
        else:
            stats[col] = {
                'non_null_count': int(ser.count()),
                'dtype': str(ser.dtype)
            }
    except Exception as e:
        stats[col] = {'error': str(e)}

output = {
    'feature_columns': [c for c in df.columns if c != 'Target'],
    'feature_stats': stats,
    'rows': len(df)
}
print(json.dumps(output))
