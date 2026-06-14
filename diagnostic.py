import pandas as pd

df = pd.read_parquet("data/trials_raw.parquet")

print("Status distribution:")
print(df["overall_status"].value_counts())

print("\nCondition sample:")
print(df["conditions"].value_counts().head(20))

print("\nZero enrollment rate:", (df["enrollment_count"]==0).mean())

print("\nTermination rate by phase:")
print(df.groupby("phase_encoded")["terminated"].mean())

print("\nTermination rate by condition (top 10):")
if 'condition' in df.columns:
    print(df.groupby("condition")["terminated"].mean().sort_values().tail(10))
elif 'conditions' in df.columns:
    print(df.groupby("conditions")["terminated"].mean().sort_values().tail(10))
