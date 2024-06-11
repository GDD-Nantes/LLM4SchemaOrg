from pathlib import Path
import pandas as pd

home_dir = "schemaorg/examples/misc"

# Positive example for both
pos_df = pd.read_parquet(f"{home_dir}/compliance-pos.parquet")
pos_df = pos_df[["ref", "prop", "example", "example_snippet"]]
pos_df["label"] = "positive"

# Compliance Checker
sem_neg_df = pd.read_parquet(f"{home_dir}/compliance-neg.parquet")
sem_neg_df["label"] = "negative"
sem_test_df = pd.concat([pos_df, sem_neg_df], ignore_index=True)
sem_test_df = sem_test_df.iloc[sem_test_df.astype(str).drop_duplicates().index, :]
sem_test_df.to_parquet(f"{home_dir}/compliance.parquet")
print("Compliance")
print(sem_test_df["label"].value_counts())

# Factual checker
fac_neg_s_df = pd.read_parquet(f"{home_dir}/factual-extrinsic-neg.parquet")
fac_neg_s_df["label"] = "negative"
fac_neg_c_df = pd.read_parquet(f"{home_dir}/factual-intrinsic-neg.parquet")
fac_neg_c_df["label"] = "negative"

# fac_test_df = pd.concat([pos_df, fac_neg_s_df, fac_neg_c_df], ignore_index=True)
# fac_test_df = fac_test_df.iloc[fac_test_df.astype(str).drop_duplicates().index, :]
# fac_test_df.to_parquet(f"{home_dir}/factual.parquet")

fac_extrinsic_test_df = pd.concat([pos_df, fac_neg_s_df], ignore_index=True)
fac_extrinsic_test_df = fac_extrinsic_test_df.iloc[fac_extrinsic_test_df.astype(str).drop_duplicates().index, :]
fac_extrinsic_test_df.to_parquet(f"{home_dir}/factual-extrinsic.parquet")
print("Factual Simple")
print(fac_extrinsic_test_df["label"].value_counts())

fac_intrinsic_test_df = pd.concat([pos_df, fac_neg_c_df], ignore_index=True)
fac_intrinsic_test_df = fac_intrinsic_test_df.iloc[fac_intrinsic_test_df.astype(str).drop_duplicates().index, :]
fac_intrinsic_test_df.to_parquet(f"{home_dir}/factual-intrinsic.parquet")
print("Factual Complex")
print(fac_intrinsic_test_df["label"].value_counts())
