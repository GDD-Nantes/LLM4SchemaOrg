from pathlib import Path
import pandas as pd

home_dir = "schemaorg/examples"

# Positive example for both
pos_df = pd.read_parquet(f"{home_dir}/semantic-pos.parquet")
pos_df = pos_df[["ref", "prop", "example", "example_snippet"]]
pos_df["label"] = "positive"

# Semantic Checker
sem_neg_df = pd.read_parquet(f"{home_dir}/semantic-neg.parquet")
sem_neg_df["label"] = "negative"
sem_test_df = pd.concat([pos_df, sem_neg_df], ignore_index=True)
sem_test_df = sem_test_df.iloc[sem_test_df.astype(str).drop_duplicates().index, :]
sem_test_df.to_parquet(f"{home_dir}/semantic.parquet")

# Factual checker
fac_neg_s_df = pd.read_parquet(f"{home_dir}/factual-simple-neg.parquet")
fac_neg_s_df["label"] = "negative"
fac_neg_c_df = pd.read_parquet(f"{home_dir}/factual-complex-neg.parquet")
fac_neg_c_df["label"] = "negative"

fac_test_df = pd.concat([pos_df, fac_neg_s_df, fac_neg_c_df], ignore_index=True)
fac_test_df = fac_test_df.iloc[fac_test_df.astype(str).drop_duplicates().index, :]
fac_test_df.to_parquet(f"{home_dir}/factual.parquet")

fac_simple_test_df = pd.concat([pos_df, fac_neg_s_df], ignore_index=True)
fac_simple_test_df = fac_simple_test_df.iloc[fac_simple_test_df.astype(str).drop_duplicates().index, :]
fac_simple_test_df.to_parquet(f"{home_dir}/factual-simple.parquet")

fac_complex_test_df = pd.concat([pos_df, fac_neg_c_df], ignore_index=True)
fac_complex_test_df = fac_complex_test_df.iloc[fac_complex_test_df.astype(str).drop_duplicates().index, :]
fac_complex_test_df.to_parquet(f"{home_dir}/factual-complex.parquet")
