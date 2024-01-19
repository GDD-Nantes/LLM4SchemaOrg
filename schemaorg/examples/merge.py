import pandas as pd

pos_df = pd.read_parquet("propcheck-pos.parquet")
pos_df = pos_df[["ref", "prop", "example_snippet"]]
pos_df["label"] = "positive"
print(len(pos_df))

neg_df = pd.read_parquet("propcheck-neg.parquet")
neg_df = neg_df[["ref", "prop", "example_snippet"]]
neg_df["label"] = "negative"
print(len(neg_df))

test_df = pd.concat([pos_df, neg_df])
test_df.to_parquet("propcheck.parquet")
