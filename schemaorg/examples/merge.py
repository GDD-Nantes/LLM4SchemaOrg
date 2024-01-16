import pandas as pd

pos_df = pd.read_feather("propcheck-pos.feather")
pos_df = pos_df[["ref", "prop", "example_snippet"]]
pos_df["label"] = "positive"
print(len(pos_df))

neg_df = pd.read_feather("propcheck-neg.feather")
neg_df = neg_df[["ref", "prop", "example_snippet"]]
neg_df["label"] = "negative"
print(len(neg_df))

test_df = pd.concat([pos_df, neg_df])
test_df.to_feather("propcheck.feather")
