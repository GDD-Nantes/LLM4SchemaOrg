import pandas as pd
from markup.utils import logger

# HaluCheck
pos_df = pd.read_parquet("propcheck-pos.parquet")
pos_df = pos_df[["ref", "prop", "example_snippet"]]
pos_df["label"] = "positive"
logger.debug(len(pos_df))

for test in ["halucheck-simple", "halucheck-complex", "propcheck"]:
    neg_df = pd.read_parquet(f"{test}-neg.parquet")
    neg_df = neg_df[["ref", "prop", "example_snippet"]]
    neg_df["label"] = "negative"
    logger.debug(len(neg_df))

    test_df = pd.concat([pos_df, neg_df])
    test_df.to_parquet(f"{test}.parquet")
