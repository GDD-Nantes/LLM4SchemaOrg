from pathlib import Path
import pandas as pd

home_dir = "schemaorg/examples/misc"

# Positive example for both
pos_df = pd.read_json(f"{home_dir}/positive.json", orient="records", lines=True)
pos_df = pos_df[["ref", "prop", "example", "example_snippet"]]
pos_df["label"] = "positive"

# Compliance Checker
compliance_neg_df = pd.read_json(f"{home_dir}/compliance_neg.json", orient="records", lines=True)
compliance_neg_df["label"] = "negative"
compliance_test_df = pd.concat([pos_df, compliance_neg_df], ignore_index=True)
compliance_test_df = compliance_test_df.iloc[compliance_test_df.astype(str).drop_duplicates().index, :].reset_index(drop=True)
compliance_test_df.to_json(f"{home_dir}/compliance.json", orient="records", lines=True, force_ascii=True)
print("Compliance")
print(compliance_test_df["label"].value_counts())

# Factual checker
factuality_neg_extrinsic_df = pd.read_json(f"{home_dir}/factual_extrinsic_neg.json", orient="records", lines=True)
factuality_neg_extrinsic_df["label"] = "negative"
factuality_neg_intrinsic_df = pd.read_json(f"{home_dir}/factual_intrinsic_neg.json", orient="records", lines=True)
factuality_neg_intrinsic_df["label"] = "negative"

# factuality_test_df = pd.concat([pos_df, factuality_neg_extrinsic_df, factuality_neg_intrinsic_df], ignore_index=True)
# factuality_test_df = factuality_test_df.iloc[factuality_test_df.astype(str).drop_duplicates().index, :]
# factuality_test_df.to_json(f"{home_dir}/factual.json")

factuality_extrinsic_test_df = pd.concat([pos_df, factuality_neg_extrinsic_df], ignore_index=True)
factuality_extrinsic_test_df = factuality_extrinsic_test_df.iloc[factuality_extrinsic_test_df.astype(str).drop_duplicates().index, :]
factuality_extrinsic_test_df.to_json(f"{home_dir}/factual_extrinsic.json", orient="records", lines=True, force_ascii=True)
print("Factual Extrinsic")
print(factuality_extrinsic_test_df["label"].value_counts())

factuality_intrinsic_test_df = pd.concat([pos_df, factuality_neg_intrinsic_df], ignore_index=True)
factuality_intrinsic_test_df = factuality_intrinsic_test_df.iloc[factuality_intrinsic_test_df.astype(str).drop_duplicates().index, :]
factuality_intrinsic_test_df.to_json(f"{home_dir}/factual_intrinsic.json", orient="records", lines=True, force_ascii=True)
print("Factual Intrinsic")
print(factuality_intrinsic_test_df["label"].value_counts())
