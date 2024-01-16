from copy import deepcopy
import glob
from itertools import chain
import json
import os
from pathlib import Path
from pprint import pprint
import re
import shutil
from bs4 import BeautifulSoup
import pandas as pd
from rdflib import URIRef
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, size, udf, col, concat_ws, expr, flatten, collect_list, sum, first
from pyspark.sql.types import ArrayType, StringType

from rdflib import ConjunctiveGraph
from utils import _trafilatura, clean_json, get_page_content, html_to_rdf_extruct, md5hex, schema_simplify, to_jsonld
import click
import trafilatura
import tldextract

from tqdm import tqdm

import numpy as np
RANDOM_SEED = 42

@click.group
def cli():
    pass

@cli.command()
def clean_data():
    g = ConjunctiveGraph()
    g.parse("schemaorg/schemaorg-all-http.nt")

    schemaorg_all_urls = []
    for s, p, o in g:
        url = s.toPython()
        if url not in schemaorg_all_urls:
            schemaorg_all_urls.append(url)

    spark = ( SparkSession.builder.master("local") 
            .appName("SchemaOrgPset") 
            .config('spark.ui.port', '4050') 
            .config("spark.local.dir", "./.tmp/spark-temp")
            .getOrCreate()
        )

    spark.sparkContext.addPyFile(f"{Path(__file__).parent}/utils.py")

    # Read the CSV file with "\t" as separator and first row as header
    pset_df = (
        spark.read 
        .option("sep", "\t")  # Specify the separator
        .option("header", "true")  # Use the first row as header
        .csv("data/WDC/Pset/pset.csv")
    )

    @udf(returnType=ArrayType(StringType()))
    def filter_schemaorg(pset):
        results = []
        for prop in pset:
            
            # If type assertion, it has to be schema.org type
            if prop.startswith("isa:"):
                m = re.search(r"isa:<(.*)>", prop)
                if m is None: continue
                ent_type = m.group(1)
                if not ent_type.startswith("schema.org"): continue
                ent_type = "http://" + ent_type
                
                if ent_type not in schemaorg_all_urls: continue
                results.append(f"isa:<{ent_type}>")
                continue
            
            # If property, it has to be schema.org prop
            if not prop.startswith("<schema.org"): continue
            else:
                prop = "http://" + prop.strip("<>")
                if prop not in schemaorg_all_urls: continue
                results.append(prop)
        return results
    
    @udf(returnType=ArrayType(StringType()))
    def filter_isa(pset):
        return [ prop for prop in pset if prop.startswith("isa") ]
                
    # Clean data
    df = (
        pset_df
            .withColumn("count", col("count").cast("int"))
            .withColumn("pset", filter_schemaorg(split("pset", " ")))
            .withColumn("sample", split("sample", " "))
            .withColumn("pset_length", size("pset"))
            .filter(size("pset") > 0)
            .withColumn("pset", concat_ws(" ", "pset"))
            .groupBy("pset").agg(
                sum("count").alias("count_sum"),
                flatten(collect_list(col("sample"))).alias("samples"),
                first("pset_length").alias("pset_length")
            )
            .withColumn("samples", concat_ws(" ", "samples"))
            .withColumn("class", concat_ws(" ", filter_isa("pset")))
    )

    df.write.mode("overwrite").parquet("data/WDC/Pset/pset.parquet", compression="snappy")

@cli.command()
def plot():
    df = pd.read_parquet("data/WDC/Pset/pset.parquet")
    df["count_sum"] = df["count_sum"].astype(int)
    df["pset_length"] = df["pset"].str.split().apply(set).apply(len)
        
    plt.clf()
    pset_count_plot = sns.displot(df, x="count_sum", kde=True)
    # pset_count_plot = sns.lineplot(df.sort_values(by="count_sum", ascending=True).reset_index(drop=True).reset_index(), x="index", y="count_sum")
    pset_count_plot.set_yscale("log")
    # pset_count_plot.invert_yaxis()
    pset_count_plot.get_figure().savefig("data/WDC/Pset/pset_count_plot.png")

    plt.clf()
    pset_length_plot = sns.displot(df, x="pset_length", kde=True)
    # pset_length_plot = sns.lineplot(df.sort_values(by="pset_length", ascending=True).reset_index(drop=True).reset_index(), x="index", y="pset_length")
    # pset_length_plot.set_yscale("log")
    pset_length_plot.get_figure().savefig("data/WDC/Pset/pset_length_plot.png")

@cli.command()
@click.argument("h", type=click.INT)
@click.argument("d", type=click.FLOAT)
@click.argument("feature", type=click.STRING)
@click.option("--stratum-sample-size", type=click.INT)
@click.option("--fpc", is_flag=True, default=False)
@click.option("--explain", is_flag=True, default=False)
@click.option("--quantile", is_flag=True, default=False)
@click.option("--clean", is_flag=True, default=False)
def extract(h, d, feature, stratum_sample_size, fpc, explain, quantile, clean):

    home_base = "data/WDC/Pset"
    pset_df_fn = f"{home_base}/pset.parquet"
    sample_df_fn = f"{home_base}/sample.parquet"
    stratum_stats = None

    if clean and os.path.exists(sample_df_fn):
        os.remove(sample_df_fn)

    if not os.path.exists(sample_df_fn) or explain:
        pset_df = pd.read_parquet(pset_df_fn).reset_index()
        pset_df["count_sum"] = pset_df["count_sum"].astype(int)
        pset_df["stratum"] = pd.qcut(pset_df[feature], h) if quantile else pd.cut(pset_df[feature], h)
        pset_df["samples"] = pset_df["samples"].str.split(" ")
        
        def sample_url(row):

            index_list = deepcopy(row["unit_index"])
            urls = []
            indexes = []
            classes = []
            url_blocklist = {}

            n_total = int(row["stratum_sample_size"])
            progress_bar = tqdm(total=n_total, desc="Sampling...")

            print("BEFORE", len(index_list), len(urls))

            while len(index_list) > 0 and len(urls) < n_total:
                np.random.seed(RANDOM_SEED)
                index = np.random.choice(index_list)
                index_list.remove(index)

                sample: list = deepcopy(pset_df["samples"].iat[index])
                
                content = None
                while len(sample) > 0 and content is None:
                    np.random.seed(RANDOM_SEED)
                    url = np.random.choice(sample)
                    url_id = md5hex(url)
                    sample.remove(url)

                    domain = tldextract.extract(url).registered_domain

                    if domain not in url_blocklist.keys():
                        url_blocklist[domain] = 0

                    if url_blocklist[domain] == 5: continue
                    
                    print(f"Examining {url}...")
                    try: content = get_page_content(url)
                    except Exception as e: 
                        print(e)
                        if str(e).startswith("Could not extract content"): raise e
                        else: pass

                    if content is not None:
                        unit_classes = [ 
                            re.search(r"isa:<(.*)>", item).group(1)
                            for item in  pset_df["pset"].iat[index].split() 
                            if item.startswith("isa:") 
                        ]

                        kg_extruct = html_to_rdf_extruct(f".cache/{url_id}_raw.html")
                        ref_markups = to_jsonld(kg_extruct, simplify=True, keep_root=True)

                        ref_markup_types = []
                        for ref_markup in ref_markups.values():
                            ref_classes = ref_markup["@type"]
                            for ref_class in ref_classes:
                                ref_markup_types.append(f"http://schema.org/{ref_class}")
                        
                        has_expected_markup = any(rmt in unit_classes for rmt in ref_markup_types )
                        if has_expected_markup:
                            print(f"Adding {url}...")
                            indexes.append(index)
                            urls.append(url)
                            classes.append(unit_classes)
                            progress_bar.update(1)
                    else:
                        url_blocklist[domain] += 1
            
            print("AFTER", len(index_list), len(urls))

            row["unit_classes"] = " ".join([ "|".join(cs) for cs in classes ])
            row["unit_index"] = " ".join([str(x) for x in indexes])
            row["unit_url"] = " ".join(urls)
            return row
                
        N = len(pset_df)
        # tscore = t.ppf(d, 1, loc=0, scale=1)
        V = (d*N)**2
        
        stratum_stats = (
            pset_df[["stratum", feature, "index"]]
            .groupby(by="stratum")
            .agg(
                min=(feature, "min"),
                max=(feature, "max"),
                mean=(feature, "mean"),
                std=(feature, "std"),
                stratum_size=(feature, "count"),
                unit_index=("index", list)
            )
        ).reset_index()

        stratum_stats["stratum"] = stratum_stats["stratum"].astype(str)
        
        # if fpc:
        #     stratum_stats["std"] = stratum_stats.apply(lambda row: np.sqrt(N-row["stratum_size"] / N-1) * row["std"], axis=1)
        
        if stratum_sample_size is None:
            Nh_sh_sum_squared = (stratum_stats["std"] * stratum_stats["stratum_size"]).sum()**2
            Nh_vh_sum = (stratum_stats["std"]**2 * stratum_stats["stratum_size"]).sum()
            n_0 = Nh_sh_sum_squared / V
            n = n_0 / (1 + Nh_vh_sum/V)
            stratum_stats["stratum_sample_size"] = stratum_stats.apply(lambda row: (n * row["stratum_size"]) / N, axis=1)
        else:
            stratum_stats["stratum_sample_size"] = stratum_sample_size
        
        stats_cols = ['stratum', 'min', 'max', 'mean', 'std', 'stratum_size', 'stratum_sample_size']
        stratum_stats[stats_cols].to_csv(f"{home_base}/sample_stats.csv", index=False)
        if explain:
            print(stratum_stats[stats_cols])
            return
        stratum_stats = stratum_stats.apply(sample_url, axis=1)
        stratum_stats.to_parquet(sample_df_fn)
    else:
        stratum_stats = pd.read_parquet(sample_df_fn)
    
    for stratum_idx, row in stratum_stats.iterrows():
        stratum_home_corpus = f"{home_base}/stratum_{stratum_idx}/corpus"
        stratum_home_baseline = f"{home_base}/stratum_{stratum_idx}/baseline"
        if clean: 
            shutil.rmtree(stratum_home_corpus, ignore_errors=True)
            shutil.rmtree(stratum_home_baseline, ignore_errors=True)
        
        Path(stratum_home_corpus).mkdir(parents=True, exist_ok=True)
        Path(stratum_home_baseline).mkdir(parents=True, exist_ok=True)
        
        for unit_classes, unit_url in zip(row["unit_classes"].split(" "), row["unit_url"].split(" ")):
            url_id = md5hex(unit_url)            
            corpus_fn = f"{stratum_home_corpus}/{url_id}.txt"
            unit_class_fn = f"{stratum_home_corpus}/{url_id}_class.json"
            unit_classes = unit_classes.split("|") 

            kg_extruct = html_to_rdf_extruct(f".cache/{url_id}_raw.html")
            ref_markups = to_jsonld(kg_extruct, simplify=True, keep_root=True)

            for ref_markup in ref_markups.values():
                # pprint(ref_markup)
                ref_classes = ref_markup["@type"]
                for ref_class in ref_classes:
                    ref_markup_fn = f"{stratum_home_baseline}/{url_id}_{ref_class}.jsonld"
                    if os.path.exists(ref_markup_fn):
                        print(f"{ref_markup_fn} already exists, skipping...")
                        continue
                    if f"http://schema.org/{ref_class}" not in unit_classes: 
                        print(f"There is no markup for type {ref_class} document {url_id}, expecting {unit_classes}")
                        continue
                    with open(ref_markup_fn, "w") as f:
                        ref_markup = clean_json(ref_markup)
                        json.dump(ref_markup, f)

            with open(corpus_fn, "w") as cfs, open(unit_class_fn, "w") as jfs:
                json.dump(unit_classes, jfs)                
                content = get_page_content(unit_url)
                cfs.write(content)    

@cli.command()
def test_content_extractor():
    acceptable_clean_ratio = 0.3 # @cite CCNet 
    records = []
    for html_raw_fn in tqdm(glob.glob(".cache/*_raw.html")):
        url_id = re.search(r"(\w+)_raw\.html", html_raw_fn).group(1)
        with open(html_raw_fn, "r") as f:
            html_raw = f.read()
            _, tok_ref, tok_clean, clean_ratio = _trafilatura(html_raw)
            
            records.append({
                "url_id": url_id,
                "url_file": html_raw_fn,
                "ref_tok": tok_ref,
                "clean_tok": tok_clean,
                "clean_ratio": clean_ratio
            })
    
    df = pd.DataFrame.from_records(records)
    df.to_csv("trafilatura_test.csv")
    print(df)

if __name__ == "__main__":
    cli()