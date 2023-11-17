import glob
import re
import shutil

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import countDistinct, expr, udf, collect_list, array_intersect, col, array, lit
from pyspark.sql.types import BooleanType
import requests
from tqdm import tqdm

from markup.utils import get_page_content, get_ref_attrs, md5hex, ping
import pandas as pd

from ftlangdetect import detect as lang_detect


import click

@click.group
def cli():
    pass

def lang_detect(url):
    try:
        content = requests.get(url, allow_redirects=False, timeout=5).text
        lang = lang_detect(content.replace("\n", ""))["lang"]
        return lang
    except:
        return None
        
@cli.command()
@click.argument("indir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
def merge_csv(indir, outfile):
    csvs = glob.glob(f"{indir}/part*.csv")
    df = pd.DataFrame()
    for csv in tqdm(csvs):
        df = pd.concat([df, pd.read_csv(csv)], ignore_index=True)
    
    df.to_csv(outfile, index=False)

@cli.command()
@click.argument("infile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument("schema_type", type=click.STRING)
def extract_top_coverage(infile, outdir, schema_type):
    
    udf_hash = udf(md5hex)
    udf_ping = udf(ping, BooleanType())
    udf_langdetect = udf(lang_detect)
    
    ref_props = get_ref_attrs(f"https://schema.org/{schema_type}")
    expected_nb_props = len(ref_props)

    # Initialize a SparkSession
    spark = ( SparkSession.builder.master("local") 
        .appName("NQuadsGetSample") 
        .config('spark.ui.port', '4050') 
        .config("spark.local.dir", "./tmp/spark-temp")
        .getOrCreate()
    )

    shutil.rmtree(outdir, ignore_errors=True)

    # Read the input NQ file into a DataFrame
    lines = spark.sparkContext.textFile(infile)

    # Define a function to filter and parse N-Quads
    def parse_nquad(line):
        # Parse the line as an N-Quad
        quad_motif = re.compile(r'([^\s]+)\s([^\s]+)\s(.+)\s([^\s]+)\s+\.')
        result = quad_motif.match(line)
        subj = result.group(1).strip()
        pred = result.group(2).strip()
        obj = result.group(3).strip()
        source = result.group(4).strip().strip("<>")
            
        return Row(predicate=pred, object=obj, source=source)

    # Use PySpark's map transformation to filter and parse N-Quads in parallel
    valid_nquads_rdd = lines.map(parse_nquad).toDF()
    df = (valid_nquads_rdd.groupBy("source")
        .agg(
            #countDistinct("predicate").alias("nbUsedPredicate"),
            collect_list("predicate").alias("lstPred"),
            collect_list(expr("concat_ws(' -> ', predicate, object)")).alias("lstObject")
        )
        .filter(expr(f"array_contains(lstObject, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> -> <http://schema.org/{schema_type}>')"))
        .withColumn("id", udf_hash("source"))
        .withColumn("lang", udf_langdetect("source"))
        .withColumn("lstPred", array_intersect(col("lstPred"), lit(ref_props)))
        .withColumn("nbUsedPredicate", expr("size(lstPred)"))
        .withColumn("coverage", expr(f"nbUsedPredicate / {expected_nb_props}"))
    )

    # Save the valid N-Quads to the output NQ file
    #df = df.withColumn("lstObject", concat_ws(" | ", df["lstObject"]))
    df.drop("lstObject").drop("lstPred").write.csv(outdir, header=True, mode="overwrite")

    # Stop the SparkSession
    spark.stop()
    
if __name__ == "__main__":
    cli()
