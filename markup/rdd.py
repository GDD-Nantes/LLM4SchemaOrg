import glob
import json
import logging
from pathlib import Path
import re
import shutil

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import countDistinct, expr, udf, collect_list, array_intersect, col, array, lit, concat_ws
from pyspark.sql.types import BooleanType, ArrayType, StringType
from rdflib import ConjunctiveGraph, URIRef
from utils import filter_graph, get_type_definition, lang_detect, md5hex, ping, to_jsonld
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import click

import seaborn as sns

@click.group
def cli():
    pass
        
@cli.command()
@click.argument("indir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
def merge_csv(indir, outfile):
    csvs = glob.glob(f"{indir}/part*.csv")
    df = pd.DataFrame()
    for csv in tqdm(csvs):
        df = pd.concat([df, pd.read_csv(csv)], ignore_index=True)
    
    df.to_csv(outfile, index=False)
    
# Define a function to filter and parse N-Quads
def __parse_nquad(line, line_number):
    # Parse the line as an N-Quad
    quad_motif = re.compile(r'([^\s]+)\s([^\s]+)\s(.+)\s([^\s]+)\s+\.')
    result = quad_motif.match(line)
    subj = result.group(1)
    pred = result.group(2)
    obj = result.group(3)
    source = result.group(4).strip("<>")
            
    return Row(subject=subj, predicate=pred, object=obj, source=source, line_number=line_number)

@cli.command()    
@click.argument("infile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("source", type=click.STRING)
def extract_markup(infile, outfile, source):
    # Use PySpark's map transformation to filter and parse N-Quads in parallel
    # Initialize a SparkSession
    spark = ( SparkSession.builder.master("local") 
        .appName("NQuadsGetSample") 
        .config('spark.ui.port', '4050') 
        .config("spark.local.dir", "./tmp/spark-temp")
        .getOrCreate()
    )
    
    outdir = f"{Path(outfile).parent}/{Path(outfile)}.tmp"
    shutil.rmtree(outdir, ignore_errors=True)
    
    lines = spark.sparkContext.textFile(infile)
    valid_nquads_rdd = lines.zipWithIndex().map(lambda x: __parse_nquad(x[0], x[1])) \
        .filter(lambda x: x.source == source) \
        .map(lambda x: f"{x.subject} {x.predicate} {x.object} {x.source} .")
    
    valid_nquads_rdd.saveAsTextFile(outdir)
    spark.close()
    
@cli.command()
@click.argument("infile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
def extract_stats_bioschema(infile):
    g = ConjunctiveGraph()
    g.parse(infile)
    
    records = []
    
    for src in tqdm(g.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), unique=True)):
        markup = filter_graph(g, root=src)
        id = md5hex(src.toPython())
        outfile = f"{Path(infile).parent}/corpus/baseline/{id}.json"
        dumpfile = f"{Path(infile).parent}/corpus/baseline/{id}.ttl"
        
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        markup.serialize(dumpfile)
        jsonld = to_jsonld(markup)
        with open(outfile, "w") as f:
            json.dump(jsonld, f)
        
        #TODO: Find a way to measure on-scale the coverage
        records.append({"id": id, "url": src.toPython(), "type": jsonld["http://purl.org/dc/terms/conformsTo"]})
       
    df = pd.DataFrame.from_records(records)
    df.set_index("id")
    df.to_csv(f"{Path(infile).parent}/bioschema.csv", index=False)

@cli.command()
@click.argument("incsv", type=click.Path(exists=True, file_okay=True, dir_okay=False))    
def extras(incsv):
    df = pd.read_csv(incsv)
    df["lang"] = df["source"].progress_apply(lang_detect)
    df.to_csv(f"{Path(incsv).parent}/{Path(incsv).stem}_lang.csv", index=False)

@cli.command()
@click.argument("incsv", type=click.Path(exists=True, file_okay=True, dir_okay=False))  
def plot(incsv):
    df = pd.read_csv(incsv)
    g = sns.kdeplot(data=df, x="coverage")
    g.figure.savefig(f"{Path(incsv).parent}/{Path(incsv).stem}_coverage_dist.png")

@cli.command()
@click.argument("infile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument("schema_type", type=click.STRING)
def extract_stats_wdc(infile, outdir, schema_type):

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    udf_hash = udf(md5hex)
    udf_ping = udf(ping, BooleanType())

    @udf(StringType())
    def first(array):
        return array[0]
    
    ref_props = get_type_definition(f"http://schema.org/{schema_type}")
    expected_nb_props = len(ref_props)

    # Initialize a SparkSession
    spark = ( SparkSession.builder.master("local") 
        .appName("NQuadsGetSample") 
        .config('spark.ui.port', '4050') 
        .config("spark.local.dir", "./tmp/spark-temp")
        .getOrCreate()
    )

    spark.sparkContext.addPyFile(f"{Path(__file__).parent}/utils.py")

    shutil.rmtree(outdir, ignore_errors=True)

    # Read the input NQ file into a DataFrame
    lines = spark.sparkContext.textFile(infile)

    # Use PySpark's map transformation to filter and parse N-Quads in parallel
    valid_nquads_rdd = lines.zipWithIndex().map(lambda x: __parse_nquad(x[0], x[1])).toDF()
    df = (
        valid_nquads_rdd.groupBy("source")
        .agg(
            collect_list("line_number").alias("lstOffset"),
            collect_list("predicate").alias("lstPred"),
        )
        .withColumn("id", udf_hash("source"))
        .withColumn("offset", first("lstOffset"))
        .withColumn("length", expr("size(lstOffset)"))
        .withColumn("lstClassPred", expr(f"array_intersect(array({', '.join([repr(p) for p in ref_props])}), lstPred)"))
        .withColumn("nbUsedPredicate", expr("size(lstClassPred)"))
        .withColumn("coverage", expr(f"nbUsedPredicate / {expected_nb_props}"))
    )

    # Save the valid N-Quads to the output NQ file
    # df = df.withColumn("lstPred", concat_ws(" | ", df["lstPred"]))
    # df = df.withColumn("lstOffset", concat_ws(" | ", df["lstOffset"]))
    (
        df
        .drop("lstPred")
        .drop("lstClassPred")
        .drop("lstOffset")
        .write.csv(outdir, header=True, mode="overwrite")
    )

    # Stop the SparkSession
    spark.stop()
    
if __name__ == "__main__":
    cli()
