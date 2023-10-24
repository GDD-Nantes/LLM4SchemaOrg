from rdflib import Graph
from rdflib.exceptions import ParserError as ParseError

from rdflib import Graph
from tqdm import tqdm

import os
import shutil

from pyspark.sql import SparkSession

# Initialize a SparkSession
spark = SparkSession.builder.master("local") \
    .appName("NQuadProcessing") \
    .config('spark.ui.port', '4050') \
    .getOrCreate()

input_nq_file = "data/part_*.gz"
output_nq_file = "data/recipe_lax.gz"

shutil.rmtree(output_nq_file, ignore_errors=True)

# Read the input NQ file into a DataFrame
lines = spark.sparkContext.textFile(input_nq_file)

# Define a function to filter and parse N-Quads
def parse_nquad(line):
    try:
        # Parse the line as an N-Quad
        g = Graph()
        g.parse(data=line, format="nquads")
        return line
    except ParseError:
        return None

# Use PySpark's map transformation to filter and parse N-Quads in parallel
valid_nquads_rdd = lines.map(parse_nquad).filter(lambda x: x is not None)

# Save the valid N-Quads to the output NQ file
valid_nquads_rdd.saveAsTextFile(output_nq_file, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

# Stop the SparkSession
spark.stop()