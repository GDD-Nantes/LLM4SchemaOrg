import re
from bs4 import BeautifulSoup
import requests

import shutil

import warnings
warnings.filterwarnings("error")

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import countDistinct, expr, udf, collect_list
from pyspark.sql.types import BooleanType

from utils import ping
    
udf_ping = udf(ping, BooleanType())

def get_ref_attrs(schema_type_url):
    schema_attrs = []
    soup = BeautifulSoup(requests.get(schema_type_url).text, "html.parser")
    table = soup.find(class_="definition-table")
    for tr in soup.find_all("th", class_="prop-nam"):
        schema_attrs.append(tr.get_text().strip())
    
    return len(schema_attrs)

expected_nb_props = get_ref_attrs("https://schema.org/Recipe")

# Initialize a SparkSession
spark = SparkSession.builder.master("local") \
    .appName("NQuadsGetSample") \
    .config('spark.ui.port', '4050') \
    .getOrCreate()

#input_nq_file = "data/part_*.gz"
input_nq_file = "data/recipe_lax.nq.gz"
output_nq_file = "data/recipe_samples.csv"

shutil.rmtree(output_nq_file, ignore_errors=True)

# Read the input NQ file into a DataFrame
lines = spark.sparkContext.textFile(input_nq_file)

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
        countDistinct("predicate").alias("nbUsedPredicate"),
        collect_list(expr("concat_ws(' -> ', predicate, object)")).alias("lstObject")
    )
    .filter(expr("array_contains(lstObject, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> -> <http://schema.org/Recipe>')"))
    .withColumn("coverage", expr(f"nbUsedPredicate / {expected_nb_props}"))
    .orderBy("coverage", ascending=False)
    .limit(100)
)

# Save the valid N-Quads to the output NQ file
#df = df.withColumn("lstObject", concat_ws(" | ", df["lstObject"]))
df.drop("lstObject").write.csv(output_nq_file, header=True, mode="overwrite")
#valid_nquads_rdd.saveAsTextFile(output_nq_file, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
#valid_nquads_rdd.saveAsTextFile(output_nq_file)

# Stop the SparkSession
spark.stop()


