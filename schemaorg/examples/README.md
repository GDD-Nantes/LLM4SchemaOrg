# Schema.org examples

This directory contains ressources to generate validation set for Factuality and Compliance checker from Schema.org example.

- `(factual|compliance)/*`: JSON files showing all (property, value, type) triples that were really evaluated during validation

- `schemaorg-all-examples.txt`: This is the file downloaded from [Schema.org Github Repository](https://github.com/schemaorg/schemaorg/blob/main/data/releases/23.0/schemaorg-all-examples.txt)

- `schemaorg-all-examples.ttl`: This file is produced by executing `scrape.py`, which parse the semi-structured `.txt` file into RDF Turtle. You can load this file into any RDF store and make queries there. Note: Some text (pre-markup) has been re-generated using GPT-4 when they don't cover the entire markup, i.e., the Cosine Similarity is less than 22% (emprical threshold).

- `misc/*`: artifacts produced by running `markup/schemaorg_examples_dataset.py`, `merge.py` and `clean.py`