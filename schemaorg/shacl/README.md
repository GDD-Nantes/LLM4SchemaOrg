# Schema.org Shapes

This directory contains ressources to generate shape constraints from Schema.org ontology.

- `schemaorg/shacl/schemaorg_datashapes.shacl`: The SHACL shapes generated using `generate_shacl_shape` command from `markup/shacl.py`. Use this file when you want to validate under Open World Assumption.

- `schemaorg/shacl/schemaorg_datashapes_closed.shacl`: The SHACL shapes generated using `close_schemaorg_ontology` command from `markup/shacl.py`. Use this file when you want to validate under Close World Assumption.