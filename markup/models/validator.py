import os
from pathlib import Path
import pyshacl
from rdflib import Graph
import requests


class AbstractValidator:
    def __init__(self, **kwargs) -> None:
        pass
    
    def validate(self, json_ld):
        pass
    
class ValidatorFactory:
    @staticmethod
    def create_validator(_class, **kwargs) -> AbstractValidator:
        return globals()[_class](**kwargs)

class SchemaOrgWebValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        self._validator_url = "https://validator.schema.org/"
    
    def validate(self, json_ld):
        return super().validate(json_ld)

class SchemaOrgShaclValidator(AbstractValidator):
            
    def validate(self, json_ld):
        shapeGraph = "https://datashapes.org/schema.ttl"
        dataGraph = Graph().parse(json_ld, format="json-ld")
        conforms, results_graph, results_text = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph)
        print(conforms)
        report_path = f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.json"
        results_graph.serialize(report_path, format="json-ld")
        print(results_text)