import json
from pathlib import Path
from rdflib import ConjunctiveGraph
import requests

import pyshacl
# from pyshex import ShExEvaluator
# from pyshex.shex_evaluator import evaluate_cli


class AbstractValidator:
    def __init__(self, **kwargs) -> None:
        pass
    
    def validate(self, json_ld):
        pass
    
class ValidatorFactory:
    @staticmethod
    def create_validator(_class, **kwargs) -> AbstractValidator:
        return globals()[_class](**kwargs)

# class ShexValidator(AbstractValidator):   
    
#     def __init__(self, shape_graph, **kwargs) -> None:
#         self.__shape_graph = shape_graph
#         self.__results = None
#         super().__init__(**kwargs)
        
#     def validate(self, json_ld):
#         evaluator = ShExEvaluator(schema=self.__shape_graph, start="http://schema.org/validation#ValidSchemaProduct")
#         # self.__results = evaluator.evaluate(json_ld, rdf_format="json-ld")
#         self.__results = evaluate_cli(json_ld, self.__shape_graph)
        
#     def get_messages(self):
#         print(self.__results)
#         for r in self.__results:
#             if not r.result:
#                 print(r.reason)

class ShaclValidator(AbstractValidator):
    
    def __init__(self, shape_graph, **kwargs) -> None:
        self.__shape_graph = shape_graph
        self.__results_graph = None
        self.__results_msgs = None
        super().__init__(**kwargs)
            
    def validate(self, json_ld) -> ConjunctiveGraph:
        shapeGraph = self.__shape_graph
        dataGraph = ConjunctiveGraph().parse(json_ld, format="json-ld")
        _, self.__results_graph, self.__results_msgs = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph)
        report_path = f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.ttl"
        print(f"Writing to {report_path}")
        self.__results_graph.serialize(report_path, format="turtle")
        
    def get_messages(self):
        print(self.__results_msgs)
        
