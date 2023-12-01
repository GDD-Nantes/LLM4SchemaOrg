import json
from pathlib import Path
import pprint
from rdflib import BNode, ConjunctiveGraph
from utils import get_type_definition, to_jsonld
from models.retrieval import *

import pyshacl
# from pyshex import ShExEvaluator
# from pyshex.shex_evaluator import evaluate_cli


class AbstractValidator:
    def __init__(self, **kwargs) -> None:
        pass
    
    def validate(self, json_ld, **kwargs):
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
            
    def validate(self, json_ld, **kwargs) -> ConjunctiveGraph:
        shapeGraph = self.__shape_graph
        dataGraph = ConjunctiveGraph().parse(json_ld, format="json-ld")
        _, self.__results_graph, self.__results_msgs = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph)
        report_path = f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.ttl"
        print(f"Writing to {report_path}")
        self.__results_graph.serialize(report_path, format="turtle")
        
    def get_messages(self):
        print(self.__results_msgs)
        
class FactualConsistencyValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever: AbstractRetrievalModel = globals()[kwargs.get("retriever")]()
        
    def validate(self, json_ld, **kwargs):
        def visit_json(json_stub: dict):
            prompts = []
            ent_type = None
            for k, v in json_stub.items():
                if k == "@context": continue
                if k == "@type":
                    ent_type = v    
                    continue
                # Recursively add prompt for dependant entities             
                if isinstance(v, dict):
                    prompts.extend([ f"{k} {prompt}" for prompt in visit_json(v)])
                else: 
                    prompts.append(f"{ent_type} {k} {repr(v)}") 
            
            return prompts   
        
        document = kwargs["document"]
        with open(json_ld, "r") as json_fs, open(document, "r") as doc_fs:
            data = json.load(json_fs)
            prompts = visit_json(data)
            document_content = doc_fs.read()
            
            for prompt in prompts:
                print(prompt)
                scores = self.__retriever.query(prompt, document_content)
                
class TypeConformanceLLMValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs.get("retriever")
        
    def validate(self, json_ld, **kwargs):
        prompt_base = f"""
        Given the schema.org markup element below:
        
        ```json
        %MARKUP%
        ```
        
        Given the definition below:
        
        ```txt
        %DEFINITION%
        ```
        
        Does the markup element match the definition? (Yes/No)
        
        """
        
        def visit_json(json_stub: dict):
            prompts = []
            ent_type = None
            for k, v in json_stub.items():
                if k == "@type":
                    ent_type = v  
                    continue
                print(k, v)
                pprint.pprint(json_stub)

                # Recursively add prompt for dependant entities             
                if isinstance(v, BNode):
                    child_stub = json_stub[v]
                    prompts.extend(visit_json(child_stub))
                elif isinstance(v, dict):
                    prompts.extend(visit_json(v))
                else: 
                    markup = {str(k): str(v)}
                    print(markup)
                    definition: dict = get_type_definition(ent_type, prop=str(k), simplify=True, verbose=True)
                    prompt = ( prompt_base
                        .replace("%MARKUP%", str(markup))
                        .replace("%DEFINITION%", str(definition))
                    )
                    prompts.append(prompt) 
            
            return prompts   
        
        data = to_jsonld(json_ld)        
        prompts = visit_json(data)
                            
        for prompt in prompts:
            response = self.__retriever.query(prompt, remember=False)