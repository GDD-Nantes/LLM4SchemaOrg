import json
import os
from pathlib import Path
from pprint import pprint
import re
import textwrap
from rdflib import BNode, ConjunctiveGraph
from utils import get_schema_example, get_type_definition, schema_simplify, to_jsonld
from models.retrieval import *

import pyshacl
from pyshacl.rdfutil import stringify_node
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
        super().__init__(**kwargs)
            
    def validate(self, json_ld, **kwargs) -> ConjunctiveGraph:
        """Validate syntaxically JSON-LD.

        Args:
            json_ld (_type_): _description_

        Returns:
            ConjunctiveGraph: _description_
        """
        shapeGraph = self.__shape_graph
        dataGraph = ConjunctiveGraph().parse(json_ld, format="json-ld")
        valid, report_graph, report_msgs = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph, inference="both")
        report_path = f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.json"
        print(f"Writing to {report_path}")
        # report_graph.serialize(report_path, format="turtle")
        
        # Write the clean message
        report = {
            "valid": valid,
            "msgs": []
        }
        
        query = """
        SELECT ?focusNode ?resultMessage ?resultPath ?sourceShape ?value WHERE {
            ?report <http://www.w3.org/ns/shacl#result> ?result .
            ?result <http://www.w3.org/ns/shacl#focusNode> ?focusNode ;
                    <http://www.w3.org/ns/shacl#resultMessage> ?resultMessage ;
                    <http://www.w3.org/ns/shacl#resultPath> ?resultPath ;
                    <http://www.w3.org/ns/shacl#sourceShape> ?sourceShape ;
                    <http://www.w3.org/ns/shacl#value> ?value
        }
        """
        
        qresults = report_graph.query(query)
        for qres in qresults:
            # focusNode = qres.get("focusNode")
            resultMessage = qres.get("resultMessage")
            resultPath = stringify_node(report_graph, qres.get("resultPath"))
            sourceShape = stringify_node(report_graph, qres.get("sourceShape"))
            value = qres.get("value").toPython()
                                    
            node_info = f"( shape {sourceShape}, path {resultPath} )"
            message = str(resultMessage).strip()
            if message.startswith("Node"):
                if "is closed. It cannot have value" in message:
                    message = re.sub(r"\[.*\]", node_info, message)
                    message = f"({resultPath}) is not an property of ({sourceShape}). Remove the property."
            elif message.startswith("Value"):
                message = re.sub(r"Value", f"Node {node_info}: {value}", message)
            
            if message not in report["msgs"]:
                report["msgs"].append(message)
        
        with open(report_path, "w") as f:
            json.dump(report, f)
        
        return report
        
class FactualConsistencyValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        retriever = kwargs.get("retriever")
        if isinstance(retriever, str):
            self.__retriever: AbstractRetrievalModel = globals()[retriever]()
        else:
            self.__retriever = retriever
        
    def validate(self, json_ld, **kwargs):
        def visit_json(json_stub: dict):
            prompts = []
            ent_type = None
            for k, values in json_stub.items():
                if k == "@type":
                    ent_type = schema_simplify(values)  
                    continue
                                
                # Recursively add prompt for dependant entities  
                if isinstance(values, dict):
                    child_prompts = visit_json(values)
                    prompts.extend(child_prompts)
                else: 
                    for v in values:
                        # prompts.append(f"{ent_type} {schema_simplify(k)} {schema_simplify(v)}") 
                        prompts.append(schema_simplify(v))
            
            return prompts   
        
        data = to_jsonld(json_ld, simplify=True)
        prompts = visit_json(data)
        
        logfile = kwargs.get("outfile") or f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json"
                
        document = kwargs["document"]
        doc_fs = open(document, "r")
        log_fs = open(logfile, "w+")
        try:
            log = json.load(log_fs) if os.stat(logfile).st_size > 0 else {}
            document_content = doc_fs.read()
            
            valids = 0
            idx = 0
            for prompt in prompts:                
                if isinstance(self.__retriever, AbstractRetrievalModel):
                    scores = self.__retriever.query(prompt, document=document_content)
                    print(scores)
                    #TODO: Need a way to return binary answer yes/no. Logistic Regression?
                    raise NotImplementedError()
                else:
                    if prompt not in log:

                        extended_prompt = textwrap.dedent(f"""
                        Given the document below
                        ```markdown
                        {document_content}
                        ```

                        Is the following element mentioned (explicitly or implicitly) in the document? 
                        Answer with "Yes" or "No" then explain.
                                            
                        ```text
                        {prompt}
                        ```
                        """)
                    
                        response = self.__retriever.query(extended_prompt, remember=False).strip()
                    
                        log[prompt] = {
                            "response": response
                        }
                    
                    match = re.search(r"^(Yes|No)\s*", log[prompt]["response"])
                    
                    if match is None: raise RuntimeError(f"Response must be Yes/No. Got: {repr(response)}")
                    if match.group(1) == "Yes": valids += 1
                    else: print(f"Invalid markup: {prompt}")
                
            log["score"] = valids / len(prompts)
            idx += 1
        finally:
            json.dump(log, log_fs)                    
            doc_fs.close()
            log_fs.close()

        return log["score"]
                    
class TypeConformanceLLMValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs.get("retriever")
        
    def validate(self, json_ld, **kwargs):
        
        prompt_base = textwrap.dedent(f"""
        - Given the schema.org markup element below:
                
        ```json
        %MARKUP%
        ```
                
        - Given the definition below:
                
        ```txt
        %DEFINITION%
        ```

        - Here are few examples of correct markups:

        %EXAMPLES%
                
        Does the markup element match the definition? 
        Answer with "Yes" or "No" then explain.
                
        """)
        
        # Recursively visit json file
        def visit_json(json_stub: dict):
            prompts = []
            ent_type = None
            for k, values in json_stub.items():
                if k == "@type":
                    ent_type = values  
                    continue

                # Recursively add prompt for dependant entities             
                if isinstance(values, dict):
                    prompts.extend(visit_json(values))
                else:  
                    markup = {str(k): schema_simplify(values[0])} if len(values) == 1 else {str(k): str([schema_simplify(v) for v in values])}
                    definition: dict = get_type_definition(ent_type, prop=str(k), simplify=True, include_comment=True)
                    
                    prompt = None
        
                    if len(definition) == 0:
                        definition = None
                        prompt = f"{str(k)} is not a property of {ent_type}"
                    else:
                        examples = get_schema_example(str(k))
                        examples = [ f"Example {i+1}:\n {example}" for i, example in enumerate(examples) ]
                        
                        prompt = ( prompt_base
                            .replace("%MARKUP%", str(markup))
                            .replace("%DEFINITION%", str(definition))
                            .replace("%EXAMPLES%", "\n".join(examples))
                        )
                    prompts.append((markup, definition, prompt)) 
            
            return prompts 
        
        data = to_jsonld(json_ld) 
        prompts = visit_json(data)
        logfile = kwargs.get("outfile") or f"{Path(json_ld).parent}/{Path(json_ld).stem}_type-llm.json"
                
        valids = 0 
        log_fs = open(logfile, "w+")
        try: 
            log = json.load(log_fs) if os.stat(logfile).st_size > 0 else {}
            # TODO mark the file name at the beginning   
            for markup, definition, prompt in prompts:
                
                key = str(markup)  
                response = None
                          
                if key not in log:                  
                    response = self.__retriever.query(prompt, remember=False).strip()
                    
                    log[key] = {
                        "definition": definition,
                        "response": response
                    }
                      
                # Count the correct answer    
                match = re.search(r"^(Yes|No)\s*", log[key]["response"])
                if match is None: raise RuntimeError(f"Response must be Yes/No. Got: {repr(response)}")
                                                                
                if match.group(1) == "Yes": valids += 1
                else: print(f"Invalid markup: {prompt}")
                
                json.dump(log, log_fs)      
    
            log["score"] = valids/len(prompts)
        finally:
            json.dump(log, log_fs)       
            log_fs.close()  
        
        return log["score"]   