from collections import OrderedDict
import json
import os
from pathlib import Path
from pprint import pprint
import re
import textwrap
from rdflib import BNode, ConjunctiveGraph
from utils import collect_json, get_schema_example, get_type_definition, schema_simplify, schema_stringify, to_jsonld, transform_json
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
        dataGraph = ConjunctiveGraph()
        print(json_ld)
        dataGraph.parse(json_ld)
        valid, report_graph, report_msgs = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph, inference="both")
        report_path = kwargs.get("outfile") or f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.json"
        print(f"Writing to {report_path}")
        # report_graph.serialize(report_path, format="turtle")
        
        print(report_msgs)
        
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
                    message = f"({schema_simplify(resultPath)}) is not a property of ({schema_simplify(sourceShape)})."
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
        def __write_prompt(key, values, ent_type):
            return f"{ent_type} {schema_simplify(key)} {schema_simplify(values)}" 
        
        print(json_ld)
        data = to_jsonld(json_ld, simplify=True)
        prompts = collect_json(data, value_transformer=__write_prompt)
                        
        if len(prompts) == 0:
            raise ValueError(f"Could not collect any prompt from {json_ld}!")
        
        logfile = kwargs.get("outfile") or f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json"
                
        document = kwargs["document"]
        doc_fs = open(document, "r")
        log_fs = open(logfile, "w+")
        try:
            log = json.load(log_fs) if os.stat(logfile).st_size > 0 else {}
            document_content = doc_fs.read()
            
            valids = 0
            for prompt in prompts:    
                if prompt is None: continue            
                if isinstance(self.__retriever, AbstractRetrievalModel):
                    scores = self.__retriever.query(prompt, document=document_content)
                    print(scores)
                    #TODO: Need a way to return binary answer yes/no. Logistic Regression?
                    raise NotImplementedError()
                else:
                    print(prompt)
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
        finally:
            json.dump(log, log_fs)                    
            doc_fs.close()
            log_fs.close()

        return log["score"]
                    
class SemanticConformanceValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs.get("retriever")
        
    def validate(self, json_ld, **kwargs):
        """Validate PropChecker.
        kwargs: 
        - in_context_learning: if True, inject examples in prompt.
        - chain_of_thought: if True, use chain of thought method.
        - expert: if True, use expert method
        """
        
        # Params        
        in_context_learning =  kwargs.get("in_context_learning")
        if in_context_learning is None: in_context_learning = False
        
        chain_of_thought =  kwargs.get("chain_of_thought")
        if chain_of_thought is None: chain_of_thought = False
        
        expert =  kwargs.get("expert")
        if expert is None: expert = False
        
        print(json_ld)
        
        # Recursively visit json file
        def __write_prompt(key, values, ent_type):
            
            markup = schema_stringify({key: values})
            definition: dict = get_type_definition(ent_type, prop=str(key), simplify=True, include_comment=True)
                                        
            prompt = None
        
            if len(definition) == 0:
                definition = None
                # prompt = f"{str(key)} is not a property of {ent_type}"
                raise ValueError(f"{str(key)} is not a property of {ent_type}")
            else:
                definition = f'{schema_simplify(key)}: {definition.popitem()[1]["comment"]}'
                
                examples = ["NO EXAMPLE"]
                if in_context_learning:
                    examples = get_schema_example(key, focus=True)
                    examples = "\n".join([ f"Example {i+1}:\n ```json\n{example}\n```" for i, example in enumerate(examples) ])
                        
                prompt = OrderedDict({
                    "expert": "You are an expert in the semantic web and have deep knowledge about writing schema.org markup.",
                    "context1": textwrap.dedent(f"""
                        - Given the markup below:        
                        ```json
                        {str(markup)}
                        ```
                    """),
                    "cot2": "In one sentence, what does the markup describe ?",
                    "context2": textwrap.dedent(f"""
                        - Given the definition below:      
                        ```txt
                        {str(definition)}
                        ```
                    """),
                    "examples": textwrap.dedent(f"""
                        - Here are some positive examples:
                        ```
                        {examples}
                        ```
                    """),
                    "task": textwrap.dedent("""
                        Does the value align with the property definition?  
                        Answer with either "Yes" or "No".
                    """)
                    
                })
                
                if not expert:
                    prompt.pop("expert")
                
                if not in_context_learning:
                    prompt.pop("examples")
                
                if not chain_of_thought:
                    prompt = { k: v for k, v in prompt.items() if not k.startswith("cot") }
            
            return markup, definition, prompt
        
        data = to_jsonld(json_ld)
        prompts = collect_json(data, value_transformer=__write_prompt)
                
        logfile = kwargs.get("outfile") or f"{Path(json_ld).parent}/{Path(json_ld).stem}_semantic.json"        
        valids = 0 
        log_fs = open(logfile, "w+")
        try: 
            log = json.load(log_fs) if os.stat(logfile).st_size > 0 else {}
            
            #TODO Error management: raise it or warn it?
            if len(prompts) == 0:
                print(data)
                raise RuntimeError(f"Could not generate prompt for {json_ld} because there is no workable attributes")
            
            # TODO mark the file name at the beginning   
            for markup, definition, prompt in prompts:
                
                key = str(markup)  
                response = None
                
                if definition is None:
                    print(prompt)
                    continue
                          
                if key not in log:                      
                    response = self.__retriever.chain_of_thoughts(prompt) if chain_of_thought else self.__retriever.query("\n".join(prompt.values()), remember=False)
                    response = response.strip()
                    
                    log[key] = {
                        "definition": definition,
                        "response": response
                    }
                      
                # Count the correct answer    
                match = re.search(r"^(Yes|No)\s*", log[key]["response"])
                if match is None: raise RuntimeError(f"Response must be Yes/No. Got: {repr(log[key]['response'])}")
                                                                
                if match.group(1) == "Yes": valids += 1
                # else: print(f"Invalid markup: {prompt}")

            log["valids"] = valids
            log["score"] = valids/len(prompts)
        finally:
            json.dump(log, log_fs)       
            log_fs.close()  
        
        return log["score"]   
    
class SameAsValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs.get("retriever")
        
    def validate(self, json_ld, **kwargs):
        
        pred = to_jsonld(json_ld, simplify=True)
        expected = to_jsonld(kwargs.get("expected_file"), simplify=True)
        
        prompt = textwrap.dedent(f"""
        Do the following two entity description match? 
        Answer with "Yes" if they do and "No" if they do not.
        
        Entity A:
        ```json
        {pred}
        ```
        
        Entity B:
        ```json
        {expected}
        ```
        
        """)
        
        response = self.__retriever.query(prompt).strip()
        
        if (re.search(r"^(Yes|No)\s*", response) is None):
            raise ValueError("Answer must be either 'Yes' or 'No'")
        
        
        return response == "Yes"