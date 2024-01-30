from collections import OrderedDict
import json
import os
from pathlib import Path
from pprint import pprint
import re
import textwrap
import pandas as pd
from rdflib import BNode, ConjunctiveGraph
from utils import logger, collect_json, get_schema_example, get_type_definition, schema_simplify, schema_stringify, to_jsonld, transform_json
from models.retrieval import *

import pyshacl
from pyshacl.rdfutil import stringify_node
from llm_cost_estimation import count_tokens, models, estimate_cost

class AbstractValidator:
    def __init__(self, **kwargs) -> None:
        pass

    def map_reduce_validate(self, json_ld, n_chunks=5, aggregator=lambda x: sum(x)/len(x), **kwargs):
        """Perform the task `validate` in Map-Reduce manner. The input will be divided into chunks. 
        Each chunk will be evaluated. The overall score will be an aggregation of each chunk's evaluation.

        Args:
            json_ld (_type_): _description_
            n_chunks (int, optional): _description_. Defaults to 5.
            aggregator (_type_, optional): _description_. Defaults to lambdax:sum(x)/len(x).

        Returns:
            _type_: _description_
        """
        
        document_fn = kwargs["document"]
        with open(document_fn, "r") as f:
            document = f.read()
            tok_count, _ = count_tokens(document, "gpt-4")
            logger.info(f"There are {tok_count} tokens in {document_fn}!")

            if tok_count <= 10000:
                return self.validate(json_ld, **kwargs)

            sents = nltk.sent_tokenize(document)
            logger.debug(f"Broken down to {len(sents)} sentences")       
                 
            for i, chunk in enumerate(range(n_chunks)):
                lower = i*chunk
                upper = min((i+1)*chunk, len(sents))
                logger.debug(sents[lower:upper])
                content = "\n".join(sents[lower:upper])
                log = self.validate(json_ld, data=content, map_reduce_chunk=i, verbose=True, **kwargs)
            
            final_score = ( 
                pd.DataFrame.from_dict(log, orient="index")
                .fillna(False)
                .map(lambda x: (x["response"] if isinstance(x, dict) else x) == "TOKPOS" )
            ).apply(lambda x: x.any()).astype(int).mean()
            return final_score

    
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

    def map_reduce_validate(self, json_ld, n_chunks=5, aggregator=lambda x: sum(x) / len(x), **kwargs):
        raise NotImplementedError("Cannot perform map reduce for ShaclValidator!")
            
    def validate(self, json_ld, **kwargs) -> ConjunctiveGraph:
        """Validate syntaxically JSON-LD.

        Args:
            json_ld (_type_): _description_

        Returns:
            ConjunctiveGraph: _description_
        """
        shapeGraph = self.__shape_graph
        dataGraph = ConjunctiveGraph()
        logger.info(json_ld)
        dataGraph.parse(json_ld)
        valid, report_graph, report_msgs = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph, inference="both")
        report_path = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.json")
        logger.info(f"Writing to {report_path}")
        # report_graph.serialize(report_path, format="turtle")
        
        logger.info(report_msgs)
        
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
            json.dump(report, f, ensure_ascii=False)
        
        return report

def load_or_create_dict(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}  # Return empty dictionary if file doesn't exist

def update_and_dump_dict(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, ensure_ascii=False, indent=4)
        
class FactualConsistencyValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        retriever = kwargs["retriever"]
        if isinstance(retriever, str):
            self.__retriever: AbstractRetrievalModel = globals()[retriever]()
        else:
            self.__retriever = retriever
    
    def map_reduce_validate(self, json_ld, n_chunks=5, aggregator=lambda x: sum(x) / len(x), **kwargs):
        return super().map_reduce_validate(json_ld, n_chunks, aggregator, **kwargs)
        
    def validate(self, json_ld, **kwargs):

        # Params        
        in_context_learning =  kwargs.get("in_context_learning", False)
        chain_of_thought =  kwargs.get("chain_of_thought", False)
        expert =  kwargs.get("expert", False)
        force_validate = kwargs.get("force_validate", False)
        map_reduce_chunk = "chunk_" + str(kwargs.get("map_reduce_chunk", 0))
        verbose = kwargs.get("verbose", False)

        def __write_prompt(key, values, ent_type):
            if ent_type is None:
                return f"{key} {values}" 
            else:
                return f"{ent_type} {key} {values}" 
        
        logger.info(json_ld)
        data = to_jsonld(json_ld, simplify=True, clean=True)
        infos = collect_json(data, value_transformer=__write_prompt)
                        
        if len(infos) == 0:
            raise ValueError(f"Could not collect any prompt from {json_ld}!")
        
        log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json")
                
        doc_fn = kwargs["document"]
        doc_fs = open(doc_fn, "r")
        try:
            log = load_or_create_dict(log_fn)
            
            if map_reduce_chunk not in log.keys():
                log[map_reduce_chunk] = {}
            
            doc_content = kwargs.get("data", doc_fs.read())
            if doc_content.strip() == "":
                print()
                raise RuntimeError(f"Empty document {doc_fn}")
            
            valids = 0
            for info in infos:    
                if info is None: continue            
                if isinstance(self.__retriever, AbstractRetrievalModel):
                    scores = self.__retriever.query(info, document=doc_content)
                    logger.info(scores)
                    #TODO: Need a way to return binary answer TOKPOS/no. Logistic Regression?
                    raise NotImplementedError()
                else:
                    logger.info(info)

                    if info not in log[map_reduce_chunk] or force_validate:

                        prompt = OrderedDict({
                            "expert": "You are an expert in the semantic web and have deep knowledge about writing schema.org markup.",
                            "context1": textwrap.dedent(f"""
                                Given the document below
                                ```markdown
                                {doc_content}
                                ```
                            """),
                            "context2": textwrap.dedent(f"""
                                Given the information below:
                                                    
                                ```text
                                {info}
                                ```
                            """),
                            "task": textwrap.dedent("""
                                Is the information mentioned (explicitly or implicitly) in the document? 
                                Answer "TOKPOS" if the information is mentioned or "TOKNEG" if not.
                            """)
                            
                        })
                
                        if not expert:
                            prompt.pop("expert")
                        
                        # if not in_context_learning:
                        #     prompt.pop("examples")
                        
                        if not chain_of_thought:
                            prompt = { k: v for k, v in prompt.items() if not k.startswith("cot") }
                    
                        response = self.__retriever.chain_of_thoughts(prompt) if chain_of_thought else self.__retriever.query(prompt, remember=False)
                        response = response.strip()
                    
                        log[map_reduce_chunk][info] = {
                            "response": response
                        }
                    
                    if "TOKPOS" in log[map_reduce_chunk][info]["response"]: valids += 1                 
                    elif "TOKNEG" in log[map_reduce_chunk][info]["response"]: pass
                    else: raise RuntimeError(f"Response must be TOKPOS/TOKNEG. Got: {repr(response)}")
         
            log[map_reduce_chunk]["score"] = valids / len(infos)
        finally:
            update_and_dump_dict(log, log_fn)
            doc_fs.close()

        return log if verbose else log[map_reduce_chunk]["score"]
                    
class SemanticConformanceValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs["retriever"]
        
    def map_reduce_validate(self, json_ld, n_chunks=5, aggregator=lambda x: sum(x) / len(x), **kwargs):
        return super().map_reduce_validate(json_ld, n_chunks, aggregator, **kwargs)
        
    def validate(self, json_ld, **kwargs):
        """Validate PropChecker.
        kwargs: 
        - in_context_learning: if True, inject examples in prompt.
        - chain_of_thought: if True, use chain of thought method.
        - expert: if True, use expert method
        """
        
        # Params        
        in_context_learning =  kwargs.get("in_context_learning", False)
        chain_of_thought =  kwargs.get("chain_of_thought", False)        
        expert =  kwargs.get("expert", False)
        force_validate = kwargs.get("force_validate", False)
        map_reduce_chunk = "chunk_" + str(kwargs.get("map_reduce_chunk", 0))
        verbose = kwargs.get("verbose", False)
        
        logger.info(json_ld)
        
        # Recursively visit json file
        def __write_prompt(key, values, ent_type):
            
            markup = schema_stringify({key: values})
            definition: dict = get_type_definition(ent_type, prop=str(key), simplify=True, include_comment=True)
                                        
            prompt = None
        
            if len(definition) == 0:
                definition = None
                logger.warning(f"{str(key)} is not a property of {ent_type}")
                # raise ValueError(f"{str(key)} is not a property of {ent_type}")
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
                        Answer with "TOKPOS" if the value aligns with the definition or "TOKNEG" if not.
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
                
        log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_semantic.json") 
        try: 
            log = load_or_create_dict(log_fn)
            
            if map_reduce_chunk not in log.keys():
                log[map_reduce_chunk] = {}
            
            #TODO Error management: raise it or warn it?
            if len(prompts) == 0:
                logger.error(data)
                raise RuntimeError(f"Could not generate prompt for {json_ld} because there is no workable attributes")
            
            valids = 0 
            for markup, definition, prompt in prompts:
                
                info = str(markup)  
                response = None
                
                if definition is None:
                    logger.warning(prompt)
                    continue
                          
                if info not in log[map_reduce_chunk] or force_validate:                   
                    response = self.__retriever.chain_of_thoughts(prompt) if chain_of_thought else self.__retriever.query(prompt, remember=False)
                    response = response.strip()
                                                            
                    log[map_reduce_chunk][info] = {
                        "definition": definition,
                        "response": response
                    }
                      
                # Count the correct answer    
                if "TOKPOS" in log[map_reduce_chunk][info]["response"]: 
                    valids += 1                 
                elif "TOKNEG" in log[map_reduce_chunk][info]["response"]: 
                    pass
                else: raise RuntimeError(f"Response must be TOKPOS/TOKNEG. Got: {repr(response)}")

            log[map_reduce_chunk]["score"] = valids/len(prompts)
        finally:
            update_and_dump_dict(log, log_fn)       
        
        return log if verbose else log[map_reduce_chunk]["score"]   
    
class SameAsValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs["retriever"]
        
    def validate(self, json_ld, **kwargs):
        
        pred = to_jsonld(json_ld, simplify=True)
        expected = to_jsonld(kwargs.get("expected_file"), simplify=True)
        
        prompt = textwrap.dedent(f"""
        Do the following two entity description match? 
        Answer with "TOKPOS" if they do and "TOKNEG" if they do not.
        
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
        
        # TODO: need fix
        if (re.search(r"^(TOKPOS|No)\s*", response) is None):
            raise ValueError("Answer must be either 'TOKPOS' or 'TOKNEG'")
        
        
        return response == "TOKPOS"