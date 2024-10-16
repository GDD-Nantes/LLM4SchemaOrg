from collections import OrderedDict
from copy import deepcopy
import enum
import glob
from itertools import chain
import json
import os
from pathlib import Path
from pprint import pprint
import re
import textwrap
from typing import Literal, get_args
import backoff
from httpx import ReadTimeout
import numpy as np
from openai import APIError, APITimeoutError, RateLimitError
import pandas as pd
from pydantic import BaseModel, Field
from rdflib import BNode, ConjunctiveGraph
from tqdm import tqdm
from utils import BinaryPrediction, LlamaCPPError, chunk_document, get_infos, logger, collect_json, get_schema_example, get_type_definition, schema_simplify, schema_stringify, to_jsonld, transform_json

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
        report_summary_path = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.json")
        report_log_path = f"{Path(report_summary_path).parent}/{Path(report_summary_path).stem}.report"
        
        force_validate = kwargs.get("force_validate", False)

        dataGraph = ConjunctiveGraph()
        
        try: dataGraph.parse(json_ld)
        except UnboundLocalError as e:
            raise e
            
            update_and_dump_dict({
                "valid": False,
                "status": "parsing_error",
                "score": None
            }, report_summary_path)
            
            return None
                        
        # Write the clean message
        report = {
            "msgs": {},
            "score": None
        }
        
        # Load the report if exists
        if os.path.exists(report_summary_path) and os.stat(report_summary_path).st_size > 0 and not force_validate:
            logger.debug(f"Loading from {report_summary_path}")
            with open(report_summary_path, "r") as f:
                report = json.load(f)
        else:
            logger.debug(f"Writing to {report_summary_path}")
            # Check for OOV terms

            info = to_jsonld(json_ld)

            def check_oov(prop, value, ent_type):  
                prop_simplified = prop.replace("http://schema.org/", "")

                if ent_type is None:
                    return ["[TOK_Q_DELIM]".join((prop, str(value), str(ent_type)))]
                
                if isinstance(ent_type, str):
                    ent_type = [ent_type]

                result = []
                for et in ent_type:
                    et_simple = et.replace("http://schema.org/", "")
                    logger.debug(f"Checking {prop_simplified} {value} {et_simple}")
                    if et is not None and et not in report["msgs"]:
                        if len(get_type_definition(class_=str(et), exit_on_first=True)) == 0:
                            msg = f"{et_simple} is not a type defined by the schema."
                            logger.debug(f"{msg}")
                            if et_simple not in report["msgs"]:
                                report["msgs"][et_simple] = []
                            if msg not in report["msgs"][et_simple]:
                                report["msgs"][et_simple].append(msg)
                    result.append("[TOK_Q_DELIM]".join((prop, str(value), et)))
                return result

            logger.debug(f"Collecting info from {json_ld}")
            info_values = set(chain(*collect_json(info, value_transformer=check_oov)))
            report["n_infos"] = len(info_values)
            logger.warning(f"There are {len(info_values)} property-value pairs in {json_ld}!")
               
            # Validate using PySHACL or load the report if exists  
            if os.path.exists(report_log_path) and os.stat(report_log_path).st_size > 0 and not force_validate:
                logger.debug(f"Loading from {report_log_path}")
                try:
                    report_graph = ConjunctiveGraph()
                    report_graph.parse(report_log_path, format="turtle")
                except Exception as e:
                    force_validate = True
            else:
                force_validate = True
            
            if force_validate:    
                logger.debug(f"Validating {json_ld} with {shapeGraph}")
                _, report_graph, _ = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph, inference="both")
                logger.info(f"Writing to {report_summary_path}")
                try:
                    report_graph.serialize(report_log_path, format="turtle")
                except Exception as e:
                    if "does not look like a valid URI" in str(e):
                        pass
                    else:
                        raise e
            
            # Shape constraint validation
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
                resultPath_simple = schema_simplify(resultPath).replace("schema1:", "")
                sourceShape = stringify_node(report_graph, qres.get("sourceShape"))
                sourceShape_simple = schema_simplify(sourceShape).replace("schema1:", "")
                
                value = qres.get("value").toPython()

                query = "[TOK_Q_DELIM]".join([resultPath_simple, str(value), sourceShape_simple])
                                        
                node_info = f"( shape {sourceShape}, path {resultPath} )"
                message = str(resultMessage).strip()
                if message.startswith("Node"):
                    if "is closed. It cannot have value" in message:
                        message = re.sub(r"\[.*\]", node_info, message)
                        message = f"({resultPath_simple}) is not a property of ({sourceShape_simple})."
                elif message.startswith("Value"):
                    message = re.sub(r"Value", f"Node {node_info}: {value}", message)
                
                logger.debug(f"PySHACL: {message}")
                    
                if query not in report["msgs"]:
                    report["msgs"][query] = []
                
                if message not in report["msgs"][query]:
                    report["msgs"][query].append(message)
            
            # Clean up
            for k, v in report["msgs"].items():
                if len(v) == 0:
                    report["msgs"].pop(k)
        
        # Compute the score no matter what
        epsilon = 1e-6
        score = 1 - len(report["msgs"]) / (report["n_infos"] + epsilon ) 
        report["valid"] = ( len(report["msgs"]) == 0 )
        report["score"] = score

        update_and_dump_dict(report, report_summary_path)
                
        return score

def load_or_create_dict(file_path):
    if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}  # Return empty dictionary if file doesn't exist

def update_and_dump_dict(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, ensure_ascii=False, indent=4)
        
class ValidatorError(Exception):
    pass

class EmptyMarkupError(ValidatorError):
    pass
        
class FactualConsistencyValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        retriever = kwargs["retriever"]
        self.__retriever = retriever
    
    @backoff.on_exception(backoff.expo, (APITimeoutError, RateLimitError, APIError, ReadTimeout))
    def map_reduce_validate(self, json_ld, **kwargs):
        document_fn = kwargs["document"]
        explain_log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json")

        kwargs_copy = deepcopy(kwargs)
        kwargs_copy["explain"] = True
        kwargs_copy["outfile"] = f"{Path(explain_log_fn).parent}/{Path(explain_log_fn).stem}_explain.json" 
        
        with open(document_fn, "r") as f:
            content = f.read()
            prompt_estimate = self.validate(json_ld, data="", verbose=True, **kwargs_copy)
            
            # Estimate the maximum token count for prompt
            max_prompt_estimate_tok_count = -np.inf
            for chunk_id, res in prompt_estimate.items():
                
                if chunk_id.startswith("chunk_"):
                    for k, v in res.items():
                        if k in ["status", "score"]:
                            continue
                        prompt_estimate_tok_count = v["response"]
                        if prompt_estimate_tok_count > max_prompt_estimate_tok_count:
                            max_prompt_estimate_tok_count = prompt_estimate_tok_count
            
            # TODO: Take care of the case when chunk_tok_count_limit is negative
            logger.debug(f"context_windows_length={self.__retriever._context_windows_length}, max_output_length={self.__retriever._max_output_length}, max_prompt_estimate_tok_count={max_prompt_estimate_tok_count}")
            chunk_tok_count_limit = self.__retriever._context_windows_length - self.__retriever._max_output_length - max_prompt_estimate_tok_count

            content_tok_count = self.__retriever._estimator.estimate_tokens(content)
            logger.info(f"document_tokcount={content_tok_count}, chunk_tok_count_limit={chunk_tok_count_limit}")
            #raise RuntimeError()
                    
            if content_tok_count <= chunk_tok_count_limit:
                return self.validate(json_ld, data=content, verbose=False, **kwargs)
            
            # Generate chunks with overlapping
            chunks = chunk_document(content, chunk_tok_count_limit, self.__retriever._estimator)
            logger.info(f"Splitted into {len(chunks)} chunks!")

            # Validate each chunk
            log = None
            for i, chunk in tqdm(enumerate(chunks)):
                log = self.validate(json_ld, data=chunk, map_reduce_chunk=i, verbose=True, **kwargs)
                if log[f"chunk_{i}"].get("msgs") == "parsing_error":
                    return log[f"chunk_{i}"]["score"]

            # Aggregate the results
            log.pop("aggregation", None)
            final_score = ( 
                pd.DataFrame.from_dict(log, orient="index")
                .drop(columns=["status", "score"], errors="ignore")
                .fillna(False)
                .map(lambda x: (x["response"] if isinstance(x, dict) else x) == "TOKPOS" )
            ).apply(lambda x: x.any())
                        
            log["aggregation"] = final_score.to_dict()

            #pprint(log["aggregation"])
            log["aggregation"]["score"] = final_score.astype(int).mean()
            
            log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json")
            update_and_dump_dict(log, log_fn)
            
            return log["aggregation"]["score"]
        
    @backoff.on_exception(backoff.expo, (RecursionError, LlamaCPPError))
    def validate(self, json_ld, **kwargs):

        logger.info(f"{json_ld}")

        # Params        
        map_reduce_chunk = kwargs.pop("map_reduce_chunk", 0)
        map_reduce_chunk_key = "chunk_" + str(map_reduce_chunk)
        verbose = kwargs.pop("verbose", False)
        prompt_template_file = kwargs.pop("prompt_template")
        
        log_fn = kwargs.pop("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json")
        log = load_or_create_dict(log_fn)
                
        doc_fn = kwargs.pop("document")
        doc_fs = open(doc_fn, "r")
        doc_content = kwargs.pop("data", doc_fs.read())

        explain = kwargs.get("explain", False)
        probe = kwargs.pop("probe", False)
        force_validate = kwargs.pop("force_validate", False) or explain or probe

        try:
            data = to_jsonld(json_ld, simplify=True, clean=True)            
            infos = set(chain(*collect_json(data, value_transformer=get_infos)))
            if len(infos) == 0:
                raise EmptyMarkupError(f"Could not collect any prompt from {json_ld}!")
            
            if map_reduce_chunk_key not in log.keys():
                log[map_reduce_chunk_key] = {}
            
            # if doc_content.strip() == "":
            #     raise RuntimeError(f"Empty document {doc_fn}")
            
            valids = 0
            for query in infos:
                prop, value, parent_class = query.split("[TOK_Q_DELIM]")
                logger.info(f"Validating {prop} {value} {parent_class}")
                
                info = {prop: value}
                if parent_class is not None:
                    info.update({"@type": parent_class})

                # Check if previous chunk response is TOKPOS
                if map_reduce_chunk > 0:
                    previous_chunk = log[f"chunk_{map_reduce_chunk-1}"]
                    if query in previous_chunk:
                        previous_response = previous_chunk[query].get("response")
                        previous_prob = previous_chunk[query].get("prob")
                        logger.debug(f"Response for {query} in previous chunk is {previous_response}, type={type(previous_response)}")
                        if previous_response and previous_response == "TOKPOS":
                            logger.debug(f"Skipping evaluation for {query} on chunk {map_reduce_chunk}")
                            log[map_reduce_chunk_key]["status"] = "success"
                            log[map_reduce_chunk_key][query] = {
                                "query": f"prop={prop}, value={value}, parent_class={parent_class}",
                                "response": previous_response,
                                "prob": previous_prob
                            }
                        
                # If not, execute
                if query not in log[map_reduce_chunk_key] or force_validate:
                    logger.debug(f"Validating {query} on chunk {map_reduce_chunk}")
                    with open(prompt_template_file, "r") as f:
                        prompt_template = json.load(f, object_pairs_hook=OrderedDict)

                    prompt = OrderedDict()
                    for comp_name, comp_template in prompt_template.items():
                        comp_template = (
                            comp_template
                            .replace("[PARENT_CLASS]", str(parent_class))
                            .replace("[PROP]", str(prop))
                            .replace("[VALUE]", str(value))
                            .replace("of None?", "?")
                        )
                        if comp_name == "document":
                            prompt["document"] = comp_template.replace("[DOCUMENT]", doc_content)
                        # elif comp_name == "affirmation":
                        #     prompt["affirmation"] = comp_template.replace("[AFFIRMATION]", info)
                        else:
                            prompt[comp_name] = comp_template

                    # Use instructor to constraint LLM answer
                    search_classes = None if probe else [BinaryPrediction]
                    
                    # Use instructor to constraint answers
                    response = self.__retriever.query(
                        prompt, search_classes=search_classes, 
                        partial=False, explain=explain
                    )

                    response = response["prompt_tokens"] if explain else response[0].label if search_classes else response[0]
                    logger.info(f"Response: {response}")

                    # Log the response
                    log[map_reduce_chunk_key]["status"] = "success"
                    log[map_reduce_chunk_key][query] = {
                        "query": f"prop={prop}, value={value}, parent_class={parent_class}",
                        "response": response,
                        "prob": None if explain else response[1]
                    }
                
                if not explain and not probe:
                    if "TOKPOS" in log[map_reduce_chunk_key][query]["response"]: valids += 1                 
                    elif "TOKNEG" in log[map_reduce_chunk_key][query]["response"]: pass
                    else: raise RuntimeError(f"""Response must be TOKPOS/TOKNEG. Got: {repr(log[map_reduce_chunk_key][query]["response"])}""")
            
            log[map_reduce_chunk_key]["score"] = None if explain else valids / len(infos)

        except UnboundLocalError as e:
            raise e
            log = {
                map_reduce_chunk_key: {
                    "status": "parsing_error",
                    "score": None
                }
            }
        except EmptyMarkupError:
            log = {
                map_reduce_chunk_key: {
                    "status": "empty_markup_error",
                    "score": None
                }
            }
        finally:
            if not probe:
                update_and_dump_dict(log, log_fn)
            doc_fs.close()

        return log if verbose else log[map_reduce_chunk_key]["score"]
                    
class SemanticConformanceValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs["retriever"]
        
    def map_reduce_validate(self, json_ld, **kwargs):
        return self.validate(json_ld, **kwargs)
        
    def validate(self, json_ld, **kwargs):
        """Validate PropChecker.
        kwargs: 
        - in_context_learning: if True, inject examples in prompt.
        - chain_of_thought: if True, use chain of thought method.
        - expert: if True, use expert method
        """
        
        # Params        
        probe = kwargs.get("probe", False)
        force_validate = kwargs.get("force_validate", False) or probe
        map_reduce_chunk = "chunk_" + str(kwargs.get("map_reduce_chunk", 0))
        verbose = kwargs.get("verbose", False)
        prompt_template_file = kwargs.get("prompt_template")
                          
        log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_semantic.json") 
        log = load_or_create_dict(log_fn)
        
        try: 
            data = to_jsonld(json_ld, simplify=True, clean=True)            
            infos = set(chain(*collect_json(data, value_transformer=get_infos))   )
                                    
            if map_reduce_chunk not in log.keys():
                log[map_reduce_chunk] = {}
            
            #TODO Error management: raise it or warn it?
            if len(infos) == 0:
                raise EmptyMarkupError(f"Could not generate prompt for {json_ld} because there is no workable attributes")
            
            valids = 0 
            for query in infos:

                prop, value, parent_class = query.split("[TOK_Q_DELIM]")

                logger.debug(f"Validating {prop} {value} {parent_class}")                
                info = json.dumps({prop: value})

                prop_url = f"http://schema.org/{prop}" if not prop.startswith("http") else prop

                definition: dict = get_type_definition(class_=parent_class, prop=prop_url, simplify=True, include_comment=True)
                logger.debug(f"{prop} {definition}")

            
                if len(definition) == 0:
                    definition = None
                    logger.warning(f"{str(prop)} is not a property of {parent_class}")
                    continue

                definition = f'{prop}: {definition.popitem()[1]["comment"]}'

                with open(prompt_template_file, "r") as f:
                    prompt_template = json.load(f, object_pairs_hook=OrderedDict)

                prompt = OrderedDict()
                for comp_name, comp_template in prompt_template.items():
                    if comp_name == "markup":
                        prompt["markup"] = (
                            comp_template
                            .replace("[MARKUP]", info)
                            .replace("[PROP]", prop)
                        )
                    elif comp_name == "definition":
                        prompt["definition"] = (
                            comp_template
                            .replace("[DEFINITION]", definition)
                            .replace("[PROP]", prop)
                        )
                    elif comp_name == "example":
                        examples = get_schema_example(prop, focus=True)
                        for i, example in enumerate(examples):
                            prompt["pos-example"] = (
                                comp_template
                                .replace("[EXAMPLE_ID]", i)
                                .replace("[EXAMPLE_MARKUP]", example)
                            )
                    else:
                        prompt[comp_name] = comp_template
                
                response = None
                                          
                if query not in log[map_reduce_chunk] or force_validate:  
                    # Use instructor to constraint LLM answer
                    search_classes = None if probe else [BinaryPrediction]
                    response, prob = self.__retriever.query(
                        prompt, search_classes=search_classes, partial=False
                    )

                    response = response.label if search_classes else response   
                    logger.info(f"Response: {response}")

                    # response = response.strip()
                    log[map_reduce_chunk]["status"] = "success"
                    log[map_reduce_chunk][query] = {
                        "query": definition,
                        "response": response,
                        "prob": prob
                    }
                else:
                    response = log[map_reduce_chunk][query]["response"]
                      
                # Count the correct answer  
                if not probe:  
                    if "TOKPOS" in log[map_reduce_chunk][query]["response"]: valids += 1                 
                    elif "TOKNEG" in log[map_reduce_chunk][query]["response"]: pass
                    else: raise RuntimeError(f'Response must be TOKPOS/TOKNEG. Got: {repr(log[map_reduce_chunk][query]["response"])}')

            log[map_reduce_chunk]["score"] = valids/len(infos)
        except UnboundLocalError as e:
            raise e
            log = {
                map_reduce_chunk: {
                    "status": "parsing_error",
                    "score": None
                }
            }
        except EmptyMarkupError:
            log = {
                map_reduce_chunk: {
                    "status": "empty_markup_error",
                    "score": None
                }
            }
        finally:
            if not probe:
                update_and_dump_dict(log, log_fn)       
        
        return log if verbose else log[map_reduce_chunk]["score"]   
    
class SameAsValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__retriever = kwargs["retriever"]
        
    def validate(self, json_ld, **kwargs):

        promt_template_file = kwargs.get("prompt_template", "prompts/validation/sameas.json")
        with open(promt_template_file, "r") as f:
            prompt_template = json.load(f)
        
        pred = to_jsonld(json_ld, simplify=True)
        expected = to_jsonld(kwargs.get("expected_file"), simplify=True)
        
        prompt = OrderedDict()
        for comp_name, comp_template in prompt_template.items():
            if comp_name == "markupA":
                prompt["markupA"] = comp_template.replace("[MARKUP_A]", json.dumps(pred, ensure_ascii=False))
            elif comp_name == "markupB":
                prompt["markupB"] = comp_template.replace("[MARKUP_B]", json.dumps(expected, ensure_ascii=False))
            else:
                prompt[comp_name] = comp_template
        
        response, prob = self.__retriever.query(prompt, stream=True, search_classes=[BinaryPrediction], partial=False, stop=list(get_args(BinaryPrediction.model_fields["label"].annotation)))
        
        if "TOKPOS" in response:
            return True
        elif "TOKNEG" in response:
            return False
        
        raise ValueError(f"""Response must be either "TOKPOS" or "TOKNEG", response = {response} """)
        