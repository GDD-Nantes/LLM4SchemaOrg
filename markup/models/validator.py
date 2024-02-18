from collections import OrderedDict
import enum
import json
import os
from pathlib import Path
from pprint import pprint
import re
import textwrap
import pandas as pd
from pydantic import BaseModel
from rdflib import BNode, ConjunctiveGraph
from utils import chunk_document, logger, collect_json, get_schema_example, get_type_definition, schema_simplify, schema_stringify, to_jsonld, transform_json

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
        report_path = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_shacl.json")
        
        force_validate = kwargs.get("force_validate", False)

        def dump_log(report):
            with open(report_path, "w") as f:
                json.dump(report, f, ensure_ascii=False)

        dataGraph = ConjunctiveGraph()
        
        try: dataGraph.parse(json_ld)
        except UnboundLocalError as e:
            raise e
            
            dump_log({
                "valid": False,
                "status": "parsing_error",
                "score": None
            })
            
            return None
                
        valid, report_graph, report_msgs = pyshacl.validate(data_graph=dataGraph, shacl_graph=shapeGraph, inference="both")
        logger.info(f"Writing to {report_path}")
        # report_graph.serialize(report_path, format="turtle")
        
        logger.info(report_msgs)
        
        # Write the clean message
        report = {
            "valid": valid,
            "msgs": {},
            "score": None
        }
        
        if os.path.exists(report_path) and os.stat(report_path).st_size > 0 and not force_validate:
            with open(report_path, "r") as f:
                report = json.load(f)
                return report["score"]
        
        info = to_jsonld(json_ld)
        info_values = collect_json(info, value_transformer=lambda k,v,e: (k, v, e))
        
        # Check for OOV terms
        for prop, _, ent_type in info_values:
            # logger.debug(f"{prop}, {ent_type}")
            if ent_type is not None:
                for et in ent_type:
                    if len(get_type_definition(class_=str(et))) == 0:
                        msg = f"{et} is not a type defined by the schema."
                        if et not in report["msgs"]:
                            report["msgs"][et] = []
                        if msg not in report["msgs"][et]:
                            report["msgs"][et].append(msg)
            
            if len(get_type_definition(prop=prop)) == 0:
                msg = f"{prop} is not a type defined by the schema."
                if prop not in report["msgs"]:
                    report["msgs"][prop] = []
                report["msgs"][prop].append(msg)
        
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
            resultPath_simple = schema_simplify(resultPath)
            sourceShape = stringify_node(report_graph, qres.get("sourceShape"))
            sourceShape_simple = schema_simplify(sourceShape)
            
            value = qres.get("value").toPython()
                                    
            node_info = f"( shape {sourceShape}, path {resultPath} )"
            message = str(resultMessage).strip()
            if message.startswith("Node"):
                if "is closed. It cannot have value" in message:
                    message = re.sub(r"\[.*\]", node_info, message)
                    message = f"({resultPath_simple}) is not a property of ({sourceShape_simple})."
            elif message.startswith("Value"):
                message = re.sub(r"Value", f"Node {node_info}: {value}", message)
                
            if resultPath_simple not in report["msgs"]:
                report["msgs"][resultPath_simple] = []
            
            if message not in report["msgs"][resultPath_simple]:
                report["msgs"][resultPath_simple].append(message)
        
        # Clean up
        for k, v in report["msgs"].items():
            if len(v) == 0:
                report["msgs"].pop(k)
        if len(info_values) != 0:
            score = 1-len(report["msgs"])/len(info_values)
        score = 1-len(report["msgs"])

        report["valid"] = report["valid"] and ( len(report["msgs"]) == 0 )
        report["score"] = score

        dump_log(report)
                
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

class BinaryLabels(str, enum.Enum):
    TOKPOS="TOKPOS"
    TOKNEG="TOKNEG"

class BinaryPrediction(BaseModel):
    prediction: BinaryLabels
        
class FactualConsistencyValidator(AbstractValidator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        retriever = kwargs["retriever"]
        self.__retriever = retriever
    
    def map_reduce_validate(self, json_ld, chunk_size_limit=2000, **kwargs):
        document_fn = kwargs["document"]
        with open(document_fn, "r") as f:
            document = f.read()
            tok_count, _ = count_tokens(document, "gpt-4")
            logger.info(f"There are {tok_count} tokens in {document_fn}!")

            if tok_count <= chunk_size_limit:
                return self.validate(json_ld, **kwargs)

            chunks = chunk_document(document, chunk_size_limit, self.__retriever._estimator)
            for i, chunk in enumerate(chunks):
                log = self.validate(json_ld, data=chunk, map_reduce_chunk=i, verbose=True, **kwargs)
                if log[f"chunk_{i}"].get("msgs") == "parsing_error":
                    return log[f"chunk_{i}"]["score"]
                            
            final_score = ( 
                pd.DataFrame.from_dict(log, orient="index")
                .fillna(False)
                .map(lambda x: (x["response"] if isinstance(x, dict) else x) == "TOKPOS" )
            ).apply(lambda x: x.any())
                        
            log["aggregation"] = final_score.to_dict()
            log["aggregation"]["score"] = final_score.astype(int).mean()
            
            log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json")
            with open(log_fn, "w") as f:
                json.dump(log, f, ensure_ascii=False)
            
            return log["aggregation"]["score"]
        
    def validate(self, json_ld, **kwargs):

        # Params        
        in_context_learning =  kwargs.get("in_context_learning", False)
        chain_of_thought =  kwargs.get("chain_of_thought", False)
        chain_prompt =  kwargs.get("chain_prompt", False)
        expert =  kwargs.get("expert", False)
        force_validate = kwargs.get("force_validate", False)
        map_reduce_chunk = "chunk_" + str(kwargs.get("map_reduce_chunk", 0))
        verbose = kwargs.get("verbose", False)
        prompt_template_file = kwargs.get("prompt_template")
        
        logger.info(f"{json_ld}")
                
        log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_factual.json")
        log = load_or_create_dict(log_fn)
                
        doc_fn = kwargs["document"]
        doc_fs = open(doc_fn, "r")
        try:
            data = to_jsonld(json_ld, simplify=True, clean=True)
            infos = collect_json(data, value_transformer=lambda k,v,e: (k,v,e))
                            
            if len(infos) == 0:
                raise EmptyMarkupError(f"Could not collect any prompt from {json_ld}!")
            
            if map_reduce_chunk not in log.keys():
                log[map_reduce_chunk] = {}
            
            doc_content = kwargs.get("data", doc_fs.read())
            if doc_content.strip() == "":
                print()
                raise RuntimeError(f"Empty document {doc_fn}")
            
            valids = 0
            for prop, value, parent_class in infos:    
                
                info = {prop: value}
                if parent_class is not None:
                    info.update({"@type": parent_class})
                
                info = (
                    json.dumps(info) if chain_prompt or parent_class is None else 
                    f"There is a {parent_class} with {prop} {value}"
                )
       
                if prop not in log[map_reduce_chunk] or force_validate:

                    with open(prompt_template_file, "r") as f:
                        prompt_template = json.load(f, object_pairs_hook=OrderedDict)

                    prompt = OrderedDict()
                    for comp_name, comp_template in prompt_template.items():
                        comp_template = (
                            comp_template
                            .replace("[PARENT_CLASS]", parent_class)
                            .replace("[PROP]", prop)
                            .replace("[VALUE]", value)
                        )
                        if comp_name == "document":
                            prompt["document"] = comp_template.replace("[DOCUMENT]", doc_content)
                        elif comp_name == "affirmation":
                            prompt["affirmation"] = comp_template.replace("[AFFIRMATION]", info)
                        else:
                            prompt[comp_name] = comp_template
                
                    response = (
                        self.__retriever.chain_query(prompt, verbose=True) if chain_prompt else 
                        self.__retriever.query(prompt, stream=True, search_classes=[BinaryPrediction], partial=False, stop=list(BinaryLabels.__members__.values()))
                    )
                    response = response.strip()
                    log[map_reduce_chunk]["status"] = "success"
                    log[map_reduce_chunk][prop] = {
                        "query": prompt["task"],
                        "response": response
                    }
                
                if "TOKPOS" in log[map_reduce_chunk][prop]["response"]: valids += 1                 
                elif "TOKNEG" in log[map_reduce_chunk][prop]["response"]: pass
                else: raise RuntimeError(f"""Response must be TOKPOS/TOKNEG. Got: {repr(log[map_reduce_chunk][prop]["response"])}""")
         
            log[map_reduce_chunk]["score"] = valids / len(infos)
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
            update_and_dump_dict(log, log_fn)
            doc_fs.close()

        return log if verbose else log[map_reduce_chunk]["score"]
                    
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
        in_context_learning =  kwargs.get("in_context_learning", False)
        chain_prompt = kwargs.get("chain_prompt", False)
        chain_of_thought =  kwargs.get("chain_of_thought", False)        
        expert =  kwargs.get("expert", False)
        force_validate = kwargs.get("force_validate", False)
        map_reduce_chunk = "chunk_" + str(kwargs.get("map_reduce_chunk", 0))
        verbose = kwargs.get("verbose", False)
        prompt_template_file = kwargs.get("prompt_template")
                          
        log_fn = kwargs.get("outfile", f"{Path(json_ld).parent}/{Path(json_ld).stem}_semantic.json") 
        log = load_or_create_dict(log_fn)
        
        try: 
            data = to_jsonld(json_ld, simplify=True, clean=True)
            infos = collect_json(data, value_transformer=lambda k,v,e: (k,v,e))
                                    
            if map_reduce_chunk not in log.keys():
                log[map_reduce_chunk] = {}
            
            #TODO Error management: raise it or warn it?
            if len(infos) == 0:
                raise EmptyMarkupError(f"Could not generate prompt for {json_ld} because there is no workable attributes")
            
            valids = 0 
            for prop, value, parent_class in infos:
                
                info = json.dumps({prop: value})
                definition: dict = get_type_definition(parent_class, prop=f"http://schema.org/{prop}", simplify=True, include_comment=True)
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
                                          
                if prop not in log[map_reduce_chunk] or force_validate:                   
                    response = (
                        self.__retriever.chain_query(prompt) if chain_prompt 
                        else self.__retriever.query(prompt, stream=True, search_classes=[BinaryPrediction], partial=False, stop=list(BinaryLabels.__members__.values()))
                    )
                    response = response.strip()
                    log[map_reduce_chunk]["status"] = "success"
                    log[map_reduce_chunk][prop] = {
                        "query": info,
                        "definition": definition,
                        "response": response
                    }
                else:
                    response = log[map_reduce_chunk][prop]["response"]
                      
                # Count the correct answer    
                if "TOKPOS" in log[map_reduce_chunk][prop]["response"]: 
                    valids += 1                 
                elif "TOKNEG" in log[map_reduce_chunk][prop]["response"]: 
                    pass
                else: raise RuntimeError(f"Response must be TOKPOS/TOKNEG. Got: {repr(response)}")

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
        
        response = self.__retriever.query(prompt, stream=True, search_classes=[BinaryPrediction], partial=False, stop=list(BinaryLabels.__members__.values()))
        
        if "TOKPOS" in response:
            return True
        elif "TOKNEG" in response:
            return False
        
        raise ValueError(f"""Response must be either "TOKPOS" or "TOKNEG", response = {response} """)
        