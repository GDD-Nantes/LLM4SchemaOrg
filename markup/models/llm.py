from collections import OrderedDict
from copy import deepcopy
import json
from math import ceil
import os
from pathlib import Path
from pprint import pprint
import re
import textwrap
from typing import Dict
import numpy as np
import pandas as pd

import openai
from openai.error import RateLimitError, ServiceUnavailableError, Timeout, APIError
from openai.embeddings_utils import get_embedding

from rdflib import ConjunctiveGraph, URIRef
import torch
import yaml
from models.validator import ValidatorFactory
from utils import extract_json, logger, collect_json, extract_preds, filter_graph, get_schema_example, get_type_definition, lookup_schema_type, schema_simplify, to_jsonld, chunk_document,scrape_webpage

from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker

from nltk import bleu, meteor, nist, chrf, gleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.stem import WordNetLemmatizer
from rouge_score import rouge_scorer

from scipy.spatial.distance import cosine, jaccard
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data if you haven't already
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from hugchat import hugchat
from hugchat.login import Login
from hugchat.exceptions import ChatError

from ftlangdetect import detect as lang_detect

import pycountry
import backoff

from llm_cost_estimation import count_tokens, models, estimate_cost
from llama_cpp import Llama, llama_get_embeddings

LLM_MODEL_CACHEDIR=".models"
LLM_CACHE = {}
LLM_CACHE_FILENAME = ".cache/llm_cache.json"
if os.path.exists(LLM_CACHE_FILENAME):
    with open(LLM_CACHE_FILENAME, "r") as f:
        LLM_CACHE = json.load(f)

LLAMA_CPP_CONFIG = "configs/llama_cpp.yaml"

def preprocess_text(text: str):
    lang = lang_detect(text.replace("\n", ""))["lang"]
    if pycountry.languages.get(alpha_2=lang) is not None:
        lang = pycountry.languages.get(alpha_2=lang).name.lower()
    else:
        lang = "english"
    stop_words = set(stopwords.words(lang))
    wordnet_lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [wordnet_lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

class AbstractModelLLM:
    def __init__(self, **kwargs) -> None:
        self.__name = "LLM"
        self._conversation = []
        self._stats = []
        self.system(kwargs.get("system_prompt", "You are an expert in the semantic web and have deep knowledge about writing schema.org markup."))

    def system(self, prompt):
        if len(self._conversation) > 0 and self._conversation[0]['role'] == 'system':
            self._conversation.pop(0)
        self._conversation.insert(0,{'role':'system','content': prompt})
        
    @backoff.on_exception(backoff.expo, (ServiceUnavailableError, Timeout, RateLimitError))
    def query(self, prompt, **kwargs):
        """Prompt the model and retrieve the answer. 
        The prompt will be concatenated to the chat logs before being sent to the model

        Args:
            prompt (_type_): _description_
        """
            
        model = "gpt-3.5-turbo-16k"
        prompt_str = "\n".join(prompt.values()) if isinstance(prompt, dict) else prompt
        prompt_tokens, estimated_completion_tokens = count_tokens(prompt_str, model)
        estimated_cost = estimate_cost(prompt_str, model)

        estimation = {
            "prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
            "estimated_cost": estimated_cost
        }
        
        self._stats.append(estimation)

    def get_stats_df(self):
        return pd.DataFrame.from_records(self._stats)
    
    def reset(self):
        self._conversation = []
        self._stats = []
        
    def get_name(self):
        return self.__name
    
    def chain_query(self, plan: OrderedDict, verbose=False):
        self.reset()
        chain = []
        c_plan = deepcopy(plan)
        thought = []
        while len(c_plan) > 0:
            k, v = c_plan.popitem(last=False)
            thought.append(v)
            if k.startswith("chain") or len(c_plan) == 0:
                chain.append("\n".join(thought))
                thought = []
            if k.startswith("cot"):
                chain[-1] = "\n".join([chain[-1], v])
                thought = []
                                        
        responses = []
        for i, thought in enumerate(chain): 
            prompt = str(thought).replace("[PREV_RES]", responses[-1]) if i > 0 else thought
            response = self.query(prompt)
            responses.append(response)
        
        return "\n".join(responses) if verbose else responses[-1]
    
    
    def map_reduce_predict(self, schema_types, content, **kwargs):

        chunk_tok_count_limit = 6000
        outfile = kwargs["outfile"]
        model = "gpt-3.5-turbo-16k"
        tok_count, _ = count_tokens(content, model)
        logger.info(f"There are {tok_count} tokens in the document!")

        if tok_count <= chunk_tok_count_limit:
            return self.predict(schema_types, content, verbose=True, **kwargs)

        chunks = chunk_document(content, chunk_tok_count_limit)
        markups = []
        for i in range(len(chunks)):
            chunk_outfile = f"{Path(outfile).parent}/{Path(outfile).stem}_chunk{i}.jsonld"
            if os.path.exists(chunk_outfile) and os.stat(chunk_outfile).st_size > 0:
                with open(chunk_outfile, "r") as f:
                    current_markup = json.load(f)
            else:
                current_markup = self.predict(schema_types, chunks[i], verbose=True, **kwargs)
                with open(chunk_outfile, "w") as f:
                    json.dump(current_markup, f)

            markups.append(current_markup)

        jsonld = {}
        for markup in markups:
            jsonld.update(markup)

        return jsonld
        

            
    def predict(self, schema_types, content, **kwargs) -> Dict:

        remember = kwargs.get("remember", False)
        in_context_learning =  kwargs.get("in_context_learning", False)
        chain_of_thought =  kwargs.get("chain_of_thought", False)
        expert =  kwargs.get("expert", False)
        subtarget_classes = kwargs.get("subtarget_classes") or []
        map_reduce_chunk = kwargs.get("map_reduce_chunk")

        def get_type_from_content(explain=False):
            # Get the correct schema-type
            prompt = textwrap.dedent(f"""
            -------------------
            {content}
            -------------------
            Give me the schema.org types that best describes the above content.
            The answer should be under json format.
            """)

            return prompt if explain else self.query(prompt).strip()
        
        def generate_jsonld(schema_type_urls, explain=False):

            prompt = OrderedDict({
                "context1": textwrap.dedent(f"""
                    - Given the content delimited with XML tags:
                    <content>
                    {content}
                    </content>
                    """
                )
            })
            # For each of the type, make a markup
            for i, schema_class in enumerate(set(schema_type_urls + subtarget_classes)):
                schema_attrs = get_type_definition(schema_class, simplify=True)
                prompt.update({
                    f"definition{i}": textwrap.dedent(f"""
                    - You can use the following properties for the schema.org type {schema_class} :
                        <definition> 
                        {schema_attrs}
                        </definition>
                    """)
                })
            # Task
            rules = [       
                f"\t- Only use properties if the information is mentioned implicitly or explicitly in the content.\n"
                f"\t- Fill properties with as much information as possible.\n"
                f"\t- In case there are many sub-entities described, when possible, the output must include them all.\n"
                f"\t- The output have to be encoded in JSON-LD format.\n"    
            ]
            
            if map_reduce_chunk is None:
                rules.insert(1, f"\t- The output must include 1 main entity of type {schema_type_urls}.\n")
                
            if len(subtarget_classes) > 0 and subtarget_classes != [schema_type_urls]:
                rules.insert(len(rules)-1, f"\t- The output must include at least 1 sub-entity of type(s) {subtarget_classes}.")
            
            rules = "\n".join(rules)

            self.system(f"""
                You are an expert in the semantic web and have deep knowledge about writing schema.org markup for type {schema_type_urls}.
                Given the following content, definition(s), please ouput only the JSON-LD markup from the content according to the definition which respect to the rules.
                - Rules: 
                {rules}     
                """)

            # if not expert: 
            #     prompt.pop("expert")
            
            if in_context_learning:
                self.system(f"""
                You are an expert in the semantic web and have deep knowledge about writing schema.org markup for type {schema_type_urls}.
                Given the following content, definition(s), example(s), please ouput only the JSON-LD markup from the content according to the definition which respect to the rules.
                - Rules: 
                {rules}     
                """)
                for schema_type_url in schema_type_urls:
                    examples = get_schema_example(schema_type_url)
                    for i, example in enumerate(examples):
                        prompt.update({
                            f"example{i}": textwrap.dedent(f"""
                            Example {i}:
                            ```json
                            {example}
                            ```
                            """)
                        })

            print(f"""
                You are an expert in the semantic web and have deep knowledge about writing schema.org markup for type {schema_type_urls}.
                Given the following content, definition(s), please ouput only the JSON-LD markup from the content according to the definition which respect to the rules.
                - Rules: 
                {rules}     
                """)
            raise RuntimeError()
            return prompt if explain else self.query(prompt, remember=remember)
        
        explain = kwargs.get("explain", False)
            
        if explain:
            prompt = generate_jsonld(schema_types, explain=True)
        
            model = "gpt-3.5-turbo-16k"
            prompt_tokens, estimated_completion_tokens = count_tokens(prompt, model)
            estimated_cost = estimate_cost(prompt, model)

            return {
                "prompt_tokens": prompt_tokens,
                "estimated_completion_tokens": estimated_completion_tokens,
                "estimated_cost": estimated_cost
            }

        self.reset()
        
        jsonld_string = generate_jsonld(schema_types)
        jsonld = extract_json(jsonld_string)
        
        if not isinstance(jsonld, dict):
            raise RuntimeError(f"Expecting dict, got {type(jsonld)}, content={jsonld}")
        return jsonld
    
    def _evaluate_coverage(self, pred, expected, **kwargs):   
        
        target_class = kwargs["target_class"]
                    
        pred_graph = ConjunctiveGraph()
        
        try: pred_graph.parse(pred)
        except UnboundLocalError:
            return {
                "class": target_class,
                "pred": None,
                "expected": None
            }
        
        expected_graph = ConjunctiveGraph()
        expected_graph.parse(expected)
        
        type_defs = set(get_type_definition(target_class, simplify=True))       
        pred_p = extract_preds(pred_graph, target_class, simplify=True) & type_defs
        expected_p = extract_preds(expected_graph, target_class, simplify=True) & type_defs
        
        pred_p_count = len(pred_p)
        expected_p_count = len(expected_p)
        
        logger.debug(f"Definition for {target_class}: {type_defs}")
        logger.debug(f"Predicates (predicted): {pred_p}")
        logger.debug(f"Predicates (expected): {expected_p}")
                
        class_count = len(type_defs)
        
        epsilon = 1e-5
        
        pred_coverage = pred_p_count / (class_count + epsilon)
        expected_coverage = expected_p_count / (class_count + epsilon)
        
        return { 
            "class": target_class,
            "pred": pred_coverage,
            "expected": expected_coverage
        }
    
    def _evaluate_shacl(self, pred, expected, **kwargs):
        """Validate the generated markup against SHACL validator

        Args:
            pred (_type_): _description_
            expected (_type_): _description_
        """
        validator = ValidatorFactory.create_validator("ShaclValidator", shape_graph="schemaorg/shacl/schemaorg_datashapes_closed.shacl")
        
        pred_outfile = f"{Path(pred).parent}/{Path(pred).stem}_shacl_pred.json"
        pred_score = validator.validate(pred, outfile=pred_outfile)
        
        expected_outfile = f"{Path(pred).parent}/{Path(expected).stem}_shacl_expected.json"
        expected_score = validator.validate(expected, outfile=expected_outfile)
        
        return { 
            "pred": pred_score,
            "expected": expected_score,
        }
        
    def _evaluate_factual_consistency(self, pred, expected = None, **kwargs):
        # validator = ValidatorFactory.create_validator("FactualConsistencyValidator", retriever="BM25RetrievalModel")
        validator = ValidatorFactory.create_validator("FactualConsistencyValidator", retriever=self)

        pred_basename = kwargs.get("basename", Path(pred).stem)
        pred_outfile = f"{Path(pred).parent}/{pred_basename}_factual_pred.json"
        pred_result = validator.map_reduce_validate(pred, outfile=pred_outfile, **kwargs)
        # pred_result = validator.validate(pred, outfile=pred_outfile, **kwargs)

        if expected is None:
            return {"pred":pred_result}
        else :
            expected_basename = kwargs.get("basename", Path(expected).stem)
            expected_outfile = f"{Path(pred).parent}/{expected_basename}_factual_expected.json"
            expected_result = validator.map_reduce_validate(expected, outfile=expected_outfile, **kwargs)
            # expected_result = validator.validate(expected, outfile=expected_outfile, **kwargs)

            return {
                "pred": pred_result,
                "expected": expected_result
            }
        
    def _evaluate_semantic_conformance(self, pred, expected=None, **kwargs):
        validator = ValidatorFactory.create_validator("SemanticConformanceValidator", retriever=self)
        
        pred_basename = kwargs.get("basename", Path(pred).stem)
        pred_outfile = f"{Path(pred).parent}/{pred_basename}_semantic_pred.json"
        # pred_result = validator.map_reduce_validate(pred, outfile=pred_outfile, **kwargs)
        pred_result = validator.validate(pred, outfile=pred_outfile, **kwargs)


        if expected is None:
            return {
                "pred": pred_result,
            }
        else:
            expected_basename = kwargs.get("basename", Path(expected).stem)
            expected_outfile = f"{Path(pred).parent}/{expected_basename}_semantic_expected.json"
            # expected_result = validator.map_reduce_validate(expected, outfile=expected_outfile)
            expected_result = validator.validate(expected, outfile=expected_outfile)
            
            return {
                "pred": pred_result,
                "expected": expected_result
            }
        
    def _evaluate_sameas(self, pred, expected, **kwargs):
        validator = ValidatorFactory.create_validator("SameAsValidator", retriever=self)
        sameas = validator.validate(pred, expected_file=expected)
        
        return {
            "sameas": sameas
        }
    
    def evaluate(self, method, pred, expected=None, **kwargs):        
        pred_verbalized_fn = None
        expected_verbalized_fn = None
        
        if method in ["text-emb", "ngrams"]:
            pred_verbalized_fn = os.path.join(Path(pred).parent, Path(pred).stem + "_pred.md")
            expected_verbalized_fn = os.path.join(Path(pred).parent, Path(pred).stem + "_expected.md")
            
            if not os.path.exists(pred_verbalized_fn) or os.stat(pred_verbalized_fn).st_size == 0:
                with open(pred_verbalized_fn, "w") as ofs:
                    filtered_graph = ConjunctiveGraph()
                    filtered_graph.parse(pred)
                    filtered_graph = filter_graph(filtered_graph, pred=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), obj=URIRef(lookup_schema_type(schema_type)))
                    verbalized = self.verbalize(filtered_graph.serialize())
                    ofs.write(verbalized)
            
            if not os.path.exists(expected_verbalized_fn) or os.stat(expected_verbalized_fn).st_size == 0:
                with open(expected_verbalized_fn, "w") as ofs:
                    filtered_graph = ConjunctiveGraph()
                    filtered_graph.parse(expected)
                    filtered_graph = filter_graph(filtered_graph, pred=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), obj=URIRef(lookup_schema_type(schema_type)))
                    verbalized = self.verbalize(filtered_graph.serialize())
                    ofs.write(verbalized)
        
        if method == "graph-emb":
            return self._evaluate_graph_emb(pred, expected, **kwargs)
        elif method == "text-emb":            
            return self._evaluate_text_emb(pred_verbalized_fn, expected_verbalized_fn, **kwargs)
        elif method == "ngrams":
            return self._evaluate_ngrams(pred_verbalized_fn, expected_verbalized_fn, **kwargs) 
        elif method == "shacl":
            return self._evaluate_shacl(pred, expected, **kwargs) 
        elif method == "coverage":
            return self._evaluate_coverage(pred, expected, **kwargs)
        elif method == "factual":
            return self._evaluate_factual_consistency(pred, expected, **kwargs)
        elif method == "semantic":
            return self._evaluate_semantic_conformance(pred, expected, **kwargs)
        elif method == "sameas":
            return self._evaluate_sameas(pred, expected, **kwargs)
        else:
            raise NotImplementedError(f"The evaluator for {method} is not yet implemented!")
        
    def verbalize(self, jsonld):
        self.reset()
        
        prompt = f"""
        Given the schema.org markup below:
        ------------------
        {jsonld}
        ------------------
        
        Generate the corresponding Markdown document.
        The output must only use all provided information.
        The output should contain the markdown code only.
        """
        
        result = self.query(prompt).strip()
        
        if "```" in result:
            if not result.endswith("```"):
                result += "```"
            result = re.search(r"```markdown([\w\W]*)```", result).group(1)
        return result
    
class LlamaCPP(AbstractModelLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_repo = kwargs["model_repo"]
        model_file = kwargs["model_file"]
        
        model_path = hf_hub_download(repo_id=model_repo, filename=model_file, cache_dir=".models")

        with open(LLAMA_CPP_CONFIG) as f:
            llama_configs = yaml.safe_load(f)
            self.__llm = Llama(model_path=model_path, **llama_configs)
        
    def query(self, prompt, **kwargs):
        
        remember = kwargs.pop("remember", False)
        explain = kwargs.pop("explain", False)
        kwargs["temperature"] = kwargs.get("temperature", 0.0)

        super().query(prompt)

        prompt = "\n".join(prompt.values()) if isinstance(prompt, dict) else prompt

        if explain:
            logger.info(self._stats[-1])
            return

        logger.debug(f">>>> Q: {prompt}")
                
        chatgpt_prompt = {"role": "user", "content": prompt}
        if remember:
            self._conversation.append(chatgpt_prompt)
            
        history = self._conversation if remember and len(self._conversation) > 1 else self._conversation + [chatgpt_prompt]
        
        reply = LLM_CACHE.get(prompt)
        if reply is None:                            
            chat = self.__llm.create_chat_completion(messages=history, **kwargs)
            reply = chat["choices"][0]["message"]["content"]
            logger.debug(f">>>> A: {reply}")
        else:
            logger.debug(f">>>> A (CACHED): {reply}")
        
        if remember:
            self._conversation.append({"role": "assistant", "content": reply})
        return reply


class Vicuna_7B(LlamaCPP):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_path="lmsys/vicuna-7b-v1.5-16k", **kwargs)

class Mistral_7B_Instruct(LlamaCPP):
    def __init__(self, **kwargs) -> None:
        quant_method = kwargs.pop("quant_method", "Q4_K_M")
        super().__init__(
            model_repo="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            model_file=f"mistral-7b-instruct-v0.2.{quant_method}.gguf",
            **kwargs
        )   
    
    def query(self, prompt, **kwargs):
        if isinstance(prompt, dict):
            for k, v in prompt.items():
                if k.startswith("task") and not v.startswith("[INST]"):
                    prompt[k] = f"[INST] {v} [/INST]"
        return super().query(prompt, **kwargs) 

class Mixtral_8x7B_Instruct(LlamaCPP):
    def __init__(self, **kwargs) -> None:
        quant_method = kwargs.pop("quant_method", "Q4_K_M")
        super().__init__(
            model_repo="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            model_file=f"mixtral-8x7b-instruct-v0.1.{quant_method}.gguf",
            **kwargs
        )   
    
    def query(self, prompt, **kwargs):
        if isinstance(prompt, dict):
            for k, v in prompt.items():
                if k.startswith("task") and not v.startswith("[INST]"):
                    prompt[k] = f"[INST] {v} [/INST]"
        return super().query(prompt, **kwargs) 
            
class GPT(AbstractModelLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__name = "GPT"
        self._model = kwargs.get("model") or "gpt-3.5-turbo-16k"
                    
        openai.api_key_path = ".openai/API.txt"
        Path(openai.api_key_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(openai.api_key_path):
            openai.api_key = input('YOUR_API_KEY: ')
            with open(openai.api_key_path, "w") as f:
                f.write(openai.api_key)
        else:
            with open(openai.api_key_path, "r") as f:
                openai.api_key = f.read()

    @backoff.on_exception(backoff.expo, (ServiceUnavailableError, Timeout, RateLimitError, APIError))
    def query(self, prompt, **kwargs):
        
        remember = kwargs.pop("remember", False)
        explain = kwargs.pop("explain", False)
        kwargs["temperature"] = kwargs.get("temperature", 0.0)

        super().query(prompt)

        if explain:
            logger.info(self._stats[-1])
            return
        
        prompt = "\n".join(prompt.values()) if isinstance(prompt, dict) else prompt

        logger.debug(f">>>> Q: {prompt}")
                
        chatgpt_prompt = {"role": "system", "content": prompt}

        if remember:
            self._conversation.append(chatgpt_prompt)
            
        history = self._conversation if remember and len(self._conversation) > 1 else self._conversation + [chatgpt_prompt]
        
        reply = LLM_CACHE.get(prompt)
        if reply is None:                            
            chat = openai.ChatCompletion.create(model=self._model, messages=history, **kwargs)
            reply = chat.choices[0].message.content
            logger.debug(f">>>> A: {reply}")
        else:
            logger.debug(f">>>> A (CACHED): {reply}")
        
        if remember:
            self._conversation.append({"role": "assistant", "content": reply})
        return reply

class ModelFactoryLLM:
    @staticmethod
    def create_model(model_class, **kwargs) -> AbstractModelLLM:
        return globals()[model_class](**kwargs)
    

    
    