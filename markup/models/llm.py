import json
import os
from pathlib import Path
import re
from typing import Dict
import numpy as np

import openai
from rdflib import ConjunctiveGraph, URIRef
import torch
from models.validator import ValidatorFactory
from utils import filter_graph_by_type, get_ref_attrs, lookup_schema_type

from huggingface_hub import login, whoami
from huggingface_hub.utils._headers import LocalTokenNotFoundError

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

from ftlangdetect import detect as lang_detect

import pycountry

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
    def __init__(self, target_type) -> None:
        self.__name = "LLM"
        self.__schema_type = target_type
        self._conversation = []
        
    def query(self, prompt):
        """Prompt the model and retrieve the answer. 
        The prompt will be concatenated to the chat logs before being sent to the model

        Args:
            prompt (_type_): _description_
        """
        raise NotImplementedError()
    
    def reset(self):
        self._conversation = []
        
    def get_name(self):
        return self.__name
            
    def predict(self, content) -> Dict:

        def get_type_from_content():
            # Get the correct schema-type
            prompt = f"""
            -------------------
            {content}
            -------------------
            Give me the schema.org types that best describes the above content.
            The answer should be under json format.
            """

            result = self.query(prompt)
            return result.strip()
        
        def generate_jsonld(schema_type_url, schema_attrs):
            # For each of the type, make a markup
            prompt = f"""
            Given the content below:
            -------------------
            {content}
            -------------------

            These are the attribute for Type {schema_type_url}:
            -------------------
            {schema_attrs}
            -------------------

            Give me the JSON-LD markup that matches the content.
            The type must be {schema_type_url} .
            Only fill attributes with the information provided in the content.
            Fill attributes with as much information as possible.
            In case there are many {self.__schema_type} described, the output must include them all.
            The output should only contain the JSON code.
            """

            return self.query(prompt)

        self.reset()
        #schema_type = get_type_from_content()
        schema_type_url = lookup_schema_type(self.__schema_type)
        schema_attrs = get_ref_attrs(schema_type_url)
        jsonld = generate_jsonld(schema_type_url, schema_attrs).strip()

        if "```" in jsonld:
            #jsonld = re.sub(r"(\}\s+)(```)?(\s*\w+)", r"\1```\3", jsonld)
            jsonld = re.search(r"```json([\w\W]*)```", jsonld).group(1)
        #try:    
        schema_markup = json.loads(jsonld)
        # except json.decoder.JSONDecodeError: 
        #     return jsonld
        return schema_markup
    
    def _evaluate_coverage(self, pred, expected):
        ref_type = lookup_schema_type(self.__schema_type)
        def extract_preds(graph: ConjunctiveGraph, root=None):
            results = set()
            if root is None:
                for s in graph.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef(ref_type)):
                    results.update(extract_preds(graph, root=s))
            else:
                for p, o in graph.predicate_objects(root):
                    results.add(p)
                    results.update(extract_preds(graph, root=o))
            return results
        
        pred_graph = ConjunctiveGraph()
        pred_graph.parse(pred)
        
        expected_graph = ConjunctiveGraph()
        expected_graph.parse(expected)
        
        pred_p = extract_preds(pred_graph)
        expected_p = extract_preds(expected_graph)
        
        pred_p_count = len(pred_p)
        expected_p_count = len(expected_p)
        
        print(pred_p)
        print(expected_p)
        
        class_count = len(get_ref_attrs(ref_type.strip("<>")))
        
        pred_coverage = pred_p_count / class_count
        expected_coverage = expected_p_count / class_count
        
        return { 
            "pred": pred_coverage,
            "expected": expected_coverage,
            "pred/expected": pred_coverage/expected_coverage
        }
    
    def _evaluate_graph_emb(self, pred, expected):
        """Calculate the semantic distance between two KGs, i.e, two markups

        Args:
            pred (_type_): _description_
            expected (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
                
        def createKG(path_to_graph):
            g = ConjunctiveGraph()
            parse_format = Path(path_to_graph).suffix.strip(".")
            parse_format = "json-ld" if parse_format == "json" else parse_format
            g.parse(path_to_graph, format=parse_format)
            
            print(g.serialize())
            
            graph = KG()
            entities = []
                        
            for s, p, o in g:
                subj = Vertex(s.n3())
                obj = Vertex(o.n3())
                predicate = Vertex(p.n3(), predicate=True, vprev=subj, vnext=obj)
                graph.add_walk(subj, predicate, obj)
                if subj.name not in entities:
                    entities.append(subj.name)
            
            return graph, entities
        
        def embed(g: ConjunctiveGraph, entities):
            # Create our transformer, setting the embedding & walking strategy.
            transformer = RDF2VecTransformer(
                Word2Vec(epochs=1000),
                walkers=[RandomWalker(4, 10, with_reverse=True, n_jobs=2)],
                # verbose=1
            )                                    
            return transformer.fit_transform(g, entities)
        
        predicted_graph, predicted_entities = None, None
        expected_graph, expected_entities = None, None
        try:
            predicted_graph, predicted_entities = createKG(pred) 
            expected_graph, expected_entities = createKG(expected)
        except:
            return {"cosine_sim": "error_invalid_kg"}
                    
        # Get our embeddings.
        predicted_embeddings, _ = embed(predicted_graph, predicted_entities)
        expected_embeddings, _ = embed(expected_graph, expected_entities)
            
        predicted_embeddings = np.mean(predicted_embeddings, axis=0)        
        expected_embeddings = np.mean(expected_embeddings, axis=0)
        cosine_distance = cosine(np.array(predicted_embeddings).flatten(), np.array(expected_embeddings).flatten())
        # print(f"Cosine: {1 - cosine_distance}")
        return { "cosine_sim": 1 - cosine_distance }
            
    def _evaluate_text_emb(self, pred, expected):
        """Mesure the semantic similarity between the prediction text, expected text and the web page.

        Args:
            pred (_type_): _description_
            expected (_type_): _description_
        """
                
        def evaluate(reference_text, hypothesis_text):    
            # Create TF-IDF vectorizer and transform the corpora
            # vectorizer = TfidfVectorizer()
            # tfidf_matrix = vectorizer.fit_transform([reference_text, hypothesis_text])
            
            # Load pre-trained model and tokenizer
            model_name = "bert-base-multilingual-cased"  # You can use a different model here
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            refs_tokens = tokenizer(reference_text, return_tensors='pt', padding=True, truncation=True)
            hyp_tokens = tokenizer(hypothesis_text, return_tensors='pt', padding=True, truncation=True)
            
            # Get the model embeddings for the input text
            with torch.no_grad():
                refs_embeddings = model(**refs_tokens).last_hidden_state.mean(dim=1).numpy()
                hyp_embeddings = model(**hyp_tokens).last_hidden_state.mean(dim=1).numpy()
                                
                # Calculate cosine similarity
                cosine_sim = cosine_similarity(refs_embeddings, hyp_embeddings)
                # print(f"Cosine: {cosine_sim.item()}")
                return { "cosine_sim": cosine_sim.item() }
            
        
        reference_text = open(f"{Path(expected).parent.parent}/{Path(expected).stem.split('_')[0]}.txt", "r").read()
        predicted_text = open(pred, "r").read()
        baseline_text = open(expected, "r").read()
        
        # print("==== WEBPAGE - PREDICTION ====")
        # evaluate(reference_text, predicted_text)
        
        # print("==== WEBPAGE - BASELINE ====")
        # evaluate(reference_text, baseline_text)
        
        # print("==== PREDICTION - BASELINE ====")
        # evaluate(predicted_text, baseline_text)
                
        return { 
            "webpage-pred": evaluate(reference_text, predicted_text),
            "webpage-baseline": evaluate(reference_text, baseline_text),
            "pred-baseline": evaluate(predicted_text, baseline_text)
        }
    
    def _evaluate_ngrams(self, pred, expected):
        """Compare the verbalization of predicted KG, i.e, the generated markup and the input text.

        Args:
            pred (_type_): _description_
            expected (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
                
        def evaluate(reference_text, hypothesis_text):
      
            stop_words = set(stopwords.words('english'))
            # Tokenize the sentences
            ref_tokens = preprocess_text(reference_text).split()
            hyp_tokens = preprocess_text(hypothesis_text).split()
    
            
            # BLEU Score
            bleu_score = bleu(ref_tokens, hyp_tokens, smoothing_function=SmoothingFunction().method3)

            # NIST Score
            nist_score = nist(ref_tokens, hyp_tokens)

            # ROUGE Score
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_text, hypothesis_text)
            rouge_1_score = scores['rouge1'].fmeasure
            rouge_2_score = scores['rouge2'].fmeasure
            rouge_L_score = scores['rougeL'].fmeasure
            
            # GLEU
            gleu_score = gleu(ref_tokens, hyp_tokens)
            
            # CHRF
            chrf_score = chrf(ref_tokens, hyp_tokens)
            
            # METEOR
            # meteor_score = meteor(ref_tokens, hyp_tokens)

            # Print the scores
            float_prec = 4
            # print(f"BLEU: {round(bleu_score, float_prec)}")
            # print(f"NIST Score: {round(nist_score, float_prec)}")
            # print(f"ROUGE-1: {round(rouge_1_score, float_prec)}")
            # print(f"ROUGE-2: {round(rouge_2_score, float_prec)}")
            # print(f"ROUGE-L: {round(rouge_L_score, float_prec)}")
            # print(f"GLEU: {round(gleu_score, float_prec)}")
            # print(f"CHLF: {round(chrf_score, float_prec)}")
            # print(f"METEOR: {round(meteor_score, float_prec)}")
            
            return {
                "BLEU": bleu_score, 
                "ROUGE-1": rouge_1_score, 
                "ROUGE-2": rouge_2_score, 
                "ROUGE-L": rouge_L_score, 
                "NIST": nist_score,
                "GLEU": gleu_score,
                "CHLF": chrf_score
            }
        
        reference_text = open(f"{Path(expected).parent.parent}/{Path(expected).stem.split('_')[0]}.txt", "r").read().lower()
        predicted_text = open(pred, "r").read().lower()
        baseline_text = open(expected, "r").read().lower()
        
        # print("==== WEBPAGE - PREDICTION ====")
        # evaluate(reference_text, predicted_text)
        
        # print("==== WEBPAGE - BASELINE ====")
        # evaluate(reference_text, baseline_text)
        
        # print("==== PREDICTION - BASELINE ====")
        # evaluate(predicted_text, baseline_text)
        
        return { 
            "webpage-pred": evaluate(reference_text, predicted_text),
            "webpage-baseline": evaluate(reference_text, baseline_text),
            "pred-baseline": evaluate(predicted_text, baseline_text)
        }
    
    def _evaluate_shacl(self, pred):
        """Validate the generated markup against SHACL validator

        Args:
            pred (_type_): _description_
            expected (_type_): _description_
        """
        try:
            validator = ValidatorFactory.create_validator("SchemaOrgShaclValidator")
            conforms, shacl_report = validator.validate(pred)
            return { "shacl_conform": conforms }
        except:
            return { "shacl_conform": "error_invalid_kg" }
    
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
                    filtered_graph = filter_graph_by_type(filtered_graph, self.__schema_type)
                    verbalized = self.verbalize(filtered_graph.serialize())
                    ofs.write(verbalized)
            
            if not os.path.exists(expected_verbalized_fn) or os.stat(expected_verbalized_fn).st_size == 0:
                with open(expected_verbalized_fn, "w") as ofs:
                    filtered_graph = ConjunctiveGraph()
                    filtered_graph.parse(expected)
                    filtered_graph = filter_graph_by_type(filtered_graph, self.__schema_type)
                    verbalized = self.verbalize(filtered_graph.serialize())
                    ofs.write(verbalized)
        
        if method == "graph-emb":
            return self._evaluate_graph_emb(pred, expected)
        elif method == "text-emb":            
            return self._evaluate_text_emb(pred_verbalized_fn, expected_verbalized_fn)
        elif method == "ngrams":
            return self._evaluate_ngrams(pred_verbalized_fn, expected_verbalized_fn) 
        elif method == "shacl":
            return self._evaluate_shacl(pred) 
        elif method == "coverage":
            ref_type = kwargs.get("ref_type")
            return self._evaluate_coverage(pred, expected)
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
        The output must use all provided information.
        The output must not contain information that is not provided.
        The output should contain the markdown code only.
        """
        
        result = self.query(prompt).strip()
        
        if "```" in result:
            if not result.endswith("```"):
                result += "```"
            result = re.search(r"```markdown([\w\W]*)```", result).group(1)
        return result
    

class HuggingFaceLLM(AbstractModelLLM):

    def __init__(self, model, **kwargs) -> None:
        super().__init__()  
        self.__name = "HuggingFace"
        self._conversation = []

        try: whoami()
        except LocalTokenNotFoundError: login()
        
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)

        #self._max_length = 30 if kwargs.get("max_length") is None else kwargs.get("max_length")
        
    def query(self, prompt):
        # TODO: concat to chat history
        print(f">>>> Q: {prompt}")
        self._conversation.append(prompt)

        prompt_xtd = "\n".join(self._conversation)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        generate_ids = self._model.generate(inputs.input_ids)
        reply = self._tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f">>>> A: {reply}")
        #self._conversation.append(reply)
        return reply
    
class Llama2_70B(HuggingFaceLLM):
    def __init__(self, **kwargs):
        super().__init__("meta-llama/Llama-2-70b-chat-hf", **kwargs)
        self.__name = "Llama2_70B"
               
class Llama2_7B(HuggingFaceLLM):
    def __init__(self, **kwargs):
        super().__init__("meta-llama/Llama-2-7b-chat-hf", **kwargs)
        self.__name = "Llama2_7B"
        
class Llama2_13B(HuggingFaceLLM):
    def __init__(self, **kwargs):
        super().__init__("meta-llama/Llama-2-13b-chat-hf", **kwargs)
        self.__name = "Llama2_13B"

class Vicuna_7B(HuggingFaceLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__("lmsys/vicuna-7b-v1.5-16k", **kwargs)

class Vicuna_13B(HuggingFaceLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__("lmsys/vicuna-13b-v1.5-16k", **kwargs)

class Mistral_7B_Instruct(HuggingFaceLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1", **kwargs)
        self._device = "cpu"
    
    def query(self, prompt):
        # TODO: concat to chat history
        print(f">>>> Q: {prompt}")
        encodeds = self._tokenizer.apply_chat_template(self._conversation, return_tensors="pt")
        model_inputs = encodeds.to(self._device)
        self._model.to(self._device)

        generated_ids = self._model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        reply = self._tokenizer.batch_decode(generated_ids)[0]
        print(f">>>> A: {reply}")
        return reply
        
class ChatGPT(AbstractModelLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__name = "ChatGPT"
        self._model = "gpt-3.5-turbo-16k"
                    
        openai.api_key_path = ".openai/API.txt"
        Path(openai.api_key_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(openai.api_key_path):
            openai.api_key = input('YOUR_API_KEY: ')
            with open(openai.api_key_path, "w") as f:
                f.write(openai.api_key)
        else:
            with open(openai.api_key_path, "r") as f:
                openai.api_key = f.read()
                
                
    def query(self, prompt):
        print(f">>>> Q: {prompt}")
        self._conversation.append({"role": "system", "content": prompt})
        chat = openai.ChatCompletion.create( model=self._model, messages=self._conversation)
        reply = chat.choices[0].message.content
        print(f">>>> A: {reply}")
        self._conversation.append({"role": "assistant", "content": reply})
        return reply

class HuggingChatLLM(AbstractModelLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__()
                
        
        cookie_path_dir = "./.cookies_snapshot"
        sign: Login = None
        cookies = None
        
        if not os.path.exists(cookie_path_dir):
            # Log in to huggingface and grant authorization to huggingchat
            email = input("Enter your username: ")
            passwd = input("Enter your password: ")
            sign = Login(email, passwd)
            cookies = sign.login()

            # Save cookies to the local directory
            sign.saveCookiesToDir(cookie_path_dir)
        else:
            # Load cookies when you restart your program:
            email = Path(os.listdir(cookie_path_dir)[0]).stem
            print(f"Logging in as {email}")
            sign = Login(email, None)
            cookies = sign.loadCookiesFromDir(cookie_path_dir) # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.

        # Create a ChatBot
        self._model = hugchat.ChatBot(cookies=cookies.get_dict()) 
        
        available_models = [m.name for m in self._model.get_available_llm_models()]

        model = kwargs.get("hf_model")
        if model not in available_models:
            raise ValueError(f"{model} is not one of {available_models}!")
        
        model_id = available_models.index(model)
        self._model.switch_llm(model_id)

        
    def reset(self):
        # Create a new conversation
        self._model.delete_all_conversations()
        id = self._model.new_conversation()
        self._model.change_conversation(id)
    
    def query(self, prompt):
        print(f">>>> Q: {prompt}")
        # The message history is handled by huggingchat
        reply = self._model.query(prompt)["text"]
        print(f">>>> A: {reply}")
        return reply

class ModelFactoryLLM:
    @staticmethod
    def create_model(model_class, **kwargs) -> AbstractModelLLM:
        return globals()[model_class](**kwargs)