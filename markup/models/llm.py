import json
from pathlib import Path
import re
from typing import Dict
import numpy as np

import openai
from rdflib import Graph
import torch
from models.validator import ValidatorFactory
from utils import get_ref_attrs, lookup_schema_type

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

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    wordnet_lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [wordnet_lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

class AbstractModelLLM:
    def __init__(self) -> None:
        self.__name = "LLM"
        self.__conversation = []
        
    def query(self, prompt):
        """Prompt the model and retrieve the answer. 
        The prompt will be concatenated to the chat logs before being sent to the model

        Args:
            prompt (_type_): _description_
        """
        pass
    
    def reset(self):
        self.__conversation = []
    
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
            The output must be generated in JSON format.
            In case there are many {schema_type} described, the output must include them all.
            """

            return self.query(prompt)

        self.reset()
        #schema_type = get_type_from_content()
        schema_type = "Recipe"
        schema_type_url = lookup_schema_type(schema_type)
        schema_attrs = get_ref_attrs(schema_type_url)
        jsonld = generate_jsonld(schema_type_url, schema_attrs)    

        if "```" in jsonld:
            jsonld = re.search(r"```json([\w\W]*)```", jsonld).group(1)
        schema_markup = json.loads(jsonld)
        return schema_markup
    
    def _evaluate_graph_emb(self, pred, expected):
        """Calculate the semantic distance between two KGs, i.e, two markups

        Args:
            pred (_type_): _description_
            expected (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
                
        def createKG(path_to_graph):
            g = Graph()
            parse_format = Path(path_to_graph).suffix.strip(".")
            parse_format = "json-ld" if parse_format == "json" else parse_format
            g.parse(path_to_graph, format=parse_format)
            
            graph = KG()
            entities = set()
                        
            for s, p, o in g:
                subj = Vertex(s.n3())
                obj = Vertex(o.n3())
                predicate = Vertex(p.n3(), predicate=True, vprev=subj, vnext=obj)
                graph.add_walk(subj, predicate, obj)
                entities.add(s.n3())
                
            return graph, list(entities)
        
        def embed(g: Graph, entities):
            # Create our transformer, setting the embedding & walking strategy.
            transformer = RDF2VecTransformer(
                Word2Vec(epochs=1000),
                walkers=[RandomWalker(4, 10, with_reverse=True, n_jobs=2)],
                # verbose=1
            )
                        
            return transformer.fit_transform(g, entities)
                
        predicted_graph, predicted_entities = createKG(pred)
        expected_graph, expected_entities = createKG(expected)
                
        # Get our embeddings.
        predicted_embeddings, _ = embed(predicted_graph, predicted_entities)
        expected_embeddings, _ = embed(expected_graph, expected_entities)
        
        expected_embeddings = np.mean(expected_embeddings, axis=0)
        cosine_distance = cosine(np.array(predicted_embeddings).flatten(), np.array(expected_embeddings).flatten())
        print(1 - cosine_distance)
            
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
                print(f"Cosine: {cosine_sim.item()}")
            
        
        reference_text = open(f"{Path(expected).parent.parent}/{Path(expected).stem}.txt", "r").read()
        predicted_text = open(pred, "r").read()
        baseline_text = open(expected, "r").read()
        
        print("==== WEBPAGE - PREDICTION ====")
        evaluate(reference_text, predicted_text)
        
        print("==== WEBPAGE - BASELINE ====")
        evaluate(reference_text, baseline_text)
        
        print("==== PREDICTION - BASELINE ====")
        evaluate(predicted_text, baseline_text)
    
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
            print(f"BLEU: {round(bleu_score, float_prec)}")
            print(f"NIST Score: {round(nist_score, float_prec)}")
            print(f"ROUGE-1: {round(rouge_1_score, float_prec)}")
            print(f"ROUGE-2: {round(rouge_2_score, float_prec)}")
            print(f"ROUGE-L: {round(rouge_L_score, float_prec)}")
            print(f"GLEU: {round(gleu_score, float_prec)}")
            print(f"CHLF: {round(chrf_score, float_prec)}")
            # print(f"METEOR: {round(meteor_score, float_prec)}")
            
            return bleu_score, rouge_1_score, rouge_2_score, rouge_L_score, nist_score
        
        reference_text = open(f"{Path(expected).parent.parent}/{Path(expected).stem}.txt", "r").read().lower()
        predicted_text = open(pred, "r").read().lower()
        baseline_text = open(expected, "r").read().lower()
        
        print("==== WEBPAGE - PREDICTION ====")
        evaluate(reference_text, predicted_text)
        
        print("==== WEBPAGE - BASELINE ====")
        evaluate(reference_text, baseline_text)
        
        print("==== PREDICTION - BASELINE ====")
        evaluate(predicted_text, baseline_text)
    
    def _evaluate_shacl(self, pred):
        """Validate the generated markup against SHACL validator

        Args:
            pred (_type_): _description_
            expected (_type_): _description_
        """
        validator = ValidatorFactory.create_validator("SchemaOrgShaclValidator")
        shacl_report: Graph = validator.validate(pred)
    
    def evaluate(self, method, pred, expected):
        if method == "graph-emb":
            return self._evaluate_graph_emb(pred, expected)
        elif method == "text-emb":
            return self._evaluate_text_emb(pred, expected)
        elif method == "ngrams":
            return self._evaluate_ngrams(pred, expected) 
        elif method == "shacl":
            return self._evaluate_shacl(pred) 
        else:
            raise NotImplementedError(f"The evaluator for {method} is not yet implemented!")
        
    def verbalize(self, jsonld):
        prompt = f"""
        Given the schema.org markup below:
        ------------------
        {jsonld}
        ------------------
        
        Generate the corresponding Markdown document.
        The output must use all provided information.
        """
        
        result = self.query(prompt)
        return result
    

class HuggingFace_LLM(AbstractModelLLM):

    def __init__(self, model, **kwargs) -> None:
        super().__init__()  
        self.__name = "HuggingFace"
        self.__conversation = []

        try: whoami()
        except LocalTokenNotFoundError: login()
        
        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__model = AutoModelForCausalLM.from_pretrained(model)

        #self._max_length = 30 if kwargs.get("max_length") is None else kwargs.get("max_length")
        
    def query(self, prompt):
        # TODO: concat to chat history
        print(f">>>> Q: {prompt}")
        self.__conversation.append(prompt)

        prompt_xtd = "\n".join(self.__conversation)
        inputs = self.__tokenizer(prompt, return_tensors="pt")
        generate_ids = self.__model.generate(inputs.input_ids)
        reply = self.__tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f">>>> A: {reply}")
        #self.__conversation.append(reply)
        return reply
    
class Llama2_70B(HuggingFace_LLM):
    def __init__(self, **kwargs):
        super().__init__("meta-llama/Llama-2-70b-chat-hf", **kwargs)
        self.__name = "Llama2_70B"
               
class Llama2_7B(HuggingFace_LLM):
    def __init__(self, **kwargs):
        super().__init__("meta-llama/Llama-2-7b-chat-hf", **kwargs)
        self.__name = "Llama2_7B"
        
class Llama2_13B(HuggingFace_LLM):
    def __init__(self, **kwargs):
        super().__init__("meta-llama/Llama-2-13b-chat-hf", **kwargs)
        self.__name = "Llama2_13B"

class Vicuna_7B(HuggingFace_LLM):
    def __init__(self, **kwargs) -> None:
        super().__init__("lmsys/vicuna-7b-v1.5-16k", **kwargs)

class Vicuna_13B(HuggingFace_LLM):
    def __init__(self, **kwargs) -> None:
        super().__init__("lmsys/vicuna-13b-v1.5-16k", **kwargs)
        
class ChatGPT(AbstractModelLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__name = "ChatGPT"
        self.__model = "gpt-3.5-turbo-16k"
        self.__messages = []
        openai.api_key = input('YOUR_API_KEY')
                
    def query(self, prompt):
        print(f">>>> Q: {prompt}")
        self.__messages.append({"role": "system", "content": prompt})
        chat = openai.ChatCompletion.create( model=self.__model, messages=self.__messages)
        reply = chat.choices[0].message.content
        print(f">>>> A: {reply}")
        self.__messages.append({"role": "assistant", "content": reply})
        return reply

class ModelFactoryLLM:
    @staticmethod
    def create_model(model_class, **kwargs) -> AbstractModelLLM:
        return globals()[model_class](**kwargs)