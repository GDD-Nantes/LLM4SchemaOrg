import numpy as np
import colbert

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

import spacy
nlp = spacy.load("en_core_web_sm")

from rank_bm25 import BM25Plus

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import string

nltk.download('punkt')
nltk.download('stopwords')

from transformers import AutoTokenizer, AutoModel
import torch

def text_preprocessing(text):
    # Tokenization
    words = word_tokenize(text)

    # Convert to lowercase
    words = [word.lower() for word in words]

    # Remove punctuation
    # words = [word for word in words if word.isalnum()]

    # Remove stop words
    # stop_words = set(stopwords.words('english'))
    # words = [word for word in words if word not in stop_words]

    # Stemming
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]

    return words


class AbstractRetrievalModel:
    def __init__(self) -> None:
        self._queries = []
        self._documents = []
    
    def add_query(self, q):
        self._queries.append(q)
        
    def add_document(self, d):
        self._documents.append(d)
    
    def query(self, query, document, topk=1):
        pass

class BM25RetrievalModel(AbstractRetrievalModel):
    def __init__(self) -> None:
        super().__init__()
        
    def query(self, query, document, topk=1):
        sentences = [ sent.text for sent in nlp(document).sents ]
        sentences_toks = [ text_preprocessing(sent) for sent in sentences]
        self.__engine = BM25Plus(sentences_toks)
        tokenized_query = text_preprocessing(query)
        topk_sentences = self.__engine.get_top_n(tokenized_query, sentences, topk)
        print(topk_sentences)
        return topk_sentences

class BertRetrievalModel(AbstractRetrievalModel):
    """Retrieve sentence that correspond to the input query based on their cosine_similarity 
    in the embedding space.
    """
    
    def __init__(self) -> None:
        super().__init__()
        # Load pre-trained model and tokenizer
        model_name = "bert-base-multilingual-cased"  # You can use a different model here
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModel.from_pretrained(model_name)
        
    def query(self, query, document, topk=1):
        def __embed(text):
            tokens = self.__tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                return self.__model(**tokens).last_hidden_state.mean(dim=1)
        
        doc = nlp(document.strip())
        sentences = [ sent.text for sent in doc.sents ]
                
        sent_embs = [ __embed(sent) for sent in sentences ]
        query_emb = __embed(query)
        
        scores = [ torch.cosine_similarity(query_emb, sent_emb).item() for sent_emb in sent_embs ]
        scores = list(enumerate(zip(sentences, scores)))
        topk_sentences = sorted(scores, key=lambda x: x[1][1], reverse=True)[:topk]
        print(topk_sentences)
        return topk_sentences
        
class ColBertRetrievalModel(AbstractRetrievalModel):
    
    def __init__(self) -> None:
        super().__init__()
        self._nbits = 2   # encode each dimension with 2 bits
        self._doc_maxlen = 300 # truncate passages at 300 tokens
        self._max_id = 0 # 10000
                
        self._indexer = None
        self._searcher = None
          
    def query(self, query, document, topk=1):

        # Break down the document into sentences
        doc = nlp(document.strip())
        collection = [sent.text for sent in doc.sents]
        
        # Index the sentences
        checkpoint = 'colbert-ir/colbertv2.0'
        with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use
            config = ColBERTConfig(doc_maxlen=self._doc_maxlen, nbits=self._nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
            self._indexer = Indexer(checkpoint=checkpoint, config=config)
            self._indexer.index(name=self._index_name, collection=collection, overwrite=True)
        
        # Find the sentences that best answer queries  
        with Run().context(RunConfig(experiment='notebook')):
            self._searcher = Searcher(index=self._index_name, collection=collection)
            results = self._searcher.search(query, k=topk)
            
            # Print out the top-k retrieved passages
            for passage_id, passage_rank, passage_score in zip(*results):
                print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {self._searcher.collection[passage_id]}") 
