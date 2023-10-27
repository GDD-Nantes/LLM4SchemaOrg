import json
import re
from typing import Dict

import openai
from utils import get_ref_attrs, lookup_schema_type

from huggingface_hub import login, whoami
from huggingface_hub.utils._headers import LocalTokenNotFoundError

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, LlamaForCausalLM

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
            Give me the schema.org Types that best describes the above content.
            Answer in 1 word.
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
    
    def _evaluate_emb(self, pred, expected):
        """Calculate the semantic distance between two KGs, i.e, two markups

        Args:
            pred (_type_): _description_
            expected (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Method not yet implemented!")
    
    def _evaluate_ngrams(self, pred, expected):
        """Compare the verbalization of predicted KG, i.e, the generated markup and the input text.

        Args:
            pred (_type_): _description_
            expected (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Method not yet implemented!")
    
    def evaluate(self, method, pred, expected):
        if method == "emb":
            return self._evaluate_emb(pred, expected)
        elif method == "ngrams":
            return self._evaluate_ngrams(pred, expected) 
    

class HuggingFace_LLM(AbstractModelLLM):

    def __init__(self, model, **kwargs) -> None:
        super().__init__()  
        self.__name = "HuggingFace"
        self.__conversation = []

        try: whoami()
        except LocalTokenNotFoundError: login()
        
        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__model = LlamaForCausalLM.from_pretrained(model)

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