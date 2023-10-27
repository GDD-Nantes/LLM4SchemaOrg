import json
import re
from typing import Dict

import openai
from utils import get_ref_attrs, lookup_schema_type

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

class AbstractModelLLM:
    def __init__(self) -> None:
        self.__name = "LLM"
        self.__message = []
        
    def query(self, prompt):
        """Prompt the model and retrieve the answer. 
        The prompt will be concatenated to the chat logs before being sent to the model

        Args:
            prompt (_type_): _description_
        """
        pass
    
    def reset(self):
        self.__message = []
    
    def predict(self, content) -> Dict:
        
        self.reset()
        
        # Get the correct schema-type
        prompt = f"""
        -------------------
        {content}
        -------------------
        Give me the schema.org Types that best describes the above content.
        Answer in 1 word.
        """
        print(f">>>> Q: {prompt}")

        result = self.query(prompt)
        print(f">>>> A: {result}")

        schema_type = result.strip()
        schema_type_url = lookup_schema_type(schema_type)

        schema_attrs = get_ref_attrs(schema_type_url)


        # For each of the type, make a markup
        prompt = f"""
        These are the attribute for Type {schema_type_url}
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
        print(f">>>> Q: {prompt}")

        result = self.query(prompt)
        print(f">>>> A: {result}")
        if "```" in result:
            result = re.search(r"```json([\w\W]*)```", result).group(1)
        schema_markup = json.loads(result)
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
        
    def __init__(self, model):
        self.__name = "HuggingFace"
        login()
        
        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__model = AutoModelForCausalLM.from_pretrained(model)
        
    def query(self, prompt):
        # TODO: concat to chat history
        inputs = self.__tokenizer(prompt, return_tensors="pt")
        generate_ids = self.__model.generate(inputs.input_ids, max_length=30)
        return self.__tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
class LLaMA2_70B(HuggingFace_LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-70b-chat-hf")
               
class Llama2_7B(HuggingFace_LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-7b-chat-hf")
        
class Llama2_13B(HuggingFace_LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-13b-chat-hf")
        
class ChatGPT(AbstractModelLLM):
    def __init__(self, model) -> None:
        self.__model = model # gpt-3.5-turbo-16k
        self.__messages = []
        openai.api_key = input('YOUR_API_KEY')
                
    def query(self, prompt):
        self.__messages.append({"role": "system", "content": prompt})

        chat = openai.ChatCompletion.create( model=self.__model, messages=self.__messages)

        reply = chat.choices[0].message.content
        self.__messages.append({"role": "assistant", "content": reply})
        return reply