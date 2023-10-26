import json
import re
from typing import Dict
from utils import get_ref_attrs, lookup_schema_type

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

class AbstractModel:
    def __init__(self) -> None:
        self.__name = "LLM"
        
    def query(self, prompt):
        pass
    
    def predict(self, content) -> Dict:
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
    


class HuggingFace_LLM(AbstractModel):
    def __init__(self, model):
        self.__name = "HuggingFace"
        login()
        
        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__model = AutoModelForCausalLM.from_pretrained(model)
        
    def query(self, prompt):
        inputs = self.__tokenizer(prompt, return_tensors="pt")
        generate_ids = self.__model.generate(inputs.input_ids, max_length=30)
        return self.__tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    

class LLaMA_70B_LLM(HuggingFace_LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-70b-chat-hf")