# MarkupAutomator

# Reproduce main experiment in the paper

- Generate prompts using LLM
```bash
snakemake -p -s snakefiles/generate.smk -c1 --rerun-incomplete --config data_dir=data/WDC/Pset prompt_template=text2kg_prompt3
```

- Evaluate against human prompt
```bash
snakemake -p -s snakefiles/evaluate.smk -c1 --rerun-incomplete --config data_dir=data/WDC/Pset prompt_template=text2kg_prompt3
```

- Assemble to a final csv
```bash
snakemake -p -s snakefiles/evaluate.smk -c1 --rerun-incomplete --config data_dir=data/WDC/Pset prompt_template=text2kg_prompt3
```

- Use the notebooks to obtain elements used in the paper.

# Bring your own LLM

- In `markup/models/llm.py`, add a new class that inherits:
    - `LlamaCPP`: any quantized model that are available on HuggingFace.
    - `GPT`: any model deployed on OpenAI's API v1 servers. 
    - `AbstractModelLLM`: others.

- Rewrite the method `predict` and `query` in your child class if necessary.

# Validate Factual & Conformance using Schema.org examples
- Run the following commands:

```bash
python markup/schemaorg_examples_dataset.py evaluate-prop-checker-zs Mixtral_8x7B_Instruct schemaorg/examples/semantic.parquet data/WDC/SchemaOrg/semantic_zs_mixtral --template=prompts/validation/semantic.json

python markup/schemaorg_examples_dataset.py evaluate-halu-checker-zs Mixtral_8x7B_Instruct schemaorg/examples/factual.parquet data/WDC/SchemaOrg/factual_zs_mixtral --template=prompts/validation/factual_p.json
```