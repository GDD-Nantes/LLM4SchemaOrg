# MarkupAutomator
This repository contains the code and results for the paper [TBM]().

# Structure

- Every inputs/outputs for the XP with 180 webpages are located under `data/WDC/Pset`.
- Every inputs/outputs for the XP with Schema.org examples are located under `data/WDC/SchemaOrg`.

# Pre-requisites:

- Python 3.10+ is required.

- Setup [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/stable/server/) (steps may vary based on your hardware):
```bash
# Install with server module on CUDA capable machine
CMAKE_ARGS="-DLLAMA_CUDA=on -DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -U "llama-cpp-python[server]" --force-reinstall --no-deps --no-cache-dir

# After modifying the config files, start the server
python -m llama_cpp.server --config configs/llama_cpp.json
```

- Install Spacy:
```bash
pip install spacy
python -m spacy download en_core_web_md
```

- Install the remaining dependencies:
```bash
pip install -r requirements.txt
```

# Reproduce main experiment in the paper

## Steps
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

## Hardware information

```
System:
    Distro: Ubuntu 22.04.3 LTS (Jammy Jellyfish)
Memory:
  RAM: total: 33.25 GiB used: 5.43 GiB (16.3%)
CPU:
  Info: 18-core model: Intel Xeon Platinum 8260 bits: 64 
  Speed (MHz): avg: 2394 
  Flags: avx avx2 ht lm nx pae sse sse2 sse3 sse4_1 sse4_2 ssse3 vmx
Graphics:
  Device-2: NVIDIA TU104GL [Tesla T4] driver: nvidia v: 535.161.07
  VRAM: 15360MiB
```

# Validate LLMs capability as Conformance and Factuality checker

## Steps

- Conformance:
```bash
python markup/schemaorg_examples_dataset.py evaluate-prop-checker-zs Mixtral_8x7B_Instruct schemaorg/examples/semantic.parquet .tmp/prop_checks_zs_mixtral_p --template=prompts/validation/semantic.json
```

- Factual:
```bash
python markup/schemaorg_examples_dataset.py evaluate-halu-checker-zs Mixtral_8x7B_Instruct schemaorg/examples/factual-simple.parquet data/WDC/SchemaOrg/halu_checks_zs_simple_mixtral_p --template=prompts/validation/factua_p_.json
```

# Bring your own LLM

- In `markup/models/llm.py`, add a new class that inherits:
    - `LlamaCPP`: any quantized model that are available on HuggingFace.
    - `GPT`: any model deployed on OpenAI's API v1 servers. 
    - `AbstractModelLLM`: others.

- Rewrite the method `predict` and `query` in your child class if necessary.

# Bring your own validator

- In `markup/models/validator.py`, add a new class that `AbstractValidator`:
  - `validate` method: returns a score
  - `map_reduce_validate` method: if your validator requires long text as input, e.g, the entire webpage, this method should split the input into chunks, apply `validate` on each chunk, then aggregate the results.

