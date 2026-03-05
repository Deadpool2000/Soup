# Soup CLI — Quick Local Test Guide (Windows)

**Hardware:** RTX 3050 (4GB VRAM), i5
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B params, ~600MB in 4-bit)
**OS:** Windows (CMD/PowerShell)

---

## 0. Install

```cmd
pip install soup-cli
pip install datasketch
```

> Unit tests (`pytest tests/`) only work from the repo clone (`pip install -e ".[dev]"`), not from pip install.

## 1. Version & Help

```cmd
soup version
soup --help
soup train --help
soup data --help
```

## 2. Init Config from Templates

```cmd
soup init -t chat -o test_chat.yaml
soup init -t code -o test_code.yaml
```

> `soup init` without `-t` opens interactive wizard (requires terminal input).

## 3. Create Test Dataset

Create file `test_data.jsonl` with this content (copy-paste into any text editor, save as `test_data.jsonl`):

```jsonl
{"instruction": "What is Python?", "input": "", "output": "Python is a high-level programming language known for its simplicity."}
{"instruction": "Explain recursion", "input": "", "output": "Recursion is when a function calls itself to solve smaller subproblems."}
{"instruction": "What is a list?", "input": "", "output": "A list is an ordered, mutable collection of elements in Python."}
{"instruction": "What is a dictionary?", "input": "", "output": "A dictionary is a key-value data structure in Python."}
{"instruction": "What is OOP?", "input": "", "output": "OOP is a programming paradigm based on objects and classes."}
{"instruction": "What is an API?", "input": "", "output": "An API is an interface that allows software systems to communicate."}
{"instruction": "What is Git?", "input": "", "output": "Git is a distributed version control system for tracking code changes."}
{"instruction": "What is Docker?", "input": "", "output": "Docker is a platform for containerizing applications."}
{"instruction": "What is SQL?", "input": "", "output": "SQL is a language for managing and querying relational databases."}
{"instruction": "What is REST?", "input": "", "output": "REST is an architectural style for designing networked APIs using HTTP methods."}
```

Or create it with Python one-liner:

```cmd
python -c "import json; data=[{'instruction':q,'input':'','output':a} for q,a in [('What is Python?','A high-level programming language.'),('Explain recursion','A function calling itself.'),('What is a list?','An ordered mutable collection.'),('What is OOP?','Programming with objects and classes.'),('What is Git?','A version control system.'),('What is Docker?','A containerization platform.'),('What is SQL?','A database query language.'),('What is REST?','An API architectural style.'),('What is an API?','An interface for software communication.'),('What is CSS?','A stylesheet language for web pages.')]]; f=open('test_data.jsonl','w'); [f.write(json.dumps(d)+'\n') for d in data]; f.close(); print('Created test_data.jsonl')"
```

## 4. Data Tools

```cmd
soup data inspect test_data.jsonl

soup data validate test_data.jsonl --format alpaca

soup data stats test_data.jsonl

soup data convert test_data.jsonl --to sharegpt -o test_sharegpt.jsonl

soup data convert test_data.jsonl --to chatml -o test_chatml.jsonl

soup data inspect test_sharegpt.jsonl

soup data inspect test_chatml.jsonl

soup data merge test_data.jsonl test_sharegpt.jsonl -o test_merged.jsonl --shuffle

soup data dedup test_merged.jsonl -o test_deduped.jsonl --threshold 0.8
```

## 5. Create Config for Training

Create file `test_soup.yaml` (copy-paste into text editor):

```yaml
base: TinyLlama/TinyLlama-1.1B-Chat-v1.0

data:
  train: test_data.jsonl
  format: alpaca
  max_length: 256

training:
  epochs: 2
  lr: 2e-4
  batch_size: 2
  quantization: 4bit
  logging_steps: 1
  save_steps: 50
  lora:
    r: 8
    alpha: 16
    dropout: 0.05

output: ./test_output
```

## 6. Dry Run (validate without training)

```cmd
soup train -c test_soup.yaml --dry-run
```

## 7. Train

```cmd
soup train -c test_soup.yaml --name "local-test"
```

Training should take ~1-3 minutes on 3050 with this tiny dataset.

## 8. Experiment Tracking

```cmd
soup runs
```

Copy the Run ID from the output, then:

```cmd
soup runs show RUN_ID_HERE
```

Example: `soup runs show run_20260304_004948_983f284d`

## 9. Chat with Fine-Tuned Model

```cmd
soup chat -m ./test_output
```

Type questions, then type `exit` to quit.

## 10. Merge LoRA

```cmd
soup merge -a ./test_output -o ./test_merged_model
```

## 11. Export to GGUF (optional, needs llama.cpp + cmake)

```cmd
soup export -m ./test_merged_model -q q4_k_m -o test_model.gguf
```

## 12. Eval (optional, slow)

```cmd
pip install lm-eval
soup eval -m ./test_output --benchmarks hellaswag --batch-size 4
```

---

## Cleanup (Windows)

```cmd
rmdir /s /q test_output test_merged_model
del test_data.jsonl test_sharegpt.jsonl test_chatml.jsonl test_merged.jsonl test_deduped.jsonl
del test_soup.yaml test_chat.yaml test_code.yaml test_model.gguf
```

Or in PowerShell:

```powershell
Remove-Item -Recurse -Force test_output, test_merged_model -ErrorAction SilentlyContinue
Remove-Item test_data.jsonl, test_sharegpt.jsonl, test_chatml.jsonl, test_merged.jsonl, test_deduped.jsonl, test_soup.yaml, test_chat.yaml, test_code.yaml, test_model.gguf -ErrorAction SilentlyContinue
```

## Expected Results

| Step | Expected |
|------|----------|
| Version | `soup v0.2.1` |
| Init templates | Creates yaml files |
| Data inspect | Table with stats + sample rows |
| Data validate | "20/20 rows valid" |
| Data stats | Length distribution + histogram |
| Data convert | Creates sharegpt/chatml jsonl files |
| Data merge | Merges into single file |
| Data dedup | Removes near-duplicates |
| Dry run | "Config valid" or similar |
| Train | Loss decreasing, ~1-3 min |
| Runs | Shows run with metrics |
| Chat | Model responds (quality low with 10 samples — that's OK) |
| Merge | Creates full model in test_merged_model/ |
