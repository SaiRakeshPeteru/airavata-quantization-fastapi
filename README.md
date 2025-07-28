# Quantized Airavata: FastAPI Deployment and Benchmarking

This repository presents a quantized implementation of the `ai4bharat/Airavata` language model, deployed via a FastAPI backend. The goal is to improve inference efficiency on both CPU and GPU by reducing model size and memory requirements through quantization. The project also includes benchmarking scripts to measure latency and throughput.

## Project Overview

- Model: ai4bharat/Airavata (7B OpenHathi variant fine-tuned on the IndicInstruct dataset)
- Quantization: 4-bit using `bitsandbytes`
- Backend: FastAPI
- Tested on: Google Colab (T4 GPU) with latency and throughput benchmarks


## Quantization Details

| Parameter                     | Value       |
|------------------------------|-------------|
| Quantization Type            | int4        |
| Quantization Format          | nf4         |
| Use Double Quantization      | True        |
| Compute Data Type            | bfloat16    |
| Storage Data Type            | bfloat16    |

The model was quantized using the `bitsandbytes` 4-bit quantization method with the Hugging Face `transformers` integration.

## About the Original Model

The base model is a 7B parameter instruction-tuned LLM from AI4Bharat, trained using LoRA on the IndicInstruct dataset. This dataset includes diverse instruction-following corpora such as Anudesh, wikiHow, Flan v2, Dolly, Anthropic-HHH, OpenAssistant v1, and LymSys-Chat.

This was trained as part of the technical report Airavata: Introducing Hindi Instruction-tuned LLM. The codebase used to train and evaluate this model can be found at [IndicInstruct GitHub repository](https://github.com/AI4Bharat/IndicInstruct).

## Input Format.

 

The model expects input in a chat format similar to Open-Instruct:



```

<|user|>
Your message here!
<|assistant|>

````

It is important to include a newline after `<|assistant|>` for optimal generation quality.

## Repository Structure

| File                  | Description                                   |
|-----------------------|-----------------------------------------------|
| `app.py`              | FastAPI backend for serving generation        |
| `quantize_and_save.py`| Quantizes and saves model/tokenizer           |
| `benchmark.py`        | Script to benchmark latency and throughput    |
| `requirements.txt`    | Python dependencies                           |
| `README.md`           | This document                                 |
| `Quantized_Airavata/` | Output directory for the quantized model      |

Note: The `Quantized_Airavata/` directory is excluded from the GitHub repository due to file size constraints. It can be generated locally using `quantize_and_save.py`.



## Benchmark Results

Benchmarks were conducted on Google Colab (NVIDIA T4 GPU), comparing the base (FP16) model and the quantized (4-bit) model.

| Metric     | Base Model (FP16) | Quantized Model (4-bit) |
| ---------- | ----------------- | ----------------------- |
| Latency    | \~950 ms/request  | \~460 ms/request        |
| Throughput | \~1.05 req/sec    | \~2.17 req/sec          |
| Model Size | \~13.2 GB         | \~4.7 GB                |

The quantized model demonstrates significant improvements in memory efficiency and inference performance, while maintaining reasonable generation quality.

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Quantize the model (one-time step):

   ```bash
   python quantize_and_save.py
   ```

3. Start the FastAPI server:

   ```bash
   uvicorn app:app --reload
   ```

4. Run benchmarks (optional):

   ```bash
   python benchmark.py
   ```


```


