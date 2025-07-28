# Quantized Airavata

This repository presents a quantized implementation of the `ai4bharat/Airavata` language model, deployed via a FastAPI backend. The goal is to improve inference efficiency on both CPU and GPU by reducing model size and memory requirements through quantization. The project also includes benchmarking scripts to measure latency and throughput.

---

## Project Overview

- **Model**: ai4bharat/Airavata (7B OpenHathi variant fine-tuned on the IndicInstruct dataset)  
- **Quantization**: 4-bit using `bitsandbytes`  
- **Backend**: FastAPI  
- **Tested on**: Google Colab (T4 GPU) with latency and throughput benchmarks

---
## Google Colab Notebook

All quantization, benchmarking, and FastAPI integration were implemented and tested in a single Google Colab notebook.


The notebook includes:
- Quantization using `bitsandbytes`
- FastAPI setup and endpoint testing
- Latency and throughput benchmarking


> **Note:** Enable GPU runtime in Colab via `Runtime > Change runtime type > GPU` before running.

---
## Quantization Details

| Parameter                | Value     |
|--------------------------|-----------|
| Quantization Type        | int4      |
| Quantization Format      | nf4       |
| Use Double Quantization  | True      |
| Compute Data Type        | bfloat16  |
| Storage Data Type        | bfloat16  |

The model was quantized using the `bitsandbytes` 4-bit quantization method with Hugging Face’s `transformers` integration.

---

## About the Original Model

The base model is a 7B parameter instruction-tuned language model from AI4Bharat, fine-tuned using LoRA on the IndicInstruct dataset. This dataset includes diverse instruction-following corpora such as Anudesh, wikiHow, Flan v2, Dolly, Anthropic-HHH, OpenAssistant v1, and LymSys-Chat.

This model was introduced in the technical report *Airavata: Introducing Hindi Instruction-tuned LLM*.  
You can find the training and evaluation code at the [IndicInstruct GitHub repository](https://github.com/AI4Bharat/IndicInstruct).

---

## Repository Structure

| File/Directory            | Description                                   |
|---------------------------|-----------------------------------------------|
| `app.py`                  | FastAPI backend for model inference           |
| `quantize_and_save.py`    | Script to quantize and store model/tokenizer  |
| `benchmark.py`            | Benchmarking script for latency/throughput    |
| `requirements.txt`        | List of Python dependencies                   |
| `README.md`               | Project documentation                         |
| `Quantized_Airavata/`     | Output directory for quantized model          |

> **Note:** The `Quantized_Airavata/` directory is not included in the repository due to file size constraints. It can be generated locally using `quantize_and_save.py`.

---

## Benchmark Results


| Metric       | Base Model (bfloat16) | Quantized Model (4-bit) | Quantized Model (8-bit) |
|--------------|-------------------|--------------------------|--------------------------|
| Latency      | ~1277.75 ms/req   | ~1817.08 ms ms/req          | ~2010.42 /req          |
| Throughput   | ~0.78 req/sec     | ~0.42 req/sec            | ~0.50 req/sec            |
| Model Size   | ~13.74 GB         | ~3.85 GB                 | ~6.70 GB                 |

The quantized model shows higher latency on Colab's T4 GPU, likely due to lack of INT4 kernel optimization. Performance is expected to improve on newer GPUs like A100/H100.
The quantized model shows a **significant reduction in memory usage**.

INT8 quantization was also tested for reference, but the 4-bit version performed better overall.

---

## Output Comparison on a Hindi Prompt

To qualitatively evaluate the impact of quantization, we tested both the original and quantized models on a Hindi-language instruction prompt.

**Prompt (Hindi):**  
*“मैं अपने समय प्रबंधन कौशल को कैसे सुधार सकता हूँ? मुझे पांच बिंदु बताएं।”*  
(*Translation: How can I improve my time management skills? Give me five points.*)

---

###  Original Model Output:


यहाँ पाँच बिंदु दिए गए हैं जो आपको अपने समय प्रबंधन कौशल में सुधार करने में मदद कर सकते हैंः

1.प्राथमिकता देंः उन कार्यों को प्राथमिकता दें जिन्हें पूरा करने की आवश्यकता है और उन्हें प्राथमिकता दें।

2.एक अनुसूची बनाएँः एक अनुसूची बनाएँ और उस पर टिके रहें। यह आपको अपने कार्यों पर ध्यान केंद्रित करने और उन्हें पूरा करने के लिए आवश्यक समय देने में मदद करेगा।

3.ब्रेक लेंः नियमित ब्रेक लें और रिचार्ज करें। यह आपको अधिक उत्पादक और केंद्रित रहने में मदद करेगा।

4.ध्यान भटकाने से बचेंः अपने फोन को बंद कर दें और ध्यान भटकाने से बचें। इससे आपको अपने कार्यों पर ध्यान केंद्रित करने और उन्हें पूरा करने में मदद मिलेगी।

5.व्यवस्थित रहेंः अपने कार्यस्थल को व्यवस्थित रखें और अपने कागजात और अन्य वस्तुओं को व्यवस्थित रखें। इससे आपको अपने कार्यों पर ध्यान केंद्रित करने और उन्हें पूरा करने के लिए आवश्यक समय देने में मदद मिलेगी।

---

###  Quantized 4-bit Model Output:
समय प्रबंधन कौशल

1.एक अनुसूची बनाएँः एक अनुसूची बनाएँ और उस पर टिके रहें।

2.समय का प्रबंधन करेंः कार्यों को प्राथमिकता दें और उन्हें पूरा करने के लिए निर्धारित समय पर निर्धारित करें।

3.कार्यों को पूरा करेंः एक बार जब आप एक कार्य निर्धारित कर लेते हैं, तो उसे पूरा करें।

4.ध्यान देंः एक कार्य पर ध्यान दें और उसे पूरा करने के लिए समर्पित समय दें।

5.ब्रेक लेंः ब्रेक लेना महत्वपूर्ण है ताकि आप फिर से ऊर्जावान और उत्पादक महसूस कर सकें।"

---


### Observations

- The **original model** provides more detailed, fluent, and well-formatted output.
- The **quantized model** retains core instructional quality, though it exhibits some redundancy and slightly less coherent phrasing.
- Overall, the quantized model is still effective for Hindi instruction tasks while offering considerable efficiency gains.

---



