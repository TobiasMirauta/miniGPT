
# Custom GPT Language Model from Scratch

An end-to-end, custom Generative Pre-trained Transformer (GPT) language model built entirely from scratch using PyTorch. This project demonstrates a deep understanding of Large Language Model (LLM) architecture, custom tokenization, and handling massive, out-of-core datasets for GPU training.

## 🚀 Project Overview

Instead of fine-tuning an existing model, this project implements the core mathematics and architecture of a GPT model from the ground up. It was trained on the **OpenWebText Corpus** (an open-source recreation of OpenAI's GPT-2 WebText dataset), requiring efficient data pipelines and GPU memory management to handle ~38GB of raw training text.

## 🛠️ Tech Stack & Tools

* **Core Framework:** PyTorch (with CUDA for GPU acceleration)
* **Data Pipeline:** Hugging Face `datasets`
* **Language:** Python 3
* **Environment:** Jupyter / Anaconda

## 🧠 Core Features & Architecture

* **Custom Transformer Architecture:** Implemented self-attention mechanisms, causal masking, and feed-forward neural networks directly in PyTorch.
* **Embedding Layers:** Built custom token embedding and positional embedding tables to process sequence context.
* **Massive Data Handling:** Engineered a data ingestion pipeline using Hugging Face `datasets` to stream and tokenize the 38GB OpenWebText dataset without crashing system RAM.
* **GPU Optimization:** Handled complex CUDA memory constraints, device-side synchronizations, and batching to optimize training on limited hardware.
* **Custom Tokenizer:** Created a character-level/sub-word encoding and decoding scheme to translate raw text into tensor batches for the model.

## 📈 Key Learnings & Challenges Overcome

* Debugged complex asynchronous CUDA errors (e.g., device-side asserts) caused by embedding table index mismatches.
* Managed strict package dependency trees and environment isolation for scientific Python libraries.
* Bridged the gap between theoretical deep learning mathematics and practical, object-oriented PyTorch code.

## 🙏 Acknowledgments
* **Dataset:** [OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/) by Aaron Gokaslan and Vanya Cohen.
* **Inspiration:** Architecturally inspired by the foundational GPT papers and Andrej Karpathy's neural network series.
