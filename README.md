# Multimodal Document Retrieval System — Group 22

**PCAM\* ZG516 Applied Generative AI · Group Major Project / Assignment 2 · Part A**

| Member | BITS ID |
|---|---|
| Nandhaluru Kishore | 2025paml539 |
| Vishruth M | 2025paml536 |
| Satya Prakash | 2025paml561 |
| Takkars Bhavani | 2025paml562 |

## Status

Setup. The corpus domain is awaiting instructor approval. Nothing below is final.

## What this will be

A retrieval system that answers questions from at least 100 PDF and PPT documents, using their
text, tables and images, and cites the source document and page for every answer.

## Compute

- Development and tests: a Windows laptop (CPU only).
- Ingestion, embeddings and LLM runs: the BITS Kubeflow lab (NVIDIA A100 80 GB, driver 535 / CUDA 12.2,
  PyTorch 2.8.0+cu128).
