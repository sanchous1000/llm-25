---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 8
total_pages: 13
date_converted: "2025-11-05"
---

4
Experiments
4.1
Baselines
We adopt the following three types of baselines to conduct a holistic evaluation of AcademicEval.
## Standard
LLMs.
## We choose Gemma Instruct (7B) (Team et al., 2024) and LLaMA-3 Chat
(70B) (AI@Meta, 2024) as standard LLM baselines, each with a context length of 8K.
Long-context LLMs.
## We choose Qwen 1.5 Chat (72B) (Bai et al., 2023a), Mixtral-8x7B Instruct
(46.7B) (Jiang et al., 2024), and Nous Hermes 2 - Mixtral 8x7B-DPO (46.7B) (Teknium et al.) as long-context
LLM baselines, each with a context length of 32K.
## Retrieval-augmented language models (RALM). First, we consider two sparse retrievers:
(1)
BM25 (Robertson et al., 2009): This is a widely used retrieval model that ranks documents based on
the frequency of query terms in each document. (2) TF-IDF (Ramos et al., 2003): It scores documents
by multiplying the term frequency of each query term by the inverse document frequency. Second, we also
consider three dense retrievers: (3) DPR (Karpukhin et al., 2020): It uses a bi-encoder to retrieve relevant
documents based on dense embeddings. (4) Contriever (Izacard et al., 2021): It leverages unsupervised
contrastive learning to learn high-quality dense representations. (5) Dragon (Lin et al., 2023): It enhances
retriever training by employing data augmentation, including query and label augmentation.
4.2
Settings
API Access. In this paper, we conduct a comprehensive evaluation over AcademicEval benchmark using
the LLM API provided by together.ai5. For each API call, we fix the temperature parameter to 0 (i.e.,
greedy decoding).
## Input Truncation. By default, we use a BERT tokenizer to calculate the number of input tokens for
AcademicEval. However, since the tokenizer of each LLM is usually different, it will cause some inputs
to exceed the context length limit of the LLM. Therefore, for the evaluation of each LLM, we additionally
download its tokenizer configuration file from the official website at Hugging Face, which is utilized to ensure
correct and accurate truncation of input tokens.
## Refinement of LLM Responses. For the Title Writing task, the responses of LLMs are relatively
short.
## If the response contains some extra redundant information, it will have a greater impact on the
evaluation metric score (although we have given LLM instructions not to generate irrelevant information).
Therefore, for the Title Writing task, we additionally refine the LLM responses, for example, removing
irrelevant information such as “here is the title”. For other tasks, since LLM’s responses are relatively long,
occasional small amounts of irrelevant information will not have a significant impact on the evaluation, so
we do not perform any refinement on LLM’s responses in this case.
## Details of the Implementation of RALM. We use the inputs of AcademicEval as the external corpus
of RALM (such as Target Content and Reference Content introduced in Section D). For text split, we use
the RecursiveCharacterTextSplitter from LangChain6 and set chunk size and chunk overlap to 512 and 64,
respectively. For each retrieval, we recall up to 12 text chunks (limited by the context length of standard
LLMs) based on text similarity (semantic similarity based on inner product for dense retrievers or similarity
based on word frequency for sparse retrievers).
8
