---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 4
total_pages: 8
date_converted: "2025-11-05"
---

## Instruction Tuning Prompt
Task Description:
Please analyze the sentence provided, identifying the
type of entity for each word on a token-by-token basis.
## Each word in the sentence should be annotated with
its corresponding named entity tag, using a forward
slash / between the word and the tag. Output format is:
word_1/label_1, word_2/label_2, ...
## Guideline:
1) Use O for words that are not part of any named
entity.
2) For multi-word entities, label each word with the
same entity tag.
Use the specific entity tags: l1, l2, . . . , lm, and O.
## To help you, here are dedicated DEFINITION and
GUIDELINES for each entity tag.
{ l1 : {
DEFINITION : ,
GUIDELINES : }
}
Input: x1
x2
. . .
xn
Output: x1/ˆy1
x2ˆy2
. . .
xn/ˆyn
Fig. 3. Prompt used for Instruction-tuning LLMs.
non-entity tokens by utilizing surrounding context, reducing
reliance on direct memorization of entity names. By preserving
this contextual learning mechanism and combining it with our
simplified tagging format, we establish a robust framework
that enhances performance for entity recognition tasks. Table I
compares the result of different formats for instruction tuning.
## When applying this new instruction prompt to the Cross-
NER [12] Science dataset with 16 entity types in the training
prompt, including task description, annotations, and guidelines
for all NEs, added up to 1700 tokens, well below the context
length of the models used in our experiments. This token effi-
ciency allowed us to include comprehensive task instructions
and entity definitions directly in the input prompts without
exceeding the model’s context limit. The results demonstrate
that this format is not only feasible but also beneficial for
few-shot and zero-shot settings.
## We employ LoRA [24] fine-tuning for a single epoch to
further optimize training, reducing computational and memory
requirements while achieving performance comparable to full
fine-tuning methods. The goal is to achieve comparable or
superior performance using a fraction of the resources, partic-
ularly for low-resource NER datasets.
## For datasets with longer sentences or complex annotations,
a chunking strategy is used to split inputs into manageable
segments while preserving overall context. To ensure reliable
response generation, the context length is limited to 2048
tokens. As a result, any sequences exceeding this threshold
are automatically segmented into multiple examples. Fig. 3
shows the full instruction tuning prompt used.
V. EXPERIMENT SETUP
A. Datasets
We use PileNER [3] as the main training corpus, with key
pre-processing steps to ensure data quality and consistency.
## The pre-processing pipeline includes the following filtering
criteria:
1) Minimum sentence length threshold of 10 words to
ensure sufficient context.
2) Language filtering to retain only English text.
3) Entity type filtering to focus on 423 named entities with
established guidelines and annotations, as documented
by [4]
This filtered dataset yielded approximately 23,402 high-
quality samples. To maintain experimental consistency with
prior work [5], 10,000 samples are randomly selected from
this preprocessed pool as a starting point for our few-shot
testing.
## We evaluate the proposed approach on four established
benchmarks, each chosen to assess different aspects of model
performance:
1) CrossNER [12]: A comprehensive cross-domain dataset
that evaluates domain adaptation capabilities across di-
verse subject areas, including scientific papers, politics,
music, and literature.
2) MIT [13]: A standard benchmark for assessing out-of-
distribution (OOD) performance, particularly valuable
for evaluating generalization to novel domains.
3) BUSTER [25]: A document-level financial domain NER
benchmark that presents unique challenges through its
specialized entity types and complex document struc-
ture.
4) CoNLL [26]: CoNLL-2003 shared task dataset is a
widely-used benchmark for NER featuring multilingual
annotated text with general tags like PERSON, LOCA-
TION, and others.
B. Baseline Comparisons
To evaluate the effectiveness of our approach, we compare it
against several state-of-the-art methods for zero-shot and few-
shot Named Entity Recognition (NER) and data augmentation.
## Each baseline represents a distinct methodology or model
architecture, providing a diverse comparison framework.
1) GoLLIE [17]: A generative model based on Code-
LLAMA, designed to leverage annotation guidelines
formatted in a code-like representation. We use the 7B
variant of GoLLIE for comparability with other models
in this study.
2) GLiNER-L [21]: An encoder-only model based on De-
BERTa with 304 million parameters. Despite being the
smallest model among the selected baselines, GLiNER-
L has demonstrated competitive performance in out-of-
distribution (OOD) zero-shot NER tasks.
