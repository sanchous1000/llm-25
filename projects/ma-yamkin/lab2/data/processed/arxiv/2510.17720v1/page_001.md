---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 1
total_pages: 8
date_converted: "2025-11-05"
---

PANER: A Paraphrase-Augmented Framework for
Low-Resource Named Entity Recognition
1st Nanda Kumar Rengarajan
Concordia University
nanda.kumark@mail.concordia.ca
2nd Jun Yan
Concordia University
jun.yan@concordia.ca
3rd Chun Wang
Concordia University
chun.wang@concordia.ca
Abstract—Named Entity Recognition (NER) is a critical task
that requires substantial annotated data, making it challenging in
low-resource scenarios where label acquisition is expensive. While
zero-shot and instruction-tuned approaches have made progress,
they often fail to generalize to domain-specific entities and do not
effectively utilize limited available data. We present a lightweight
few-shot NER framework that addresses these challenges through
two key innovations: (1) a new instruction tuning template with a
simplified output format that combines principles from prior IT
approaches to leverage the large context window of recent state-
of-the-art LLMs; (2) introducing a strategic data augmentation
technique that preserves entity information while paraphrasing
the surrounding context, thereby expanding our training data
without compromising semantic relationships. Experiments on
benchmark datasets show that our method achieves performance
comparable to state-of-the-art models on few-shot and zero-
shot tasks, with our few-shot approach attaining an average F1
score of 80.1 on the CrossNER datasets. Models trained with
our paraphrasing approach show consistent improvements in
F1 scores of up to 17 points over baseline versions, offering
a promising solution for groups with limited NER training data
and compute power.
## Index Terms—Named Entity Recognition (NER), Few-Shot
Learning, Large Language Models (LLMs), Instruction Tuning,
Data Augmentation.
I. INTRODUCTION
Named Entity Recognition (NER) is a foundational task
in Natural Language Processing (NLP), enabling applications
like information extraction, question answering, and event
detection [1]. Traditional NER systems rely on supervised
learning, requiring extensive annotated data for specific do-
mains and predefined entity types. This dependency on large,
labelled datasets limits their adaptability to new domains and
entity categories. Recent breakthroughs in Large Language
Models (LLMs) have enabled more flexible NER approaches
through instruction tuning, demonstrating promising zero-
shot and few-shot capabilities without extensive labelled data.
## Approaches like InstructUIE [2] and UniversalNER [3] show
strong generalization across diverse entity types. However,
these methods often underperform in specialized domains and
face practical constraints, either requiring substantial compu-
tational resources or suffering from slow inference times [4].
## Developments in instruction tuning-based NER approaches
have introduced key innovations that inspire our approach.
## The SLIMER [4] framework emphasizes the use of enriched
prompts, incorporating definitions and annotation guidelines to
improve performance on unseen entities. GNER [5] highlights
Fig. 1. Illustration of paraphrasing-based data augmentation process.
the importance of negative instances, improving contextual un-
derstanding and entity boundary delineation by including non-
entity text. We combine the strengths of both approaches—the
negative instance inclusion suggested by GNER [5] and the
guideline-centric philosophy of SLIMER [4] — to create an
instruction-tuning template that clearly defines entity bound-
aries and types. Our framework adopts a simplified ”word/tag”
output format, reducing complexity, particularly in low-data
scenarios. This integration balances robust entity representa-
tion with efficient domain adaptation.
## We also focus on the expanded context lengths offered
by modern instruction-tuned models. Specifically, we leverage
the 128k context windows of Qwen-2.5-Instruct (7B) [6] and
LLAMA-3.1-Instruct (8B) [7], as well as the 32k context
window of Falcon3-Instruct (10B) [8], enabling our framework
to process longer and more complex inputs efficiently. These
extended context lengths, combined with enriched instruction
tuning, provide a foundation for robust and scalable NER
arXiv:2510.17720v1  [cs.CL]  20 Oct 2025
