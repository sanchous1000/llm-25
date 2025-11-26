---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 5
total_pages: 8
date_converted: "2025-11-05"
---

TABLE I
COMPARISON BETWEEN INSTRUCTION FORMATS
AI
Lit
Music
Pol
Sci
Avg
GNER-BIO
52.1
51.1
58.5
54.1
43.8
51.92
Ours-slash w/o*
59.1
67.4
72.25
70.8
66.1
67.13
Ours-slash
63.9
67.2
75.3
67.8
68.7
68.58
*w/o: prompt without guidelines
3) GNER [5]: A model released in two variants, each
leveraging a different backbone architecture:
• GNER-T5: Based on flan-t5-xxl.
• GNER-LLAMA: Built on the LLAMA-7B architec-
ture. Both versions emphasize the incorporation of
entity definitions during instruction-tuning.
4) SLIMER [4]: A model based on the LLAMA-2-7B
chat architecture, fine-tuned with LoRA [24] for 10
epochs. SLIMER integrates structured annotation guide-
lines, making it a strong benchmark for guideline-based
NER.
5) DAGA [27]: DAGA utilizes a one-layer LSTM-based
language model trained on linearized labelled sentences
from CoNLL and other sequence-tagging datasets to
generate synthetic training data for the same.
6) MELM [28]: a data augmentation framework that en-
sures label-consistent entity replacements by fine-tuning
XLM-RoBERTa with masked entity prediction.
C. Backbone LLMs and Evaluation Framework
We used Qwen-2.5-Instruct (7B), LLAMA-3.1-Instruct
(8B), and Falcon3-Instruct (10B) as our backbone models,
selected based on their extended context lengths: 128K for
Qwen-2.5 and LLAMA-3.1, and 32K for Falcon3, respectively,
as well as their state-of-the-art performance on instruction-
following benchmarks such as MT-Bench [29] and Alpaca WC
[30].
## Our evaluation strategy includes both few-shot and zero-
shot scenarios to assess model performance under varying
resource constraints. For zero-shot evaluation, we fine-tuned
the models above on 23,402 pileNER samples, as described
in Section V-A. Although we utilize more samples than [5],
our training setup (described below) is significantly more
efficient and effectively bridges the performance gap while still
serving as a cost-effective solution. For few-shot evaluation,
we used 10,000 samples from pileNER as our base dataset and
added domain-specific examples from the benchmark datasets
(CrossNER and MIT) as necessary.
## Training Setup:
All models were fine-tuned on the
Modal platform using Axolotl. Fine-tuning was conducted
with LoRA [24] settings of r = 8, α = 16, and the AdamW
optimizer [31] for one epoch. A cosine learning rate schedule
was employed, starting with a warm-up phase covering 4% of
the training steps and peaking at 2 × 10−5.
## Fig. 4. Impact of augmented sample size on model performance (F1 score,
in %) for CrossNER dataset [12].
VI. RESULTS
A. Comparison of Tagging Formats
We first validate the effectiveness of our word/tag format by
comparing it against the BIO-style output format presented in
[5]. For this experiment, LLAMA 3.1-8B-Instruct was used
as the backbone architecture, and we leveraged the same
PileNER dataset [3] and filtered for sentences with more
than 10 words only in English, resulting in approximately
23,402 samples for training. The model was fine-tuned with
LoRA with the hyperparameters in Section V-C. To maintain
consistency with GNER [5], we evaluated the model on the
same five datasets, with 200 random samples per dataset, for
zero-shot performance analysis. Results reported in Table I
represent averages over three test runs to ensure robustness.
## Reference [5] performed a similar boundary analysis with
different tagging formats to determine the optimal approach.
## While their results differ from ours, this discrepancy could be
attributed to our use of a larger model, the LLAMA 3.1-8B,
compared to their Flan-T5 large model with 780M parameters.
## Additionally, we employ a LoRA fine-tuning approach for a
single epoch, in contrast to their full fine-tuning over three
epochs, which may also contribute to the observed differences
in performance.
## Our evaluation methodology for comparing different tagging
formats, we employ entity-level F1 scores for both the BIO
and word/tag formats to ensure fair comparison. While the
formats differ in presentation, our evaluation criteria remain
consistent across both approaches: an entity prediction is
considered correct only when both the entity type and its
complete boundaries match the gold standard annotation. This
means that in both formats, partial entity identification or
incorrect boundary detection is treated as an error, regardless
of whether the entity type was correctly identified.
## Table I shows a significant improvement in the F1 score
when adopting the word/tag format compared to the tradi-
tional BIO tagging schema. A slight increase in F1 score
