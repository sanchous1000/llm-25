---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 2
total_pages: 8
date_converted: "2025-11-05"
---

performance across diverse and challenging tasks.
## Data augmentation has emerged as a key strategy for ad-
dressing limited training data in NER. Traditional methods fo-
cus on generating augmented samples through back-translation
[9], and entity-controlled generation using question-answering
techniques [10]. MELM [28] introduced a data augmentation
strategy that injects entity labels into the training context,
reducing token-label misalignment and improving entity di-
versity, particularly in low-resource and multilingual NER
settings.
## We introduce a targeted approach that modifies only the con-
text surrounding entities. This preserves semantic relationships
while expanding the linguistic variety, as illustrated in Figure
1. LLM-DA [11] has already demonstrated the effectiveness of
large language models in generating diverse training examples
by employing context-level rewriting strategies. Our approach
builds on this idea by also working on sentences with multiple
entities, ensuring paraphrased variants remain semantically
consistent and merging entity representations to refine NER
model predictions. Overall, this technique improves model
adaptability to domain-specific entities and helps bridge the
performance gap in low-resource settings.
## Our experiments on benchmark datasets, including Cross-
NER [12] and MIT [13], demonstrate that our zero-shot
framework achieves comparable performance to state-of-the-
art zero-shot models while requiring fewer computational re-
sources. Models using the paraphrasing augmentation consis-
tently outperform baseline versions (without augmented data),
validating the data strategy. Our key contributions include:
• An instruction tuning template that combines negative
instance style annotation with definition- and guideline-
based schema.
• A controlled paraphrasing technique for generating high-
quality samples combined with an easily replicable low-
resource training method for domain-specific NER.
## The remainder of this paper is organized as follows. Section
II reviews related work in Named Entity Recognition (NER),
with a focus on zero-shot and few-shot learning using Large
Language Models (LLMs). Section III describes the proposed
methodology, detailing the paraphrasing-based data augmen-
tation framework and the implementation details, including
optimization strategies and validation techniques. Section IV
talks about the instruction tuning approach and template-used.
## Section V presents the experimental setup, covering dataset
selection, baseline models, and evaluation metrics. Section VI
reports the experimental results and provides a comparative
analysis of model performance. Section VII discusses key
findings, limitations, and implications for future research.
II. RELATED WORK
Named Entity Recognition (NER) has traditionally been
approached as a sequence labelling task, where models are
trained to assign BIO (Beginning, Inside, Outside) tags to input
tokens [1]. While supervised approaches using BERT-based
architectures have shown strong performance, they remain
constrained by their reliance on predetermined label sets and
domain-specific training data [14].
## The emergence of Large Language Models (LLMs) has
introduced new paradigms for addressing NER challenges,
particularly in zero-shot and few-shot scenarios. Early work
by [15] demonstrated LLMs’ capacity for multi-task learning
through natural language instructions, laying the groundwork
for what would become known as in-context learning and
prompt engineering. However, initial attempts to apply LLMs
to Information Extraction tasks, including NER, revealed
significant limitations compared to traditional supervised ap-
proaches [2], [16].
## Few-shot NER remains challenging due to knowledge trans-
fer limitations across domains. Recent research has explored
various instruction-tuning strategies to enhance LLMs’ perfor-
mance on NER tasks. Notable approaches include InstructUIE
[2], which utilizes a T5-11B architecture fine-tuned on infor-
mation extraction datasets, and UniNER [3], which employs a
conversational template with LLAMA. These methods have
shown promise but often require substantial computational
resources for training and inference. GoLLIE [17] introduced
an innovative approach by incorporating annotation guidelines
such as Python docstrings, marking the first attempt to encode
labelling criteria within the prompt structure explicitly. While
instruction tuning has emerged as a leading method for gen-
eralization to unseen tasks [18], [19] current approaches face
several limitations.
1) Many methods rely solely on label names in prompts
without considering domain-specific definitions or com-
plex label semantics [20].
2) The computational requirements of these models often
make them impractical for organizations with limited
resources. Additionally, the challenge of effectively uti-
lizing available domain data remains largely unaddressed
in current instruction-tuning frameworks.
## Recent studies have begun to address the efficiency-
performance trade-off in NER systems. GNER [5] intro-
duced a modified BIO-like generation approach that improves
boundary detection while addressing classification indecision.
GLiNER [21] demonstrated that smaller, non-instruction-tuned
architectures could achieve competitive performance in both
supervised and zero-shot settings, suggesting the potential for
more resource-efficient approaches.
III. PROPOSED METHOD
Our data augmentation step, illustrated in Figure 1, was
developed to address a fundamental challenge in Named Entity
Recognition (NER) - the scarcity of annotated training data.
## We leveraged the LLAMA 3.3-70B [22] model for generating
paraphrases, using the prompt shown in Fig. 2. We chose this
model for its ability to match the performance of larger models
like LLAMA 3.1-405B while operating with substantially
fewer parameters (70B vs 405B), making it more practical
for real-world applications [23].
