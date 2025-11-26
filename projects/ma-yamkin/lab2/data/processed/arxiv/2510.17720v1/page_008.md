---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 8
total_pages: 8
date_converted: "2025-11-05"
---

D. Performance of Instruction Tuning Template in Zero-shot
NER
While our primary goal is to improve few-shot Named En-
tity Recognition (NER), as demonstrated above, the proposed
instruction-tuning template also performs competitively with
state-of-the-art methods in zero-shot NER. Table VI presents
the zero-shot performance compared to existing state-of-the-
art benchmarks. Notably, the Falcon-3 [8] model achieves an
average F1 score of 0.648, close to the GNER-T5 [5] and
GNER-LLAMA [5] models. This performance was obtained
via fine-tuning with LoRA [24], requiring significantly less
computational time and resources: only one epoch was suffi-
cient to achieve this performance, whereas GNER was fully
fine-tuned for three epochs.
## Further, the out-of-domain performance of our instruction-
tuning template is showcased in Table VII, where we com-
pare it against the above-mentioned models on the BUSTER
dataset. The results demonstrate that the proposed model
performs better than both GNER [5] models with an F1 of
0.336 and achieves performance comparable to SLIMER [4],
which holds the state-of-the-art F1 score of 0.4527.
VII. CONCLUSIONS
In this work, we presented PANER, a paraphrase-augmented
framework designed to enhance Named Entity Recognition
(NER) in low-resource settings. The approach integrates in-
struction tuning with paraphrase-based data augmentation,
enabling improved performance while maintaining compu-
tational efficiency. Experimental results demonstrate that
PANER achieves competitive performance with state-of-the-
art zero-shot NER models while requiring significantly fewer
computational resources. The paraphrasing technique consis-
tently improves entity recognition, particularly in domain-
specific and few-shot learning settings. The results indicate
that our approach is an effective alternative for organizations
with limited access to annotated datasets and compute power.
## However, our study has certain limitations. First, while
paraphrasing improves model generalization, the quality of
generated variations can vary depending on the complexity
of the input sentences. Additionally, the approach to include
guidelines and annotations for all entity types does not benefit
cases where the entity is negatively affected by the guidelines.
## Prior work SLIMER [4] did an entity-by-entity analysis to see
how guidelines and annotations are helping each entity, and
the results show some entities do not require or benefit from
the presence of guidelines and annotations. Since we process
entire sentences and extract all entities in a single request, it
is difficult to selectively include or exclude guidelines based
on specific entity types, which could impact performance for
certain categories.
## While
our
paraphrasing-based
augmentation
approach
demonstrates promising results, we acknowledge a few lim-
itations. First, our strict entity preservation constraints, though
effective for maintaining semantic relationships, may restrict
the diversity of the generated samples. During our analysis,
we observed that augmented sentences often exhibit limited
structural variation when multiple entities appear in close
proximity, as the model prioritizes preserving entity positions
over introducing novel sentence constructions. This trade-off
between entity integrity and linguistic diversity represents an
inherent tension in our current implementation. Additionally,
the paraphrasing approach occasionally struggles with domain-
specific terminology and complex syntactic structures, result-
ing in approximately 15% of initially generated paraphrases
failing our validation checks and requiring regeneration.
VIII. FUTURE WORK
Future work will explore more flexible entity augmentation
strategies that preserve semantic relationships while allowing
controlled entity variations (such as replacing entities with
semantically equivalent alternatives within the same type) and
adaptive paraphrasing approaches that adjust constraint strict-
ness based on sentence complexity and domain characteristics.
## We also plan to investigate multi-stage augmentation pipelines
that combine paraphrasing with other techniques to further
enhance sample diversity while maintaining the crucial entity
relationships that drive NER performance.
## Another key direction for future work is refining selective
guideline inclusion, where entity-specific constraints could be
dynamically applied during instruction tuning. Further, while
our approach has demonstrated strong performance in English-
language datasets, its multilingual effectiveness remains unex-
plored. A critical next step is studying how paraphrase-based
augmentation can be effectively applied to other languages.
## These advancements will ensure PANER remains a scalable,
adaptable, and an efficient framework for real-world, multilin-
gual applications.
REFERENCES
[1] J. Li, A. Sun, J. Han, and C. Li, “A Survey on Deep Learning for Named
Entity Recognition,” IEEE Trans. Knowl. Data Eng., vol. 34, no. 1, pp.
50–70, Jan. 2022, doi: 10.1109/TKDE.2020.2981314.
[2] X. Wang et al., “InstructUIE: Multi-task Instruction Tuning for Unified
Information Extraction,” Apr. 17, 2023, arXiv: arXiv:2304.08085. doi:
10.48550/arXiv.2304.08085.
[3] W. Zhou, S. Zhang, Y. Gu, M. Chen, and H. Poon, “UniversalNER:
Targeted Distillation from Large Language Models for Open Named
Entity Recognition,” Aug. 06, 2023, arXiv: arXiv:2308.03279.
[4] A. Zamai, A. Zugarini, L. Rigutini, M. Ernandes, and M. Maggini,
“Show Less, Instruct More: Enriching Prompts with Definitions and
Guidelines for Zero-Shot NER,” Jul. 02, 2024, arXiv: arXiv:2407.01272.
[5] Y. Ding, J. Li, P. Wang, Z. Tang, B. Yan, and M. Zhang, “Rethinking
Negative Instances for Generative Named Entity Recognition,” Jun. 18,
2024, arXiv: arXiv:2402.16602.
[6] Qwen et al., “Qwen2.5 Technical Report,” Dec. 19, 2024, arXiv:
arXiv:2412.15115. doi: 10.48550/arXiv.2412.15115.
[7] A. Grattafiori et al., “The Llama 3 Herd of Models,” Nov. 23, 2024,
arXiv: arXiv:2407.21783. doi: 10.48550/arXiv.2407.21783.
[8] “Welcome to the Falcon 3 Family of Open Models!” Available:
https://huggingface.co/blog/falcon3
[9] U. Yaseen and S. Langer, “Data Augmentation for Low-Resource Named
Entity Recognition Using Backtranslation,” presented at the ICON, Aug.
2021.
[10] K. Aggarwal, H. Jin, and A. Ahmad, “Entity-Controlled Synthetic Text
Generation using Contextual Question and Answering with Pre-trained
Language Models”.
[11] ] J. Ye et al., “LLM-DA: Data Augmentation via Large Lan-
guage Models for Few-Shot Named Entity Recognition,” 2024, doi:
10.48550/ARXIV.2402.14568.
