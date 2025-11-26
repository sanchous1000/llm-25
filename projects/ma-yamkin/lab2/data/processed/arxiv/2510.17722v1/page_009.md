---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 9
total_pages: 9
date_converted: "2025-11-05"
---

the accuracy of the context is more critical than its mere presence. Self-predicted context does not always
lead to performance gains and remains roughly on par with the no-context setting, as the model-generated
dialogue history may contain factual errors or semantic drift. These inconsistencies may accumulate over
multiple rounds, causing the model to be misled by “incorrect memories” in subsequent responses.
4.3.5
Effect of different numbers of frames
Figure 6: Ablation results of frames on different abil-
ities. (a) Results of Object Reference, Memory Recall,
Content Summary, and Proactive Interaction; (b) Re-
sults of Answer Refusal and Topic Shifting.
## In Figure 6, results of Qwen2.5-VL-7B are grouped
according to the number of frames, with the resolu-
tion fixed at 720p and the number of frames vary-
ing from 4 to 64. Several distinct trends emerge
from the results:
(1) Topic Shifting. The performance on topic shift-
ing remains largely unaffected by the number of
frames. This suggests that the ability to adapt to
unexpected user queries and maintain coherent re-
sponses is primarily dependent on dialogue-level
reasoning rather than fine-grained visual informa-
tion.
(2) Anwser Refusal. Models perform better on
answer refusal cases when fewer frames are pro-
vided. With limited visual evidence, the model becomes more cautious in generating answers and is less
likely to hallucinate unsupported content, while when more frames are provided, the model may overfit
to irrelevant visual cues and produce unwarranted responses, leading to decreased performance on this
ability.
(3) Long Context Benefits. For the other four abilities, as shown in Figure 6 (a), models’ performance
consistently improves with more frames, because longer visual evidence provides richer contextual
signals, which support more accurate reasoning.
4.3.6
Effect of different resolutions
Figure 7: Ablation results of resolutions
on different abilities.
## To further analyze the impact of video input quality on model
performance, we evaluate the performance of Qwen2.5-VL-
7B under different resolutions, with the number of frames
fixed at 32. The input video frames are set to 120p, 240p,
480p, 720p, and 960p.
## In Figure 7, the scores of nearly all abilities continue to im-
prove from 120p to 720p, while a slight decline is observed
when the resolution further increases to 960p. This suggests
that, within a certain range, higher resolution indeed en-
hances the model’s ability to capture visual details, but exces-
sive resolution may lead to a decline in performance, primar-
ily due to the increased number of input tokens that exceeds
the model’s optimal processing capacity.
5
Conclusion
In this paper, we presented MT-Video-Bench, a holistic benchmark for evaluating MLLMs in multi-turn
video dialogues. Unlike prior video understanding benchmarks that primarily focus on single-turn
factual perception, MT-Video-Bench jointly assesses perceptivity and interactivity through six carefully
defined capabilities, covering tasks such as memory recall, topic shifting, and proactive interaction. Our
evaluation of 20 state-of-the-art models provides insightful findings, and we hope our MT-Video-Bench
can establish a rigorous foundation for future research, highlighting the need for models that can reason
over long contexts while engaging in natural, adaptive conversations.
9
