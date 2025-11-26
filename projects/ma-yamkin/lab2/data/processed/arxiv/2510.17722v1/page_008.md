---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 8
total_pages: 9
date_converted: "2025-11-05"
---

(a) Qwen2.5-VL-7B
(c) Gemini 2.5 Pro
(b) InternVL3.5-8B (Think)
Figure 4: Performance comparison of Qwen2.5-VL-7B, InternVL3.5-8B (Think), and Gemini 2.5 Pro across
various tasks under single-scene and cross-scene settings.
Interaction and Memory Recall under cross-scene evaluation.
4.3.2
Performance of different video lengths
To study the impact of video length on model performance, videos are grouped into different length
ranges. From Figure 5 (a), we find that: (1) Model performance generally decreases as video length
increases, suggesting that longer videos pose greater challenges for capturing and reasoning over multi-
turn dialogue content. (2) Higher-capacity models, such as Gemini 2.5 Pro, tend to achieve higher overall
scores across all video lengths compared to smaller models like Qwen2.5VL-7B. However, all models
exhibit noticeable performance drops for very long videos. (3) The performance gap between models is
more pronounced for shorter videos, while for longer videos, the performance difference narrows.
(a) Performance of different video lengths
(c) Effect of dialogue context
(b) Performance across dialogue turns
Figure 5: Performance of different video lengths, dialogue turns, and settings of dialogue context.
4.3.3
Model performance across dialogue turns
To evaluate the impact of dialogue length on model performance, we conduct experiments with dialogues
of varying total turn numbers with Gemini-2.5-Pro and Qwen2.5VL-7B. Several key observations can be
drawn from the results shown in Figure 5 (b): Model performance tends to improve as the total number of
turns increases, although the degree and stability of this improvement vary across models. This suggests
that dialogue length plays a dual role in multi-turn video understanding: offering more contextual cues
beneficial for reasoning while increasing the burden of sustaining coherent dialogue states. One possible
reason for this pattern is that larger models are generally able to integrate contextual information more
efficiently, leveraging additional turns to further improve. Smaller models, on the other hand, tend to
rely more heavily on the accumulation of dialogue context across multiple turns.
4.3.4
Effect of dialogue context
To investigate the role of contextual information, we design three experimental settings:
Without Context. The model answers each question solely based on the video.
With Self-predicted Context. The model is provided with its own generated dialogue history.
With Golden Context. The model is provided with our meticulously curated golden dialogue history.
As shown in Figure 5 (c), the golden context yields the highest performance across all abilities. However,
8
