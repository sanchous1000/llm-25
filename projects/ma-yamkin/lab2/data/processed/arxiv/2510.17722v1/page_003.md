---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 3
total_pages: 9
date_converted: "2025-11-05"
---

Table 1: Comparison with other benchmarks. Avg. Q/V: the average number of QA pairs per video.
## Long: whether the average video length is greater than 10 minutes. Cross-Scene: whether the dialogue
covers more than 4 scenes.
## Benchmark
#QAs
Avg. Q/V
Long
Dialogue
#Turns
Cross-Scene
Annotation
MVBench (Li et al., 2024a)
4,000
1.00
%
%
1.00
-
Auto
LongVideoBench (Wu et al., 2024a)
6,678
1.77
%
%
1.00
-
Manual
Video-MME (Fu et al., 2025)
2,700
3.00
"
%
1.00
-
Manual
LVBENCH (Wang et al., 2024b)
1,549
15.04
"
%
1.00
-
Manual
MLVU (Zhou et al., 2025)
3,102
1.79
"
%
1.00
-
Manual
Video-MMLU (Song et al., 2025)
15,746
14.78
%
%
1.00
-
Auto&Manual
ScaleLong (Ma et al., 2025)
1,747
6.49
"
%
1.00
-
Manual
SVBench (Yang et al., 2025)
7,374
36.87
%
"
4.29
%
Auto&Manual
MT-Video-Bench (Ours)
5,805
43.00
"
"
5.88
"
Auto&Manual
video understanding, which subsequently supports dialogue(Li et al., 2023; Cheng et al., 2024; Maaz
et al., 2023). For example, Qwen2.5-VL (Bai et al., 2025) employs a dynamic-resolution Vision Trans-
former with MRoPE for spatiotemporal alignment, and connects an MLP merger to the Qwen2.5 LLM
decoder. InternVL3.5 (Wang et al., 2025a) integrates InternViT as the vision encoder with a ViT-MLP-LLM
paradigm, and further adopts Visual Resolution Router (ViR) with Visual Consistency Learning (ViCO)
for cross-modal alignment.
## Video Benchmarks. Significant developments have also been made in video understanding bench-
marks(Wang et al., 2023; Wu et al., 2024b; Xiao et al., 2021; Li et al., 2025). For example, MVBench (Li
et al., 2024a) focuses on concise video QA tasks to evaluate multimodal understanding abilities, while
MLVU (Zhou et al., 2025) and LVBENCH (Wang et al., 2024b) provide a comprehensive analysis for
MLLMs’ long-video understanding performance. MMBench-Video (Fang et al., 2024) is a long-form,
multi-shot benchmark that evaluates fine-grained abilities of MLLMs, including temporal reasoning,
perception, and general reasoning in video understanding. SVBench (Yang et al., 2025) is a benchmark
for temporal multi-turn dialogues in streaming videos, designed to assess the capabilities of streaming
video understanding of MLLMs. However, prior benchmarks primarily focus on evaluating the video
understanding capabilities of models, overlooking the multi-turn dialogue capabilities, which require not
only the ability to recall contextual information but also to engage in coherent, interactive communication
with users across multiple turns.
3
MT-Video-Bench
3.1
Overview
MT-Video-Bench is designed to comprehensively evaluate the “Perceptivity” and “Interactivity” of
MLLMs in multi-turn video-grounded dialogues. Different from conventional video understanding
benchmarks that primarily focus on single-turn question answering, MT-Video-Bench is specifically
designed to mimic real-world interactive scenarios, emphasizing contextual coherence, cross-scene video
comprehension, and adaptive interactivity.
MT-Video-Bench systematically evaluates six core capabilities of MLLMs through 987 meticulously
curated multi-turn dialogues with 5,805 QA pairs. Each conversation requires not only accurate video
perception but also contextual reasoning within or across video scenes, with representative examples
shown in Figure 1.
A comprehensive comparison between our MT-Video-Bench and other related benchmarks is provided
in Table 1. MT-Video-Bench presents the following critical values: (1) supports multi-turn dialogues
that evaluate contextual coherence and long-range dependency, (2) supports cross-scene reasoning that
requires integrating information across different video clips, and (3) provides a fine-grained assessment
of perceptivity and interactivity through six tasks.
3.2
Evaluation Tasks
Perceptivity assesses the model’s foundational ability to perceive and integrate information from both the
visual video content and the multi-turn conversational context. This capability is essential for accurately
understanding user queries and generating contextually grounded responses throughout the dialogue. It
3
