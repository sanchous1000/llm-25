---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 1
total_pages: 9
date_converted: "2025-11-05"
---

2025-10-21
MT-Video-Bench: A Holistic Video Understanding
Benchmark for Evaluating Multimodal LLMs in
Multi-Turn Dialogues
Yaning Pan1, Zekun Wang2, Qianqian Xie3, Yongqian Wen3, Yuanxing Zhang2,
Guohui Zhang4, Haoxuan Hu3, Zhiyu Pan3, Yibing Huang3, Zhidong Gan3,
Yonghong Lin3, An Ping3, Tianhao Peng3, Jiaheng Liu3,†
1Fudan University,
2Kuaishou Technology,
3Nanjing University,
4University of Science and Technology of China
ynpan24@m.fudan.edu.cn
liujiaheng@nju.edu.cn
Abstract
The recent development of Multimodal Large Language Models (MLLMs) has significantly ad-
vanced AI’s ability to understand visual modalities. However, existing evaluation benchmarks
remain limited to single-turn question answering, overlooking the complexity of multi-turn di-
alogues in real-world scenarios. To bridge this gap, we introduce MT-Video-Bencha, a holistic
video understanding benchmark for evaluating MLLMs in multi-turn dialogues. Specifically, our
MT-Video-Bench mainly assesses six core competencies that focus on perceptivity and interactivity,
encompassing 987 meticulously curated multi-turn dialogues from diverse domains. These capa-
bilities are rigorously aligned with real-world applications, such as interactive sports analysis and
multi-turn video-based intelligent tutoring. With MT-Video-Bench, we extensively evaluate various
state-of-the-art open-source and closed-source MLLMs, revealing their significant performance
discrepancies and limitations in handling multi-turn video dialogues. The benchmark will be
publicly available to foster future research.
ahttps://github.com/NJU-LINK/MT-Video-Bench
1
Introduction
The rapid progress of Multimodal Large Language Models (MLLMs) has markedly advanced AI’s
capacity to perceive and reason over visual modalities, especially when integrated with natural language.
## Recent systems such as Qwen2.5-VL (Bai et al., 2025), InternVL3.5 (Wang et al., 2025a), and Gemini
2.5 (Team, 2025) demonstrate impressive performance in single-turn video question answering and
long-form video comprehension (Zhang et al., 2023; Rawal et al., 2024; Sun et al., 2022; Wang et al., 2024a;
Chandrasegaran et al., 2024). Yet, real-world human–AI interaction is rarely confined to single-turn
queries. Instead, it typically unfolds as multi-turn dialogues, where users iteratively refine their questions,
shift topics, and expect contextually coherent responses grounded in video content. This interactive
setting poses unique challenges: models must not only recall and integrate prior dialogue history but also
adapt to conversational dynamics, such as handling topic shifting or gracefully refusing unanswerable
queries.
## Despite these demands, existing video understanding benchmarks (Fu et al., 2025; Wang et al., 2024b;
Zhou et al., 2025; Ma et al., 2025) predominantly focus on single-turn evaluation, emphasizing factual
perception of video content—such as recognizing objects, actions, or temporal relations—while neglecting
dialogue-level reasoning. A few recent efforts explore long-context or multi-shot video benchmarks, yet
they fall short of capturing the interplay between perceptivity (faithfully interpreting multimodal input)
and interactivity (sustaining natural, user-aware conversations). Consequently, the community lacks
a rigorous and holistic framework to measure how well MLLMs can operate in realistic multi-turn,
video-grounded dialogues.
To fill this gap, as shown in Figure 2, we introduce MT-Video-Bench, a holistic benchmark for evaluat-
ing MLLMs in multi-turn video dialogue. MT-Video-Bench systematically targets six core capabilities
spanning perceptivity (object reference, memory recall, and content summary) and interactivity (an-
swer refusal, topic shifting, and proactive interaction). The benchmark comprises 987 carefully curated
dialogues across 135 videos, covering diverse domains such as sports, education, and daily activities.
† Corresponding Author.
1
arXiv:2510.17722v1  [cs.CV]  20 Oct 2025
