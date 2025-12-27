---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 6
total_pages: 9
date_converted: "2025-11-05"
---

(a) Video Categories
(b) Task Distribution
(c) Video Duration Distribution
(d) Dialogue Turn Distribution
Figure 3: Overview of MT-Video-Bench. (a) Video Categories. MT-Video-Bench includes videos spanning
5 major categories, ensuring diverse topical coverage. (b) Task Distribution. MT-Video-Bench consists
of a total of 6 tasks with a relatively balanced distribution. (c) Video Duration Distribution. MT-Video-
Bench includes both long and short videos. (d) Dialogue Turn Distribution. Multi-turn dialogues in
MT-Video-Bench involve 5 to 8 rounds.
3.6
Evaluation Method
In multi-turn dialogues, each new turn depends on the interactions between users and assistants in
previous turns. This dynamic is particularly crucial in tasks that involve high interactivity, such as
proactive interactions. Therefore, we follow the multi-turn dialogue evaluation setup used in LLMs (Bai
et al., 2024), leveraging our meticulously curated dataset as the golden context for dialogue history, rather
than relying on self-predicted context from MLLMs.
For evaluation, we first use Gemini 2.5 Flash (Team, 2025) to construct a checklist for each QA pair.
Specifically, each checklist consists of five yes/no questions designed to assess the accuracy of the model’s
responses and its performance on specific tasks. Then, manual validation is employed to filter out
unqualified checklists. After filtering, each QA pair has an average of 3.29 questions in the final checklists,
with 62.35% answered as yes and 37.65% as no. During the evaluation process, Gemini 2.5 Flash (Team,
2025) is used to answer each checklist question based on the model-generated answers. The evaluation
metric is calculated as the accuracy (ACC), based on the proportion of correct answers across all checklists.
4
Experiments
4.1
Experimental Settings
For closed-source models, we evaluate Gemini 2.5 Pro (Team, 2025), Gemini 2.5 Flash (Team, 2025),
and Doubao-Seed-1.6-vision (Seed, 2025). For open-source models, we select 18 representative MLLMs,
including Qwen2.5 VL series (Bai et al., 2025), InternVL3.5 series (Wang et al., 2025c), LLaVA-Onevision
series (Li et al., 2024b), InterVideo2.5 series (Wang et al., 2025d), LLaVA-Video series (Zhang et al., 2024b),
LLaVA-NeXT-Video series (Zhang et al., 2024c), VideoChat-Flash series (Li et al., 2024c), VideoLlama3
series (Zhang et al., 2025a) and MiniCPM series (Yao et al., 2024).
Evaluation. For each model, we adopt a uniform sampling strategy to process video frames, setting the
number of frames to 32. Each video is resized that the longer side is limited to 720 pixels and the other
side is scaled proportionally. More details are described in Appendix B.1. For the prompts, we provide
the evaluation prompts of six tasks of MT-Video-Bench in B.2.
4.2
Main Results
As shown in Table 2, we provide the performance results of different MLLMs on our MT-Video-Bench,
and we have the following insightful and interesting observations:
• MT-Video-Bench is very challenging. Even the best-performing closed-source model, Gemini 2.5 Pro,
only achieves 68.45% overall accuracy, which is inferior to the performance of human experts a lot.
• Among all evaluated models, Gemini 2.5 Pro consistently ranks first in both overall accuracy and every
individual subtask. While closed-source systems still dominate overall performance, some open-source
6
