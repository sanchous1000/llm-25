---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 2
total_pages: 9
date_converted: "2025-11-05"
---

What happens in the first ...? Describe the sequence...
In their cave, Grizzly picks up an empty cardboard  ...
... What is the single most effective way to prevent
the spread of infectious diseases...?
According to public health organizations ...
## Returning to the video, how does Panda's bubble ...?
The bubble proves very effective. In the garbage ...
...Speaking of birds, what is the 'V' formation ...?
The 'V' formation is a specific flight pattern used by...
...a new character appears. Who is this character...?
The character is Charlie, a tall, gray, Bigfoot-like ...
## What is the reason Charlie is so interested in ...?
... because he sees it as a way to protect himself
from ...
## Topic Shifting
...
...
## Single Scene
... appears. What is his apparent profession, and ...?
The man in the formal black tuxedo is a waiter. In ...
## According to the ..., what is the name of the cafe...?
The managing director, Berndt Querfeld, states that ...
## We see the same waiter... What is he doing during...?
Later in the video ..., the waiter is seen serving ...
A customer ... provided by people like him. What ... ?
A female customer states that the service provided ...
## Considering that ..., why is his formal attire and ...?
... because they embody the long-standing,
traditional ...
## What visual evidence ... the tradition he is a part of?
The video presents historical evidence through a ...
## Object Reference
...
...
## Cross Scene
(Topic Shifting)
(Topic Shifting)
(Object Reference)
(Object Reference)
Figure 1: Illustration of multi-turn dialogues under single-scene and cross-scene settings. The evaluated
questions corresponding to tasks are marked with underlining, and the scenes involved in the entire
multi-turn dialogues are marked with blue dotted boxes.
## Moreover, unlike prior datasets, MT-Video-Bench emphasizes cross-scene reasoning, long-range depen-
dencies, and interactive adaptability, thereby aligning closely with real-world application demands.
## Based on our MT-Video-Bench, we provide a detailed evaluation of both open-source and closed-
source models, highlighting the current limitations and performance discrepancies in different abilities.
## Specifically, several insightful findings are as follows:
• The perceptual and interactive capabilities of MLLMs in multi-turn dialogues still have significant
room for improvement. On MT-Video-Bench, even the strongest closed-source model Gemini 2.5 Pro
achieves only 68.45% overall accuracy, while most open-sourced MLLMs exhibit accuracies below 50%,
except for the Qwen2.5-VL and InternVL3.5 series.
• Performance is imbalanced across different tasks and scene types. MLLMs generally perform better on
perceptual subtasks (e.g., Object Reference) than on interactive ones (e.g., Proactive Interaction), with a
substantial gap between closed- and open-source models. Moreover, all models tend to perform worse
in cross-scene settings compared to single-scene tasks.
• Model scaling is beneficial but not sufficient. Larger models consistently outperform smaller counter-
parts across most subtasks, yet scaling alone does not ensure consistent improvements. For example,
in the InternVL 3.5 series, enabling the Thinking mode allows smaller models to achieve performance
comparable to that of larger models, which demonstrates the significant benefit of the reasoning
process in enhancing model performance.
To summarize, the contributions of this paper are as follows: We identify the critical gap in evaluating
multi-turn video-grounded dialogues and propose the MT-Video-Bench, the first holistic benchmark that
operationalizes this evaluation via six well-defined capabilities across 987 dialogues and 5,805 QA pairs.
## Then, based on extensive experiments on MT-Video-Bench, we underscore the challenges and potential
directions for improvement of handling and reasoning over multi-turn dialogues, offering a roadmap for
future research and development.
2
Related Work
Multimodal LLMs. MLLMs have become a central research focus in advancing general-purpose intelli-
gence. By jointly modeling textual and visual modalities, these models are able to capture cross-modal
dependencies and enhance semantic reasoning (Zhu et al., 2023; Ma et al., 2024; Zhang et al., 2024a; Wang
et al., 2025b; 2024c). Recent advances have further extended MLLMs to the video domain, enabling
2
