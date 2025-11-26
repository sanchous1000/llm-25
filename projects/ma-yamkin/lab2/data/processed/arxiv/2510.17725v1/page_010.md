---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 10
total_pages: 13
date_converted: "2025-11-05"
---

Table 4: Main Results on AcademicEval w.r.t. RougeL.
## Models
Standard LLMs
Long-context LLMs
RALM
Gemma
LLaMA
Qwen
Mixtral
Hermes
Gemma†
LLaMA†
#Params.
7B
70B
72B
8x7B
8x7B
7B
70B
Context Length
8K
8K
32K
32K
32K
8K
8K
Setting: Title Writing
Title-10K
44.5
47.1
44.2
45.2
46.2
42.7
47.3
Title-30K
-
-
44.5
44.6
45.9
42.6
47.3
Title-31K-G
-
-
44.2
44.4
45.3
42.5
47.0
Setting: Abstract Writing
Abs-9K
22.4
25.0
24.3
24.1
26.1
23.4
24.2
Abs-28K
-
-
23.3
24.7
26.6
23.1
24.1
Abs-29K-G
-
-
23.3
24.9
26.6
23.2
24.0
Setting: Introduction Writing
Intro-8K
14.9
18.1
16.2
17.2
17.8
15.4
17.9
Intro-28K
-
-
16.3
17.5
17.5
15.3
17.8
Intro-28K-G
-
-
16.3
17.5
17.5
15.4
17.8
Setting: Related Work Writing
Related-34K
13.5
14.9
16.0
13.4
15.1
14.1
15.3
Related-53K
-
-
-
-
-
14.0
15.3
Related-53K-G
-
-
-
-
-
14.0
15.2
Bold indicates the highest score in each row.
† denotes augmentation with a retriever (Default: Contriever).
“-” means that the context length is too long to be fed into LLMs.
## For example, the Title Writing task tends to have a higher score than the Abstract Writing task,
which may indicate that the Title Writing task is easier than the Abstract Writing task. Since a
title only has a few words, LLMs only need to generate a roughly related theme to achieve a high semantic
similarity, while an abstract requires a more detailed description to achieve it.
Baseline Performance Comparison.
## Across automatic metrics, RALM with LLaMA frequently at-
tains the highest scores in multiple settings (e.g., Title-30K/31K-G, Intro-28K/28K-G, and Related-
53K/53K-G), despite using an 8K input window. Standard LLMs remain competitive and long-context
LLMs (e.g., Qwen, Hermes) lead in some settings (e.g., Related-34K on BERTScore).
## This exposes
the shortcomings of long-context LLMs’ generation capabilities, which are well revealed by AcademicE-
val. Among long-context LLMs, Hermes performs best overall, but is still slightly inferior to RALM with
LLaMA. This shows that although the current long-context LLMs have a longer context window size, they
still have great deficiencies in processing long text information. Overall, RALM often has an edge under au-
tomatic metrics, likely because retrieval concentrates salient content into shorter chunks, thereby maximizing
overlap-oriented scores.
Impact of Context Length.
## The impact of context length on performance is evident across all task
settings and both metrics, with baselines often performing worse as the context length increases, though
the extent is model- and task-dependent. For example, the Title Writing task shows a noticeable drop
in scores as the context length extends from 10K to 31K tokens. This trend is also apparent in Abstract
Writing and Introduction Writing, where longer contexts correlate with decreased model performance,
showing that our benchmark challenges LLMs in effectively processing ultra-long inputs.
10
