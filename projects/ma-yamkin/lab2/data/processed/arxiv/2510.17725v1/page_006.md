---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 6
total_pages: 13
date_converted: "2025-11-05"
---

## Table 2: Data Statistics of AcademicEval (Initial Round). It includes 4 writing tasks and provides
four settings of different context length for each task. For each setting, we list their Comp. Rate, Samples
of Each, Chronological Split, and Timespan of Test Data.
## Setting
Comp. Rate
(In-Len. / Out-Len.)
#Samples
of Each.
## Chronological Split
(Train-Val-Test)
Timespan
of Test Data
Title Writing
Title-10K
587
5098
72%-19%-9%
2024.06-
2024.07
Title-30K
1773
Title-31K-G
1807
Title-50K-M
2968
Abstract Writing
Abs-9K
36
5098
72%-19%-9%
2024.06-
2024.07
Abs-28K
108
Abs-29K-G
112
Abs-48K-M
185
Introduction Writing
Intro-8K
6
4665
71%-20%-9%
2024.06-
2024.07
Intro-28K
21
Intro-28K-G
22
Intro-48K-M
37
Related Work Writing
Related-34K
34
2240
72%-20%-8%
2024.06-
2024.07
Related-53K
53
Related-53K-G
53
Related-72K-M
72
Note: We use the BERT tokenizer by default to count the number of tokens.
settings: (1) Randomly select papers under the same category. According to the paper categories provided by
the arXiv API, we can randomly select several non-duplicate papers under the same category. (2) Randomly
Select co-author papers. The motivation is straightforward: the similarity of research directions between
co-author papers is more fine-grained. Thanks to the co-author graph, it is convenient to obtain the co-
author papers of each original paper sample. These selected papers serve as few-shot demonstrations and
are utilized as input-output pairs to enrich the input context of the original samples, providing potentially
insightful and relevant content while enabling flexible and scalable context length.
Consequently, we have completed the construction of benchmark settings, and the data statistics in the
initial collection round are shown in Table 2.
## Data Statistics. As shown in Table 2, AcademicEval has four academic writing tasks with hierarchical
abstraction levels, and each task features four settings with diverse input context lengths, some of which
are obtained by integrating few-shot demonstrations.
## For instance, each sample in Title-10K consists
of a single paper sample. Title-30K and Title-31K-G are obtained by integrating with two few-shot
demonstrations from random papers and co-author papers, respectively, while Title-50K-M is obtained by
using both of the above integration options. Actually, we can scale context length by increasing the number
of few-shot demonstrations to provide more informative references, enhancing task performance.
Furthermore, we present the text compression rate (defined as the number of input tokens divided by the
number of output tokens) for each benchmark setting in Table 2 to illustrate the diverse abstraction levels in
AcademicEval. Across the four tasks, a higher compression rate means a higher level of text abstraction
6
