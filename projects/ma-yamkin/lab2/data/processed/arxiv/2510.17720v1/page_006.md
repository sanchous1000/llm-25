---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 6
total_pages: 8
date_converted: "2025-11-05"
---

TABLE II
FEW-SHOT F1 (%) SCORES USING AUGMENTED SAMPLES ACROSS DIFFERENT DOMAINS
Model Family
of
Original
Samples
of
Augmented
Samples
Movie
Restaurant
AI
Literature
Music
Politics
Science
Average
LLAMA-3.1-8B-Instruct
0
0
43.3
30.3
59.4
61.5
68.2
62.1
62.3
55.3
100
0
45.1
35.4
61.0
67.2
75.9
70.4
70.9
60.8
100
200
64.2
39.8
64.0
77.0
80.8
73.2
72.9
67.4
Qwen-2.5-7B-Instruct
0
0
50.3
33.6
46.5
54.9
53.9
54.0
49.1
48.9
100
0
54.2
25.6
57.8
63.4
70.4
64.2
65.9
57.4
100
200
65.1
38.2
59.5
73.3
79.2
70.2
75.3
65.8
Falcon-3-10B-Instruct
0
0
64.5
38.1
63.7
58.8
68.6
61.8
59.4
59.3
100
0
63.7
37.0
67.8
67.6
82.2
72.0
77.5
66.8
100
200
77.5
42.8
72.7
79.0
85.3
81.3
82.3
74.4
TABLE III
COMPARISON OF F1 (%) SCORES ON CROSSNER FOR SUPERVISED
TECHNIQUES
AI
Lit
Music
Pol
Sci
Avg
BERT
68.7
64.9
68.3
63.6
58.8
64.9
CDLM
68.4
64.3
63.5
59.5
53.7
61.9
DAPT
72.0
68.8
75.7
69.0
62.6
69.6
NER-BERT
76.1
72.1
80.2
71.9
63.3
72.7
PANER (Ours)
72.7
79
85.3
81.3
82.3
80.1
TABLE IV
COMPARISON OF F1 (%) SCORES ON CROSSNER FOR AUGMENTATION
COMPOSITION WITH LLAMA - 3.1-8B-INSTRUCT
AI
Lit
Music
Pol
Sci
Avg
100 OG + 200 dup
60.8
67.5
72.7
68.4
64.9
66.8
100 OG + 200 aug
63.9
77.1
80.2
71.9
72.7
73.2
300 OG
67.4
79.0
80.1
73.3
76.5
75.3
is observed when definitions and guidelines are included
alongside the new format. While this increase may appear
marginal, it offers substantial benefits when integrated with our
paraphrase-augmented synthetic data during few-shot testing
(as shown in Table I). This improvement underscores the
complementary nature of clear guidelines and the word/tag
format in enhancing model accuracy and adaptability.
B. Performance of Paraphrasing in Few-shot NER
We further evaluate the effectiveness of the paraphrasing-
based data augmentation approach across multiple domain-
specific datasets, including CrossNER [12] and MIT [13]. The
three models listed in Section V-C were fine-tuned using 0,
100, and 300 augmented samples with a fixed set of 10,000
PileNER [3] samples. This dataset size was chosen to align
with the GNER [5] framework.
## As illustrated in Figure 4, there is an increase in F1
scores correlating with the increase in augmented samples.
## The results of this experiment are presented in Table II,
where we observe that the average F1 score of Falcon 3
[8] increased by 0.14, Qwen 2.5 [6] by 0.17 and LLAMA
TABLE V
PERFORMANCE COMPARISON OF DIFFERENT AUGMENTATION METHODS
ON ENGLISH (EN)
#Gold
Method
F1 Score (in %)
100
Gold-Only
50.57
Label-wise
61.34
MLM-Entity
61.22
DAGA
68.06
MELM
75.21
PANER (Ours)
80.52
200
Gold-Only
74.64
Label-wise
76.82
MLM-Entity
79.16
DAGA
79.11
MELM
82.91
PANER (Ours)
85.74
400
Gold-Only
81.85
Label-wise
84.62
MLM-Entity
83.82
DAGA
84.36
MELM
85.73
PANER (Ours)
88.11
3 [7] by 0.12, respectively. The average performance across
the CrossNER [12] datasets is illustrated in Figure 4. This
indicates that the diversity introduced through the paraphrasing
strategy effectively contributes to model performance.
## In particular, the result using Falcon-3 [8] with 300 aug-
mented samples achieved superior performance compared to
previously reported supervised techniques on the CrossNER
[12] datasets, further reinforcing our augmentation strategy.
## Although many recent IT studies that report on this dataset
focus on zero-shot performance metrics (reported below), the
results surpass few-shot and fine-tuned in which CrossNER
[12], which utilized supervised training samples. These com-
parative results, detailed in Table III, demonstrate that our data
augmentation technique can enhance model performance for
cases with few examples. Although the performance of other
Instruction-tuned models with similar augmentation strate-
gies remains unexplored, the results suggest that combining
lightweight few-shot learning with intelligent data augmen-
tation offers a promising direction for domain-specific NER
tasks.
