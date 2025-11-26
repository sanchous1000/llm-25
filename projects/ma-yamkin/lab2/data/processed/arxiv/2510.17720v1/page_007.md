---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 7
total_pages: 8
date_converted: "2025-11-05"
---

TABLE VI
COMPARISON OF ZERO-SHOT LEARNING PERFORMANCE F1 (%) SCORES
Model
Backbone
#Params
Movie
Restaurant
AI
Literature
Music
Politics
Science
Average
ChatGPT
gpt-3.5-turbo
-
5.3
32.8
52.4
39.8
66.6
68.5
67
47.5
InstructUIE
Flan-T5-xxl
11B
63
21
49
47.2
53.2
48.2
49.3
47.3
UniNER-type+sup.
LLAMA-1
7B
61.2
35.2
62.9
64.9
70.6
66.9
70.8
61.8
GoLLIE
Code-LLAMA
7B
63
43.4
59.1
62.7
67.8
57.2
55.5
58.4
GLiNER-L
DeBERTa-v3
0.3B
57.2
42.9
57.2
64.4
69.6
72.6
62.6
60.9
GNER-T5
Flan-T5-xxl
11B
62.5
51
68.2
68.7
81.2
75.1
76.7
69.1
GNER-LLAMA
LLAMA-1
7B
68.6
47.5
63.1
68.2
75.7
69.4
69.9
66.1
SLIMER
LLAMA-2-chat
7B
50.9
38.2
50.1
58.7
60
63.9
56.3
54
PANER
Qwen-2.5-Instruct
7B
51.5
37.3
62
61.7
75.9
69.72
65.63
60.5
PANER
LLAMA-3.1-Instruct
8B
52
37
63.9
67.2
75.3
67.8
68.7
61.7
PANER
Falcon3-Instruct
10B
69.4
43.3
65.5
61.3
75.8
70.3
68.3
64.8
TABLE VII
ZERO-SHOT RESULT COMPARISON ON BUSTER DATASET F1 (%) SCORES
Model
Backbone
#Params
Pr.
R
F1
GNER-LLAMA
LLAMA-1
7B
14.68
59.97
23.58
GLINER-L
DeBERTa-v3
0.3B
42.55
19.31
26.57
GoLLIE
Code-LLAMA
7B
28.82
26.63
27.68
GNER-T5
Flan-T5-xxl
11B
19.31
50.15
27.88
UniNER-type+sup.
LLAMA-1
7B
31.4
47.53
37.82
SLIMER
LLAMA-2-chat
7B
47.69
43.09
45.27
PANER
Falcon3-Instruct
10B
29.92
38.38
33.63
Additionally, we compared our paraphrasing-based data
augmentation approach against existing paraphrasing tech-
niques, including DAGA [27] and MELM [28], on the CoNLL
[26] shared task dataset. Our results indicate that leveraging
LLMs for paraphrasing yields superior performance compared
to these established techniques, as shown in Table V.
## For this experiment, we simulated a low-resource scenario
for the CoNLL dataset by using 100, 200, and 400 gold
samples, following the setup of [28], and then generating
200, 400, and 800 augmented samples, respectively. We then
trained the LLAMA 3.1-8B model using these configurations.
## Our approach consistently outperforms MELM, which utilizes
3× the number of samples compared to the 2× used in our
method. This performance gain can be attributed to our use
of LLMs for prediction, as they are already familiar with the
entity types in the CoNLL dataset (PERSON, LOCATION,
ORGANIZATION).
C. Effectiveness of Paraphrase-Based Augmentation Com-
pared to Data Duplication and In-Domain Expansion
In order to understand the isolated effects of our paraphras-
ing method against duplication. We tested the comparative
effectiveness of our data augmentation approach versus simply
adding more in-domain samples or duplicating existing data,
we conducted an additional experiment to isolate the impact
of our paraphrasing-based augmentation strategy. This exper-
iment systematically compared three training configurations,
each with a total of 300 samples but differing in composition:
1) 100 original in-domain samples augmented with 200
paraphrased variants,
2) 300 distinct original in-domain samples, and
3) 100 original in-domain samples duplicated two times.
## The results, presented in Table IV, demonstrate several
key findings. The configuration using 300 distinct original
samples achieved the highest average F1 score (75.3%), which
was expected given the inherent value of diverse, authen-
tic samples. However, our hybrid approach combining 100
original samples with 200 paraphrased variants performed
remarkably well, reaching an F1 score of (73.2%), only 2.1
percentage points below the all-original configuration. This
suggests that our paraphrasing strategy successfully preserves
the essential entity relationships while introducing beneficial
linguistic variation.
## In contrast, the simple duplication approach yielded sub-
stantially lower performance (66.8%), confirming that mere
repetition of training examples provides no meaningful diver-
sity to enhance model generalization. These findings validate
our augmentation approach as an effective strategy when
additional authentic in-domain samples are unavailable or
prohibitively expensive to obtain, offering nearly comparable
performance to training with three times the amount of original
data.
## Overall, these results demonstrate the effectiveness of LLM-
generated paraphrases in enhancing model generalization for
NER tasks, further validating our approach as a viable alter-
native to conventional data augmentation strategies.
