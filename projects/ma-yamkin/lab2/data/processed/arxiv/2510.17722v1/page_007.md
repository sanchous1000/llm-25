---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 7
total_pages: 9
date_converted: "2025-11-05"
---

## Table 2: Evaluation results on MT-Video-Bench. OR: Object Reference. MR: Memory Recall. CS: Content
Summary. AR: Answer Refusal. TS: Topic Shifting. PI: Proactive Interaction. The best performance and
the second best performance are highlighted in green and blue, respectively.
## Models
Overall
Perceptivity
Interactivity
OR
MR
CS
AR
TS
PI
Closed-Sourced Models
Gemini 2.5 Pro (Team, 2025)
68.45
66.13
67.80
80.49
67.50
73.67
55.12
Gemini 2.5 Flash (Team, 2025)
63.30
63.44
63.41
73.48
64.32
68.12
47.04
Doubao-Seed-1.6-vision (Seed, 2025)
58.55
66.19
60.85
68.95
43.84
65.99
45.50
Open-Sourced Models
Model Size > 8B
Qwen2.5-VL-72B (Bai et al., 2025)
58.48
60.60
56.40
74.20
57.07
64.27
38.35
InternVL3.5-38B (Think) (Wang et al., 2025c)
58.11
60.87
60.36
69.90
46.86
65.17
45.51
Qwen2.5-VL-32B (Bai et al., 2025)
57.88
60.20
59.63
74.88
50.71
63.41
38.47
InternVL3.5-38B (No Think) (Wang et al., 2025c)
50.04
52.51
46.37
61.86
44.24
58.78
36.46
4B < Model Size ≤8B
InternVL3.5-8B (Think) (Wang et al., 2025c)
56.29
57.81
54.82
73.18
47.62
62.50
41.84
Qwen2.5-VL-7B (Bai et al., 2025)
53.12
56.18
49.99
67.21
52.20
57.20
35.92
InternVL3.5-8B (No Think) (Wang et al., 2025c)
49.35
51.71
46.95
61.50
40.83
57.23
37.85
LLaVA-Video-7B (Zhang et al., 2025b)
49.17
53.85
43.57
63.64
41.32
56.67
35.98
MiniCPM-o (Yao et al., 2024)
48.41
55.06
43.27
61.59
34.58
57.53
38.43
MiniCPM-V4.5 (Yao et al., 2024)
47.06
51.57
43.08
56.17
38.46
52.58
40.47
InternVideo2.5-8B (Wang et al., 2025e)
47.04
44.87
43.49
60.33
45.23
54.81
33.50
VideoLLaMA3-7B (Bai et al., 2025)
46.06
52.06
42.40
55.74
45.23
48.25
32.69
LLaVA-OneVision-7B (Li et al., 2024d)
45.75
50.01
43.36
59.34
32.79
55.44
33.56
VideoChat-Flash-7B (Li et al., 2024e)
41.11
47.92
39.33
51.14
28.02
48.27
32.01
LLaVA-NeXT-Video-7B (Zhang et al., 2024d)
38.04
43.05
36.04
48.58
27.60
42.94
30.00
Model Size ≤4B
InternVL3.5-4B (Think) (Wang et al., 2025c)
52.25
54.94
53.78
67.50
37.74
54.67
44.89
Qwen2.5-VL-3B (Bai et al., 2025)
48.07
50.64
43.54
65.82
46.80
50.33
31.30
InternVL3.5-4B (No Think) (Wang et al., 2025c)
45.90
46.03
46.19
61.30
30.41
55.72
35.74
models demonstrate competitive results in specific dimensions. For example, Qwen2.5-VL-72B shows
strong ability in MR, narrowing the gap with Gemini 2.5 Pro. However, on interaction-related subtasks
such as AR, the performance difference between open-source and closed-source models remains
substantial.
• Results vary significantly across different dimensions, and models generally perform better on
perception-related subtasks, where large-scale models generally achieve stronger scores, sometimes
exceeding 60. For example, the average score of OR is 54.55, while for PI is 38.60.
• Larger models tend to achieve higher accuracy. For instance, within the Qwen2.5-VL series, the 72B
and 32B models significantly outperform the 7B and 3B variants across nearly all subtasks. Similarly,
larger InternVL3.5 models achieve better results than their smaller counterparts. However, sometimes
small MLLMs can lead to higher scores. For instance, the AR scores for Qwen2.5-VL-7B, Qwen2.5-
VL-32B, and InternVideo2.5-8B are 52.20, 50.71, and 45.23, respectively. In addition, enabling thinking
mode within the same model variant leads to significant performance improvements, suggesting that
inference strategies, beyond model size, can substantially affect benchmark outcomes.
4.3
Further Analysis
4.3.1
Performance comparison between single scene and cross scene
Based on the selected three models in Figure 4, we summarize the following conclusions: (1) Across
almost all abilities, model performance under the cross-scene setting is worse than under the single-scene
setting. (2) Regardless of the setting, Gemini 2.5 Pro consistently outperforms Qwen2.5-VL-7B and
InternVL3.5-8B across all abilities, particularly in Content Summary and Memory Recall, while also
sustaining relatively high performance under the cross-scene condition. In comparison, InternVL3.5-8B
performs comparably to Gemini 2.5 Pro in the single-scene setting but suffers from substantial degradation
in the cross-scene setting. Meanwhile, Qwen2.5-VL-7B shows severe performance drops in Proactive
7
