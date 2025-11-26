---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 3
total_pages: 13
date_converted: "2025-11-05"
---

pecially, AcademicEval conducts periodic data updates on the co-author graph to enable efficient
live evaluation, which ensures no label leakage and fair evaluation.
• We conduct comprehensive experiments on AcademicEval, and the results demonstrate its chal-
lenges and yield potential insights for improving LLMs in long-context modeling.
2
Related Work
Long-context Modeling and LLM Benchmarks. LLMs are known to be powerful in language modeling
tasks (Achiam et al., 2023; AI@Meta, 2024). However, when it comes to long-context inputs, LLMs show a
sharp decline in performance, posing a pressing challenge when benchmarking their long-context modeling
capabilities (Liu et al., 2024; Li et al., 2024; 2025). Currently, there are two mainstream technologies for
long-context modeling tasks: retrieval-augmented language models (RALM)(Ram et al., 2023; Yu et al.,
2023; Trivedi et al., 2022; Jiang et al., 2023; Asai et al., 2023; Zhang et al., 2024a; Feng et al., 2024) and
long-context LLMs (Bai et al., 2023a; Jiang et al., 2024; Teknium et al.).
RALM equips LLMs with a
retriever (Robertson et al., 2009; Ramos et al., 2003; Karpukhin et al., 2020; Izacard et al., 2021) to perform
information retrieval on short text chunks, which are then fed to LLMs together with the input query to
generate the final output. As a retrieval system, RALM is usually evaluated over retrieval-based benchmarks,
including STARK (Wu et al., 2024), RGB (Chen et al., 2024), ARES (Saad-Falcon et al., 2023), etc. In
comparison, long-context LLMs expand their context window length to accommodate longer inputs and are
benchmarked over various tasks, which include long-context QA, summarization, conversations, reasoning,
etc (Shaham et al., 2023; An et al., 2023; Dong et al., 2023; Bai et al., 2023b; Li et al., 2023; Zhang et al.,
2024b; Li et al., 2025).
## Recent works such as ResearchTown (Yu et al., 2024) and WildLong (Li et al., 2025) share conceptual
proximity to our setting but target different goals. ResearchTown is a multi-agent simulation framework
that models the dynamics of a research community via message-passing on an agent–data graph, simulating
activities such as paper and review writing. Its focus lies in simulating collaborative behavior and ensuring
the realism of outputs under controlled settings. In contrast, AcademicEval is a live, real-world benchmark
grounded in authentic academic papers, designed to evaluate LLMs on hierarchical writing tasks (Title,
Abstract, Introduction, and Related Work) under evolving and leakage-resistant conditions. While
both leverage graph structures, ResearchTown uses them for interaction simulation, whereas AcademicEval
employs a co-author graph for retrieving high-quality few-shot demonstrations, supporting scalable context
lengths, and enabling periodic data updates.
Similarly, WildLong introduces a scalable framework for synthesizing realistic long-context instruction data.
It extracts meta-information from user queries, builds co-occurrence graphs, and employs adaptive generation
to create 150K instruction–response pairs for complex multi-document reasoning tasks. While WildLong
focuses on data synthesis for instruction tuning, AcademicEval focuses on evaluation, providing a live,
automatically updated benchmark that measures LLMs’ long-context reasoning and generation abilities
on real-world academic tasks.
## Together, these works are complementary: ResearchTown and WildLong
contribute to synthetic data generation and simulation, whereas AcademicEval provides a robust evaluation
framework for real-world, graph-enabled long-context reasoning.
## Label Leakage in LLM Benchmarks. Label leakage has always been a severe issue that benchmarks
must attempt to avoid during data collection. However, recent research (Xu et al., 2024; Zhu et al., 2024b;a;
Ye et al., 2024) point out that most LLM benchmarks are composed of statically collected data, which may
be inevitably included in the large amount of training data of LLMs, causing label leakage.
## Therefore,
some works attempt to measure or detect the extent of label leakage in LLM benchmarks. Benbench (Xu
et al., 2024) leverages perplexity and N-gram accuracy to quantify potential label leakage, while PAC (Ye
et al., 2024) detects contaminated data by comparing the polarized distance of samples before and after
augmentation. Even though these approaches propose to measure or detect label leakage, there is little work
on mitigating and solving this issue (Zhu et al., 2024b). Dynabench (Kiela et al., 2021) and Dynaboard (Ma
et al., 2021) feature dynamic human-in-the-loop dataset creation while avoiding leakage, which is very labor-
intensive. DyVal (Zhu et al., 2024b) leverages pre-set constraints and directed acyclic graphs (DAG) to
dynamically generate test cases with diverse complexities, reducing the risk of label leakage. FreshBench (Zhu
3
