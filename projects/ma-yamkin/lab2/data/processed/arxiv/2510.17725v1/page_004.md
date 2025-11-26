---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 4
total_pages: 13
date_converted: "2025-11-05"
---

et al., 2024a) and StackMIA (Ye et al., 2024) collect the latest data from public websites periodically and
simply rely on the chronological split to build a dynamic benchmark.
## Long-context Summarization Benchmarks. Solving AcademicEval requires LLM’s long-context sum-
marization capability (Liu et al., 2024). Existing works include (1) query-based summarization tasks, focus-
ing on the capability of models to position and capture local key information in long texts given a specific
query (Litvak & Vanetik, 2017; Wang et al., 2022); (2) single-document or multi-document summarization
tasks concentrate on evaluating the ability of models to understand long texts holistically (Cohan et al.,
2018; Meng et al., 2021; Huang et al., 2021; Kryściński et al., 2021; Cachola et al., 2020). These long-context
summarization benchmarks suffer from the above-mentioned limitations, including requiring human-assisted
labeling and concerns about data leakage; moreover, these summarization tasks focus on one-level summa-
rization, failing to consider the summarizations at different abstraction levels.
3
AcademicEval Benchmark
In this section, we propose AcademicEval (Figure 1) for live evaluation over long-context generation tasks
with hierarchical abstraction levels.
We first describe data collection and preprocessing in Section 3.1.
Then, in Section 3.2, four academic writing tasks with diverse abstraction levels are introduced, and we
also integrate few-shot demonstrations to make the context length flexible and scalable. Finally, Section 3.3
elucidates the live evaluation with periodic data updates.
3.1
Data Curation
Co-author Graph Construction via arXiv. As a public paper preprint platform, arXiv1 has always been
favored by researchers. It archives a huge amount of papers and updates the latest ones daily, which serves
as an excellent data source and also lays the foundation for the live update of our benchmark. Thanks to the
arXiv API2, paper files can be obtained in batch without much manual effort. We first collect and construct
a co-author graph (i.e., edges are established between two co-author nodes) using the arXiv API through
breadth-first search (BFS), where the features of each author node include the published first-author papers.
By making the co-author graph the carrier of papers, we can form an interconnected whole of scattered
articles, which provides valuable structural information to be exploited for our benchmark. Furthermore, we
can enable efficient live updates on the co-author graph, which will be introduced in Section 3.3.
Academic Data Gathering and Preprocessing.
## After the co-author graph is collected, we remove
authors who have not published independent first-author papers (i.e., appear only as co-authors in the author
list) and then prune it to obtain the maximum connected component. For each paper (i.e., node features),
we collect essential metadata via the arXiv API, including author information, publication timestamp, etc.,
and download the PDF file simultaneously, which further goes through a series of pipelines to split and
extract the text of several sections in it. In detail, we leverage PyMuPDF3 to detect section headings (e.g.,
"Introduction") and extract the paper content by sections. Especially for the "Related Work" section, we
extract each cited paper’s abstract and title via the arXiv API to form an additional citation corpus. All
these processed data constitute the node feature of each author node. We will further describe in Section 3.2
how to use these data to design long-context academic writing tasks.
3.2
Benchmarking LLMs over Long-context Generation Tasks with Hierarchical Abstraction
Task Description. Employing machine learning approaches to automate academic writing has always been
a research hotspot with significant practical application value (Chen et al., 2022; 2021). Therefore, inspired
by the leave-one-out validation, we introduce four academic writing tasks with ultra-long context to evaluate
the generation capability of LLMs under different abstraction levels, as shown below:
1https://arxiv.org/
2https://info.arxiv.org/help/api/index.html
3https://github.com/pymupdf/PyMuPDF
4
