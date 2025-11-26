---
source: "..\data\raw\arxiv\2510.17725v1.pdf"
arxiv_id: "2510.17725v1"
page: 7
total_pages: 13
date_converted: "2025-11-05"
---

...
## Chronological Split
NEW
NEW
NEW
1. Node Update
2. Node & Edge Update
3. Graph Pruning
NEW
NEW
NEW
Figure 2: Live Evaluation of AcademicEval Benchmark. To support continual benchmarking, Aca-
demicEval incrementally updates the co-author graph using daily arXiv data. The procedure includes: (1)
Node Update – augmenting node features for authors with newly published first-author papers; (2) Node
and Edge Update – identifying and prioritizing new co-authors via BFS to expand the graph with recent
publications; and (3) Graph Pruning – removing outdated papers and inactive authors to maintain graph
connectivity and efficiency.
in this task. Among several settings within each task, a higher compression rate makes it tougher to exploit
information holistically but more likely to produce better outputs (since more references are integrated).
These different tasks and settings increase the diversity of the AcademicEval benchmark.
As for data splitting, we perform a chronological split in AcademicEval, which means that the test set
always contains the latest papers collected in each collection round, ensuring no label leakage. Note that
Table 2 shows only the data collected in the initial round, which will be updated periodically as described
in the next section.
3.3
Live Evaluation with Periodic Data Updates on the Co-author Graph
The daily updates of arXiv provide the basis for the live evaluation of AcademicEval: we can periodically
update the benchmark with the latest papers on arXiv. By setting a reasonable update cycle (e.g., monthly
or quarterly), we can ensure that the data in the benchmark is not contaminated so that it can be used to
evaluate LLMs fairly in a live manner. Therefore, we proposed an efficient incremental update procedure on
the co-author graph:
(1) Node Update. For each author on the co-author graph, check whether the author has a newly published
first-author paper through the arXiv API. If so, add it to the corresponding node feature on the co-author
graph.
(2) Node and Edge Update. During the traversal of Node Update, each author’s new co-authors are
added to a candidate list, and the number of new papers (including first-author and non-first-author papers)
when searching for the author is used as the priority of the co-authors (co-authors of active authors tend
to be active as well, and we can efficiently collect the latest papers from active authors). Then, we use the
prioritized candidate list to conduct BFS to update nodes and edges until a specific number of incremental
update papers is met.
(3) Graph Pruning. As the benchmark is updated, we will remove some outdated papers and inactive
authors (defined as those who have not published new first-author or non-first-author papers for a long time)
from the co-author graph.
In this way, the latest papers can be obtained sufficiently and efficiently while ensuring connectivity and a
smaller graph size.
7
