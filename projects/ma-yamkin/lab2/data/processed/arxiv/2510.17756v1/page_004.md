---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 4
total_pages: 35
date_converted: "2025-11-05"
---

traditional numerical models.
ML approaches have several benefits com-
pared to numerical models. First, complex PDEs or parameterizations are
not required for ML models because they need only observational data to be
trained. ML models can learn complex non-linear patterns or relationships
between multiple atmospheric and oceanic variables directly from observa-
tional data without any explicit knowledge of physical processes. Second,
ML models can be easily updated and trained continuously with new ob-
servational data, and this data integration can make the model prediction
more accurate and reliable. Finally, ML models require less computational
cost than numerical models. While numerical models often require intensive
computing resources (e.g., high-performance computing) to solve complex
PDEs, ML models can be implemented relatively fast by leveraging the par-
allel processing ability of graph processing units (GPUs).
## Based on such advantages, various machine learning models have been
developed to predict sea ice concentration (SIC) and sea ice velocity (SIV),
most of which rely on convolutional neural networks (CNN) (Ren and Li,
2021; Liu et al., 2021; Yan and Huang, 2018; Hoffman et al., 2023; Petrou
and Tian, 2019). As a deep learning algorithm specialized for image data,
CNN propagates the input data through convolutional layers with multiple
filters and extracts spatial patterns and features. Based on the ability of
CNN to learn complex features in image datasets, CNNs have succeeded in
various applications in environmental modeling (Wang et al., 2024; Jiang
et al., 2023; Sadeghi et al., 2020). Since the SIC and SIV data for the entire
Arctic Ocean are often provided from remote sensing data sources as grid
formats, CNN has advantages in exploiting the spatial variability in those
4
