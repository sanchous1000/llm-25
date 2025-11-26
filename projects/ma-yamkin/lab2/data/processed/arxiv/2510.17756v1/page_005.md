---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 5
total_pages: 35
date_converted: "2025-11-05"
---

sea ice data.
## However, the fidelity of fully data-driven ML models is highly dependent
on the quantity and quality of training datasets, which leads to limitations in
further improving the model performance. First and foremost, data-driven
ML models require a large amount of data to ensure generalizability.
If
training samples are insufficient or their distribution is biased, the model can
be overfitted and lack generalizability to out-of-training cases. Second, ML
models are considered black-box models and lack inherent interpretability in
how the final predictions are derived. This lack of inherent interpretability
makes it difficult to understand why the model predicts physically invalid
values about sea ice conditions (e.g., negative SIC or SIC > 100%).
## In order to address such issues of the fully data-driven ML models and
improve the model’s generalizability, it is helpful to guide the learning pro-
cess to agree with fundamental physical laws and domain knowledge. By
integrating physics knowledge and constraints into ML training, the ML
models can yield physically consistent predictions even in the presence of
imperfect data, such as missing, noise, or outliers data (Karniadakis et al.,
2021; Raissi et al., 2019; Maier et al., 2023). This integration of ML and
physics knowledge is often referred to as physics-informed machine learning
(PIML), and has been proposed for various dynamic systems (Karniadakis
et al., 2021; Maier et al., 2023). Recently, many studies have attempted this
PIML framework for various applications including, but not limited to, flow
dynamics, material sciences, molecular simulations, chemistry (Karniadakis
et al., 2021). In particular, PIML based on well-known flow laws such as
Stokes equation has shown great success in modeling the dynamics of ice
5
