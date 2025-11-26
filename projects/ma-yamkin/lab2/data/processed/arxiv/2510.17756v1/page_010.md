---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 10
total_pages: 35
date_converted: "2025-11-05"
---

underlying physics laws can accurately represent these physics laws. Given
that enough high-quality and labeled data is not always available for real-
world tasks, synthetic data constructed with physical simulations can provide
a large amount of data with high quality. However, this approach can have
limitations in perfectly reflecting real world because real-world data have
different distributions from simulation data. In addition, the cost of data
acquisition can be a critical issue, as simulation data is often generated via
expensive experiments or large-scale simulations.
## The second way to implement PIML is to design neural network architec-
tures that implicitly embed any prior knowledge and inductive biases associ-
ated with a given task (e.g., symmetry, conservation laws, etc.) (Karniadakis
et al., 2021). CNN is a popular example of this neural network architecture
that achieves extensive applicability for image recognition by respecting in-
variance and symmetry groups in natural images (Karniadakis et al., 2021;
Mallat, 2016). Another example of this category is equivariant networks,
which embed the dynamic changes in spatial coordinates to preserve the
equivariance of data points to rotation and translation (Satorras et al., 2022;
Schütt et al., 2017). Some neural network architectures also used Lagrangian
and Hamiltonian mechanics to enforce the energy conservation property of
the networks (Hao et al., 2023). Furthermore, in solving PDEs, there have
been several attempts to modify neural network architecture to satisfy the
required initial conditions (Hao et al., 2023). Although such a model ar-
chitecture approach can be effective with relatively simple and well-defined
physics or symmetry groups, this approach has limitations in extending to
highly complex problems.
10
