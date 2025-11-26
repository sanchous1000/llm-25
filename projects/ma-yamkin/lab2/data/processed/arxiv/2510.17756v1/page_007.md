---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 7
total_pages: 35
date_converted: "2025-11-05"
---

of our physics-informed learning strategies compared to fully data-
driven deep learning models that do not embed physics.
## The remainder of the paper is organized as follows. Section 2 provides a
brief review of the physical background of sea ice modeling, machine learn-
ing for sea ice prediction, and PIML. Section 3 explains details of remote
sensing and meteorological data used in this study, and section 4 presents
the architecture of our CNN model and physics-informed learning strategies.
The performance and implication of our PINN are discussed in Section 5.
2. Background
2.1. Physics of sea ice dynamics
In physical sea ice models, the volume and area of sea ice are determined
by thermodynamic evolution and dynamic motion field. The spatiotemporal
changes in SIT (h) and SIC (A) can be expressed by the following equations
for mass conservation (Hibler, 1979; Holland and Kwok, 2012; Flato, 2004):
∂h
∂t + ∇· (uh) = Sh
(1)
∂A
∂t + ∇· (uA) = SA
(2)
where u is the ice motion vector, Sh and SA are the changes in SIT and SIC
driven by thermodynamic sources (e.g., freezing or melting), respectively.
## Additionally, the following momentum equations have been used in var-
ious numerous physical sea ice models to explain the balance of horizontal
7
