---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 33
total_pages: 35
date_converted: "2025-11-05"
---

regions in the central Arctic and Canadian Archipelago (Figs. 5 and 6), our
findings suggest that PINNs can contribute to obtaining consistent model
performance despite the ongoing reduction of MYI coverage.
6. Conclusion
In this study, we propose a physics-informed learning strategy to im-
prove the fidelity of the existing deep learning model for the prediction of
daily sea ice velocity (SIV) and sea ice concentration (SIC) retrieved from
spaceborne remote sensing data. Our physics-informed learning strategy is
achieved through two pathways: (1) design additional physics-informed loss
functions that regularize SIV values based on SIC values and regularize SIC
values based on the assumption of daily thermodynamic ice growth and melt-
ing; (2) insert sigmoid activation function to restrict the output SIC values
into the range of 0 to 1. We implement extensive experiments by employ-
ing the Hierarchical information-sharing U-net (HIS-Unet), which predicts
SIV and SIC through a series of information sharing between SIV and SIC
branches. In order to investigate the impact of training samples on the model
robustness, we train the physics-informed neural network (PINN) and normal
data-driven neural network (No-Phy) separately with three different train-
ing sample ratios (100 %, 50 %, and 20 %) and four different weights to the
physics-informed loss terms (0.0, 0.2, 1.0, and 0.5).
## The results exhibit that the physics-informed learning strategy improves
both SIC and SIV predictions. In particular, the improvement of SIC pre-
diction by PINN is more obvious than SIV with a small number of samples.
## Physics-informed learning strategy helps achieve the benefits of HIS-Unet
33
