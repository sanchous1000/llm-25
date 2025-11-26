---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 16
total_pages: 35
date_converted: "2025-11-05"
---

reproject the raw ERA5 latitude-longitude grid onto the 25 km EASE grid
using bilinear interpolation.
4. Methods
In this study, we use a hierarchical information-sharing U-net (HIS-Unet)
based on Koo and Rahnemoonfar (2024) as a backbone deep learning model
(Fig. 1) to make daily predictions of SIC and SIV from inputs of SIC, SIV,
wind, and atmospheric temperature from the previous three days. As a fully
convolutional network, the HIS-Unet is designed to predict SIC and SIV
simultaneously. By sharing SIC and SIV information during the propagation
processes, the HIS-Unet achieves better fidelity than other neural networks
and statistical approaches in predicting both SIC and SIV. In particular, this
information sharing is proven to be useful for predicting sea ice conditions
where and when SIV has impacts on SIC changes (Koo and Rahnemoonfar,
2024).
## Since understanding sea ice dynamics requires both SIV and SIC
information (e.g., Eq. 2), we use the multi-task prediction of SIV and SIC
by the HIS-Unet to facilitate integrating the physics knowledge of sea ice
dynamics. To embed fundamental knowledge of sea ice dynamics into the
HIS-Unet architecture, we (1) introduce physics loss functions in the training
phase and (2) add an additional activation function to the output layer.
## This section presents the details of the HIS-Unet architecture and physics-
informed training strategies used in this study.
4.1. Hierarchical information-sharing U-net (HIS-Unet)
The HIS-Unet architecture consists of two separate task branches, each
for predicting SIV and SIC (Fig. 1). The task branches are connected with
16
