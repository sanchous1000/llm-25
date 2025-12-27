---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 12
total_pages: 35
date_converted: "2025-11-05"
---

noise. Jouvet and Cordonnier (2023) optimized their PINN by minimizing
the energy associated with high-order ice-flow equations. Cheng et al. (2024)
employed a PINN to infer basal sliding while filling gaps in sparsely observed
ice thickness data. In the regime of sea ice, there have been several attempts
to integrate physics knowledge into machine learning by training the model
with the data retrieved from physical models (Palerme and Muller, 2021;
Palerme et al., 2024). Liu et al. (2024) used a dual-task CNN architecture
and physics-informed loss function to enforce dynamic constraints of SIC and
SIV. Despite such developments of PINN for cryosphere, it is still necessary
to assess how PINN and physical constraints can help future predictions of
sea ice conditions. Therefore, this study explicitly applies the concept of
PINN to sea ice prediction and explore the how weights to physics-informed
loss function and representability of training samples contribute to the model
predictability for future unseen sea ice conditions.
3. Data
In this study, we use daily SIV and SIC data collected from satellite
observation from 2009 to 2022. We use the SIV and SIC from previous three
days (inputs of the model) to predict the next day’s SIV and SIC (output of
the model) (Table 1). Besides satellite observations of SIV and SIC, we also
use wind velocity and air temperature from reanalysis as additional input
variables for the model. Additionally, we add X and Y coordinates as inputs
to represent regional variability. The input datasets and their sources are
summarized in Table 1. This section presents the details of these datasets.
12
