---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 9
total_pages: 35
date_converted: "2025-11-05"
---

used CNN to predict monthly SIC from satellite-based SIC observations, sea
surface temperature, air temperature, albedo, and wind velocity data from
the previous months. Similarly, Fritzner et al. (2020) proposed a CNN that
predicts weekly SIC from SIC, sea surface temperature, and air temperature
from the previous six days. CNN models proposed by Ren and Li (2021) and
Ren et al. (2022) used the sequences of satellite-derived SIC observations
to make daily SIC predictions. In some studies, long short-term memory
(LSTM), an advanced recurrent neural network (RNN), has been modified
and inserted into CNN architecture to obtain better performance in both spa-
tial and temporal SIC predictions (Liu et al., 2021). CNN and LSTM have
also been used for SIV prediction in several studies. Zhai and Bitz (2021)
and Hoffman et al. (2023) used CNN to predict daily SIV, and their model
outperformed other statistical and physical models. Petrou and Tian (2019)
showed that adding LSTM units to convolutional layers improves the per-
formance of a model that predicts SIV. In this study, we adopt a multi-task
neural network framework named HIS-Unet (Koo and Rahnemoonfar, 2024)
and improve the predictability of SIV and SIC of this model by incorporating
physics-informed training scheme.
2.3. Physics-informed machine learning (PIML)
Historically, PIML has been achieved by integrating physics knowledge
into ML training in the form of: (i) data, (ii) model architecture, and (iii)
optimization (Hao et al., 2023; Karniadakis et al., 2021). First, one of the
straightforward methods to incorporate physics knowledge into ML is to gen-
erate training data from the desired physics knowledge.
## Data-driven ML
models that are trained with sufficient simulation data governed by certain
9
