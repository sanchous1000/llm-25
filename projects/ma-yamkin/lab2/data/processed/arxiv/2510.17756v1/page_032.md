---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 32
total_pages: 35
date_converted: "2025-11-05"
---

the 1980s may no longer fully represent the recent and future non-stationary
sea ice state in the Arctic. Moreover, considering that the Arctic Ocean is
projected to be nearly ice-free during summer as early as 2030-2040 (Masson-
net et al., 2012; Overland and Wang, 2013; Årthun et al., 2021), the future
sea ice dynamics in the Arctic Ocean will likely be different from those in
historical observations. Hence, this raises concerns about the reliability of
purely data-driven deep learning models trained solely on historical records,
as such models may struggle to generalize to recent and future non-stationary
sea ice conditions under rapid climate change. However, our PINN strategy
provides a more robust alternative to fully data-driven learning by integrat-
ing physical principles directly into the learning process, thereby reducing
reliance on historical data alone and enhancing model generalizability in a
non-stationary climate regime.
## In this study, we train our PINN models with 2009-2015 data and evaluate
them with 2016-2022 data. Compared to the training seven years, the test
seven years are characterized as fast-moving sea ice and lower SIC (Fig. A.7).
## Despite these differences, our PINN indeed shows significant improvement
in both SIV and SIC RMSEs compared to the fully data-driven No-Phy
model. We highlight that our PINN models maintain consistent performance
regardless of the number of training samples, whereas the performance of
No-Phy is dependent on the number of training samples (Figs. 5 and 6).
## This suggests the potential of the PINN framework to guarantee significant
predictive performance and generalizability even when the training data do
not fully represent unseen sea ice conditions. Moreover, considering that the
significant improvements by PINN are observed near multi-year ice (MYI)
32
