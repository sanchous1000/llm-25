---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 23
total_pages: 35
date_converted: "2025-11-05"
---

of RMSE improvement by PINN, we conduct paired t-tests between RM-
SEs of PINN and RMSEs of No-Phy. If the p-value from the t-test is below
0.05, we determine that the RMSE difference between PINN and No-Phy is
significant.
## In addition to RMSE, we use the mean absolute error (MAE) and anomaly
correlation coefficient (ACC) to evaluate the predictive performance of PINN
and No-Phy.
MAE(ˆy, y) =
PN
i=1 |ˆyi −yi|
N
(12)
ACC(ˆy, y) =
PN
i=1(ˆyi −¯ˆiy)(yi −¯yi)
qPN
i=1(ˆyi −¯ˆiy)2 PN
i=1(yi −¯yi)2
(13)
Lower RMSE and MAE values indicate higher prediction accuracy, while a
higher ACC (i.e., approaching 1) reflects stronger agreement between pre-
dicted and observed anomalies, indicating better predictive skill. We note
that the performance of No-Phy model trained with full training samples can
be found in Koo and Rahnemoonfar (2024) in detail. The No-Phy HIS-Unet
showed better performance than another simple statistical model (e.g., lin-
ear regression), physical model (Hybrid Coordinate Ocean Model; HyCOM),
simple convolutional network, and independent-branch U-nets.
## Since the
HIS-Unet performance has already been assessed in the previous study, this
study is focused on assessing the effect of using a physics-informed learning
strategy rather than assessing the performance of the HIS-Unet architecture
itself.
23
