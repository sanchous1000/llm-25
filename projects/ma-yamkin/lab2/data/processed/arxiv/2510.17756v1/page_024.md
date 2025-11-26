---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 24
total_pages: 35
date_converted: "2025-11-05"
---

5. Results and discussion
5.1. Performance of PINN
We evaluate the predictive performance of No-Phy and PINNs with dif-
ferent λsat and λtherm settings, and check whether PINNs can improve the
model performance compared to No-Phy. Additionally, we investigate how
this improvement pattern varies by different training sample sizes. Figure 2
shows the SIV and SIC RMSE changes by PINNs as functions of different
training sample sizes, λsat, and λtherm. Table 2 shows the RMSE, MAE, and
ACC of No-Phy baseline and PINN with 0.2 λtherm and 0.2 λsat. According
to Table 2, the errors of both SIV and SIC predictions decrease with more
training samples.
## Regarding SIV, the integration of physics loss functions, including both
λtherm and λsat, improves the model fidelity relative to the No-Phy baseline
model (Fig. 2a). The reduction in SIV RMSE by the physics loss term is
observed in all sample sizes, but this effect is more significant in the 20 %
sample size: SIV RMSE decreases by up to 0.10 km/day when λsat is set to
0.2. When 50 % and 100 % of training samples are used, the SIV RMSE
decreases by up to 0.02 km/day and 0.05 km/day, respectively. We also find
improvements of SIV MAE and ACC by PINN in all training sample cases
(Table 2).
## On the other hand, regarding SIC, the PINNs show lower SIC RMSEs
than the No-Phy baseline model (Fig. 2), and the improvement of the PINN
is more significant for smaller sample sizes. When 20 % of training samples
are used, the PINNs show approximately 1.1 % lower SIC RMSE than the
No-Phy model. When the training samples are set to 50 % and 100 %, the
24
