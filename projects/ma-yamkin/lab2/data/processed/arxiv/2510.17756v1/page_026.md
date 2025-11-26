---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 26
total_pages: 35
date_converted: "2025-11-05"
---

## Table 2: Assessment of SIV and SIC predictions for PINN (λtherm = 0.2 and λsat = 0.2)
and No-Phy with different training sample sizes. The best accuracy is highlighted in bold.
## Sample size
20 %
50 %
100 %
SIV
Model
RMSE
(km/d)
MAE
(km/d)
ACC
RMSE
(km/d)
MAE
(km/d)
ACC
RMSE
(km/d)
MAE
(km/d)
ACC
PINN
2.778
1.954
0.853
2.732
1.915
0.859
2.629
1.844
0.867
No-Phy
2.873
2.053
0.843
2.759
1.936
0.856
2.684
1.880
0.863
SIC
Model
RMSE
(%)
MAE
(%)
ACC
RMSE
(%)
MAE
(%)
ACC
RMSE
(%)
MAE
(%)
ACC
PINN
6.687
3.441
0.975
6.222
3.118
0.978
5.854
2.917
0.981
No-Phy
7.393
4.335
0.969
6.611
3.533
0.975
6.197
3.274
0.978
SIC RMSEs are approximately 0.5 % and 0.3 % lower for the PINNs than the
No-Phy models, respectively. Besides the RMSE comparison, the MAE and
ACC are also improved by PINN (Table 2). These results suggest that the
incorporation of physics-informed architecture and optimization is beneficial
for SIC predictability under limited training sample scenarios. However, the
weights to the physics loss function are not so significant for the SIC accuracy:
no consistent trend in SIC RMSE is observed with increasing λtherm or λsat
values.
5.2. Temporal characteristics of model performance
Figure 3 shows the monthly RMSE of SIV prediction for seven test years
(2016-2022) with different sample sizes. For the PINN model, we set λsat =
0.2 and λtherm = 0.2 based on the optimal configuration identified in Figure
2. In general, SIV prediction shows relatively lower errors in summer (June-
August) due to the slow drift speed during these months (Fig. A.7c). The
improvement of SIV RMSE by PINN (i.e., RMSE difference between PINN
and No-Phy model) is evident in most months when the sample size is 20
26
