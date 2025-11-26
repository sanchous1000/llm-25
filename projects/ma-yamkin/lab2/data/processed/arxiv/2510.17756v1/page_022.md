---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 22
total_pages: 35
date_converted: "2025-11-05"
---

## The total number of training samples is 2,177 for the 100 % sampling, 1,088
for 50 % sampling, and 436 for the 20 % sampling. The HIS-Unet models
trained with three different sample sizes are applied to the 2016-2022 test
data. All models are optimized by the Adam stochastic gradient descent
algorithm (Kingma and Ba, 2017) with 100 epochs and a 0.001 learning rate.
## All scripts are executed on eight NVIDIA RTX A5000 GPUs with 24 GB of
memory.
4.4. Model performance
The model performance is assessed by the root mean square error (RMSE):
RMSE(ˆy, y) =
sPN
i=1(ˆyi −yi)2
N
(11)
where ˆy denotes predicted values, y denotes true values, N is the number
of data points.
## In the case of SIV, we calculate RMSE for both x- and
y-component SIV, and the average of x- and y-component RMSEs is deter-
mined as the SIV RMSE to embrace the magnitude and angle error of SIV.
## We assess and compare the RMSEs from the HIS-Unet with three different
training sample cases and four different λsat and λtherm (i.e., 0, 0.2, 1.0, and
5.0). We use the HIS-Unet without any physics-informed regularization (i.e.,
trained only with the data loss Ldata and without the sigmoid activation
function to the SIC branch) as the baseline model, and this purely data-
driven model is notated as No-Phy. Meanwhile, we call the physics-informed
HIS-Unet simply PINN. We compare the RMSE differences between PINN
and No-Phy for different training sample sizes (20 %, 50 %, and 100 %).
## Additionally, we examine how these RMSE differences vary by month and
region in seven test years (2019-2022). To assess the statistical significance
22
