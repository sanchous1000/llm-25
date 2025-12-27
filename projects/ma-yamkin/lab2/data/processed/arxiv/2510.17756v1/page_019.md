---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 19
total_pages: 35
date_converted: "2025-11-05"
---

is spatially located (Woo et al., 2018). These channel and spatial attention
modules are also applied to the input SIV and SIC feature maps (ξin,SIV and
ξin,SIC) (Fig. 1b). Then, the attention shared information is sent to the SIV
and SIC branches after multiplying output weights (Aout,SIV and Aout,SIC)
and adding attention SIV and SIC information, respectively. More details
of how the WAMs enable sharing and highlighting SIC and SIV information
are described in Koo and Rahnemoonfar (2024).
4.2. Physics-informed training
In order to make the HIS-Unet incorporate physics knowledge of sea ice,
we embed two physics-informed regularizations: (1) include physics loss func-
tions along with the data loss function and (2) insert the sigmoid activation
function to the SIC branch to guarantee valid SIC values.
## The original HIS-Unet is optimized by the mean square error (MSE)
objective loss function (Ldata). The MSE data loss term (Ldata) is calculated
by the following equation:
Ldata =
X
(|up −uo|2 + |vp −vo|2 + |Ap −Ao|2)
(5)
where u and v denote x-component and y-component of SIV, respectively, A
denotes SIC, and the subscript o means observation and p means prediction
by HIS-Unet. In addition to this MSE data loss function, we design physics
loss functions that apply physical constraints: (i) Lsat constraining valid SIV
values and (ii) Ltherm constraining thermodynamically valid ice growth.
## First, we regularize the valid SIV values associated with SIC values. Since
the PMW-derived SIV can be only defined where sea ice presents with > 15
19
