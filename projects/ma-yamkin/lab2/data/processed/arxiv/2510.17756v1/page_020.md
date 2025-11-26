---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 20
total_pages: 35
date_converted: "2025-11-05"
---

% of SIC (Tschudi et al., 2019), SIV should be zero where SIC is less than
15 %. Therefore, we design the first physics loss term as follows:
Lsat =





|u2
p + v2
p|,
if Ap < 0.15
0,
if Ap ≥0.15
(6)
The next physics loss term (Ltherm) is based on Eq.2, which explains the
thermodynamic and dynamic SIC changes. The thermodynamic SIC changes
(SA), the right term in Eq. 2, can be calculated by temporal changes of SIC
and the combination of advection and divergence of SIC. In the case of daily
SIC prediction, the daily SA can be assumed not to exceed (-1, 1). That
is, we assume that the thermodynamic freezing and melting of sea ice are
unlikely to saturate SIC from 0 % to 100 % or remove the entire sea ice from
100 % to 0 % within a day in most cases. Thus, based on this assumption,
we design the second physics loss term as follows:
Ltherm = ReLU(|∂Ap
∂t + ∇· (upAp)| −1)
(7)
where the time derivative term (∂Ap
∂t ) is derived by subtracting the previous-
day SIC from the output SIC, and the spatial derivative term (∇· (upAp))
is derived by calculating the spatial gradients in output SIV and SIC grids.
## By using Ltherm, we can assign a linearly increasing penalty as the thermo-
dynamic ice growth prediction exceeds 1.
## Consequently, the total physics loss term (Lphy) and the final objective
loss functions (L) are defined as follows:
Lphy = λsatLsat + λthermLtherm
(8)
L = Ldata + Lphy
(9)
20
