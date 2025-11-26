---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 8
total_pages: 35
date_converted: "2025-11-05"
---

forces on sea ice based on the most common assumption of elastic-viscous-
plastic (EVP) properties of sea ice (Hibler, 1979):
mDu
Dt = −mfk × u + τai + τwi + F −mg∇H
(3)
where D/Dt = ∂/∂t+u·∇is the substantial time derivative, m is the ice mass
per unit area, k is a unit vector normal to the surface, u is the ice velocity, f is
the Coriolis parameter, τai and τwi are the forces due to air and water stresses,
H is the elevation of the sea surface, g is the gravity acceleration, and F is
the force due to variations in internal ice stress. Following this equation,
many previous studies described SIV as interacting with wind and ocean
forcings (Rampal et al., 2016; Wang et al., 2014; Timmermann et al., 2009;
Salas Mélia, 2002). Particularly, wind velocity has been treated as a major
variable in SIV, contributing to up to 70 % of the sea ice velocity variances
(Thorndike and Colony, 1982) depending on season or region. Nevertheless,
predicting sea ice dynamics based on physical models is still challenging due
to the intrinsic complexity and dependency of physical models on numerous
parameterizations.
2.2. Neural network for sea ice prediction
Convolutional neural networks (CNNs) have been used as the most pop-
ular and efficient deep learning network for modeling SIC and SIV. First, in
terms of SIC, Andersson et al. (2021) proposed a deep-learning sea ice fore-
casting system named IceNet to forecast monthly SIC for the next six months.
## Grigoryev et al. (2022) used a U-Net architecture, a type of deep CNN, for
daily SIC forecasts in several subsections of the Arctic Ocean, including the
Barents and Kara Seas, Labrador Sea, and Laptev Sea. Kim et al. (2020) also
8
