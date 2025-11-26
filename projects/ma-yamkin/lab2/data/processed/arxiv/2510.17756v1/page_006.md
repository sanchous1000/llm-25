---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 6
total_pages: 35
date_converted: "2025-11-05"
---

sheets and glaciers (i.e., land ice) (Teisberg et al., 2021; Riel et al., 2021; Riel
and Minchew, 2023; Iwasaki and Lai, 2023; Jouvet and Cordonnier, 2023;
He et al., 2023). In sea ice application, Liu et al. (2024) proposed dual-task
neural network architecture and incorporated a loss function based on sea ice
control equation. However, it remains unclear how this PIML concept and
physics-informed loss function can contribute to future sea ice predictions
under rapidly changing climate conditions.
## This study aims to improve the fidelity of deep learning models that
predict sea ice dynamics by integrating physical knowledge into the model
training. In order to enforce the deep learning model (neural network) to con-
verge into physically valid SIV and SIC values, we explicitly adopt a physics-
informed neural network (PINN) approach based on loss function. We em-
bed this physics-informed learning strategy in the hierarchical information-
sharing U-net (HIS-Unet), a CNN model for SIV and SIC predictions (Koo
and Rahnemoonfar, 2024). The main contributions of this research consist
of the following.
• We design physics loss functions and combine them with the data loss
function to regulate physical validity of SIC and SIV.
• We modify the output layer to guarantee the physically valid SIC val-
ues.
• Our extensive experiments show that the physics-informed deep learn-
ing model can improve the performance in SIC and SIV predictions
even with a small number of training samples.
• We explore the spatiotemporal variability in the improved performance
6
