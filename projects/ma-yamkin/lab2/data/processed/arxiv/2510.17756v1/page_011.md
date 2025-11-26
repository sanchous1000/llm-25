---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 11
total_pages: 35
date_converted: "2025-11-05"
---

## Lastly, physics knowledge can be integrated into the optimization and
learning process by imposing constraints of prior physics knowledge into the
form of loss functions.
## As a soft manner of penalizing the loss function
instead of enforcing a specific condition directly, this approach can indirectly
reshape the target spaces of NN output to converge to physically plausible
solutions (Karniadakis et al., 2021; Hao et al., 2023); in this study, we will
refer to this approach specifically as PINN. This approach can be regarded
as a case of multi-task learning, balancing the loss functions for two tasks
of fitting both the observation data and physical constraints. Although such
a soft constraint approach is the most common and flexible way for PIML,
balancing these two tasks can be challenging because they can counteract
the convergence to each other’s solution.
## In the modeling of land ice, PIML has emerged as an effective way to pre-
dict ice flow, and most previous studies employed the optimization approach
by adding physics loss functions or regularization terms (i.e., PINN). Teis-
berg et al. (2021) developed a PINN to predict ice thickness and velocity by
adding a physics loss function based on the mass conservation of ice sheets.
## Riel et al. (2021) introduced a physics loss function derived from the gov-
erning equations of ice flow to their PINN framework to infer spatially and
temporally varying basal drag in ice sheets. Riel and Minchew (2023) pro-
posed a PINN framework to predict the distribution of ice rigidity by adding
the loss functions of ice flow and Kullback-Leibler divergence. Iwasaki and
Lai (2023) proposed a physics loss function to fit the model results with
physics laws regarding ice flows, and this physics loss function contributed
to improving the model performance when the data was contaminated by
11
