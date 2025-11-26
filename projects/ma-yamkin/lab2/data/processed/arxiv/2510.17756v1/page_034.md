---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 34
total_pages: 35
date_converted: "2025-11-05"
---

even in the case of a lack of sufficient training datasets.
## Therefore, this
physics-informed learning strategy can contribute to improving the perfor-
mance of deep learning models for sea ice prediction, even if the past sea ice
data cannot fully represent the future non-stationary sea ice conditions af-
fected by rapid climate change. Additionally, this strategy can also be easily
applied for a multi-day forecast of sea ice conditions combined with recurrent
network architectures, such as LSTM.
## Software and data availability
The codes that were used for the prediction of sea ice concentration and
velocity using Python language (version 3.11) based on PyTorch library can
be found in Github: https://github.com/BinaLab/PINN_seaice. This repos-
itory was created by Younghyun Koo (e-mail: kooala317@gmail.com) in 2023
and contains program codes (111 MB) and training data (3.41 GB). The au-
thors’ experimental environment was as follows:
• OS: Windows 11 Pro
• CPU: Intel(R) Core(TM) i7-11700F
• RAM: 16.0 GB
• GPU: NVIDIA GeForce RTX 3070
The sea ice velocity and concentration data can be downloaded free of charge
from NSIDC (Tschudi et al., 2019; Meier et al., 2021), and the ERA5 weather
data can be downloaded free of charge from the Copernicus Climate Data
Store (Hersbach et al., 2020).
34
