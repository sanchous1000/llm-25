---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 14
total_pages: 35
date_converted: "2025-11-05"
---

3.1. Sea ice velocity
We use the NSIDC Polar Pathfinder Daily 25 km EASE-Grid Sea Ice
Motion Vectors version 4 (Tschudi et al., 2019, 2020) as the input and output
SIV. This product derives daily SIV from three primary types of sources: (1)
gridded satellite images, (2) wind reanalysis data, and (3) buoy position data
from the International Arctic Buoy Program (Tschudi et al., 2019, 2020). The
u component (along-x) and v component (along-y) of SIV are independently
derived from each of these sources and optimally interpolated onto a 25 km
Equal-Area Scalable Earth (EASE) grid. When SIV is derived from satellite
images, a correlation coefficient is calculated between a small target area in a
one-day image and a searching area in the next-day image. Then, the location
in the next-day image with the highest correlation coefficient is determined as
the displacement of ice (Tschudi et al., 2019). The mean difference between
the Polar Pathfinder and buoy measurements of SIV is approximately 0.1
km/day and 0.3 km/day for u and v components, respectively (Tschudi et al.,
2019). We note that the SIV of this product is valid over short distances
away from the ice edge in areas where ice conditions are relatively stable,
stationary, homogenous, and isotropic (Tschudi et al., 2020). In this study,
we exclude SIV values close to the coastlines within 50 km (or 2 pixels) of
distance.
3.2. Sea ice concentration
For the SIC data, we use NOAA/NSIDC Climate Data Record of Passive
Microwave Sea Ice Concentration version 4 data (Meier et al., 2021). This
data set provides a Climate Data Record (CDR) of SIC (i.e., the areal fraction
of ice within a grid cell) from passive microwave (PMW) data. The CDR
14
