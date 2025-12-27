---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 15
total_pages: 35
date_converted: "2025-11-05"
---

algorithm output is the combination of SIC estimations from two algorithms:
the NASA Team algorithm (Cavalieri et al., 1984) and NASA Bootstrap
algorithm (Comiso, 1986). These empirical algorithms estimate SIC from
the PMW brightness temperatures at different frequencies and polarizations:
vertical and horizontal polarizations at 19 GHz, 22 GHz, and 37 GHz. Several
assessments showed that the error of this SIC estimation is approximately
5 % within the consolidated ice pack during cold winter conditions (Meier,
2005; Comiso et al., 1997; Ivanova et al., 2015). However, in the summer
season, the error can rise to more than 20 % due to surface melt and the
presence of melt ponds (Kern et al., 2020). Due to the data quality issue
near coastal areas, we use the SIC data more than 50 km from the coastline.
## To match the coordinate systems of the SIC and SIV products, we reproject
the NSIDC Sea Ice Polar Stereographic grid of the SIC product into the
EASE grid of the SIV product using bilinear interpolation.
3.3. ERA5 climate reanalysis
As shown in Eq.
3, sea ice dynamics are largely associated with at-
mospheric and oceanic circulation. Thus, we use the wind speed and air
temperature as the input variables of the ML model.
## We use the fifth-
generation ECMWF (European Centre for Medium-Range Weather Fore-
casts) atmospheric reanalysis (ERA5) as the data sources for wind velocity
and air temperature. ERA5 provides hourly estimates of atmospheric, land,
and oceanic climate variables, covering the period from January 1940 to the
present (Hersbach et al., 2020). We obtain the daily average wind velocity
(zonal and meridional components) at 10 m height and 2 m air temperature
from this hourly data. To co-locate this data with the SIV and SIC data, we
15
