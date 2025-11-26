---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 21
total_pages: 35
date_converted: "2025-11-05"
---

where λsat and λtherm are the relative weights of Lsat and Ltherm, respectively,
to the data loss term. In this study, we conduct experiments with different
λsat and λtherm of 0, 0.2, 1.0, and 5.0 and examine how this weight changes
the model performance.
## As the second regularization, we enforce the output SIC to lie within the
value range of 0 to 1 (0 to 100 %) by adding the sigmoid activation function
to the last output layer of the SIC branch:
Sigmoid(x) =
1
1 + e−x
(10)
This can constrain the valid range of SIC outputs, even for out-of-training
samples.
4.3. Traininig strategy
In our HIS-Unet, we use the previous 3 days of SIV (x- and y-components),
SIC, air temperature, and wind velocity (x- and y- components) as the inputs
to predict the next day’s SIV and SIC. Consequently, the input layer has 18
channels of 256×256 grid size. All input values are normalized to -1 to 1
based on the nominal maximum and minimum values that each variable can
have. All the data is collected for 14 years from 2009 to 2022; the first seven
years of data (2009-2015) are utilized as training data, and the remaining
seven years of data (2016-2022) are utilized as test data. To examine how
the model performance is changed by the number of training data, we train
the model with three different training sample sizes: (1) using all 2009-2015
data as training samples (i.e., 100 % sampling), (2) randomly selecting 50 %
of 2009-2015 data as training samples (i.e., 50 % sampling), and (3) randomly
selecting 20 % of 2009-2015 data as training samples (i.e., 20 % sampling).
21
