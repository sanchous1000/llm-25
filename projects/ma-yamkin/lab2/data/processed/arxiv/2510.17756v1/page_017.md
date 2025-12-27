---
source: "..\data\raw\arxiv\2510.17756v1.pdf"
arxiv_id: "2510.17756v1"
page: 17
total_pages: 35
date_converted: "2025-11-05"
---

weighting attention modules (WAMS) that allow information sharing be-
tween the SIV and SIC branches. SIC and SIV branches have U-net struc-
tures (Ronneberger et al., 2015) with a series of encoders (contracting path)
and decoders (expansive path). Each encoder path consists of repeated two
3 × 3 convolutions and a 2 × 2 max pooling operation with stride 2 for
downsampling and doubling the number of features. The decoders conduct
2 × 2 up-convolution and half the number of feature channels, followed by a
concatenation with the cropped feature map and two 3×3 convolutions (Ron-
neberger et al., 2015). The hyperbolic tangent (tanh) activation function is
applied after each convolutional layer:
Tanh(x) = ex −e−x
ex + e−x
(4)
The separated U-net structures of SIV and SIC branches output SIV or
SIC, respectively, but they share and transfer their information through six
weighting attention modules (WAMs) during the propagation process. Six
WAMs are inserted into 3 encoder steps and 3 decoder steps between SIC
and SIV branches (Fig. 1).
## Each WAM first receives information from the SIV and SIC branches
and calculates the weighted sum of them (Fig. 1b). Letting a WAM receive
the SIV feature map (ξin,SIV ; height H, width W, channels C) and SIC
feature map (ξin,SIC; H × W × C), the input shared information (ξin,share)
is determined by multiplying linear weights, Ain,SIV and Ain,SIC, to ξin,SIV
and ξin,SIC, respectively (Fig. 1b). Then, this shared information ξin,share
passes through sequentially arranged channel attention (Fig. 1c) and spatial
attention modules (Fig. 1d). The channel attention highlights what channel
is meaningful, and the spatial attention highlights where an informative part
17
