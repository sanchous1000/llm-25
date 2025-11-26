---
source: "..\data\raw\arxiv\2510.17745v1.pdf"
arxiv_id: "2510.17745v1"
page: 4
total_pages: 4
date_converted: "2025-11-05"
---

(a) Chainfire on Intel i9
(b) Synfire on Intel i9
(c) Synfire on ARM Cortex
(d) DCA
Fig. 3. Performance improvements by multi-threading kernel over allocated cores. (a) Chainfire (2k neurons) on 11th Gen Intel Core i9-11900K @ 3.5 GHz
(8 cores, 16 logical). (b) Synfire (1.2k neurons, 77k synapses) on Intel i9. (c) Synfire on several ARM Cortex processors. (d) Dynamic core allocation saves
up to 70% energy as overhead for over-allocated cores thread synchornization is avoided. The performance monitor Intel PCM shows this at ms precision at
timeline of the SNN.
## Fig. 3c presents the the performance improvements of the
same Synfire network on several ARM processors that can
be used at the edge. The kernel has a similar structural
performance scaling on ARM as on Intel, as long as the cores
are used on Intel. Because two Intel logical processors share
a core, the performance degrades, which is indicated by the
dashed line in Fig. 3b.
C. Energy savings by DCA
Fig. 3d shows that DCA reduces the core threads as long as
the real-time criteria are met (first 250 ms). When the system
load demands, DCA allocates additional core threads (e.g. at
750 ms and 1800 ms), respectively frees them, when no longer
needed (e.g. at 750 ms and 1800 ms).
IV. CONCLUSION
The present work introduces a multi-threading kernel for
SNNs. It especially optimizes moderately sized SNNs de-
ployed on the multicore processors used for mobile devices.
## The kernel supports dynamic core allocation to ensure efficient
processing across SoC devices. We show impressive perfor-
mance gains on a network architecture, Synfire chain, which
is commonly used to test SNNs. We also introduce a Chainfire
network to further evaluate performance.
## The synthetic Chainfire network is instrumental to un-
derstand the performance relevant internal workings of the
former multi-threading kernel. It also enabled to validate and
quantitative measure the optimization, see Fig. 3a. Without
it, optimization information is hidden behind noise. There are
great tools available for performance profiling. However, to
identify bottlenecks, it was essential to produce specific loads,
for instance the status update only without interference of the
synaptic processing.
## With the multi-threading kernel extensions, CARLsim now
supports mid-size SNNs on modern multicore processors such
as the ARM Cortex family. Compared to neuromorphic chips
such as Intel’s Loihi or Brainchip’s Akida, which are highly
specialized on neural processing and usually require a host
system to run the base application, CARLsim utilizes the
existing compute capacity of the mobile processor. Depending
on the use case, similar energy efficiency on these specialized
neuromorphic chips might be achieved with CARLsim on
mobile processors. The new power saving policies in CARL-
sim may make the intrinsic energy efficiency of SNNs even
more attractive for edge applications. It makes neuromorphic
applications possible without additional hardware costs. Fur-
thermore, an SNN executed by software is much more flexible
in regards of changes and use case adaptation.
## The present work opens the way for a new generation of
neuromorphic applications that can be deployed on millions
of mobile devices, such as wearables like the Samsung Watch
Ultra 2025 or embedded devices based on SoCs such as the
Raspberry Pi 5.
REFERENCES
[1] L. Niedermeier, K. Chen, J. Xing, A. Das, J. Kopsick, E. Scott,
N. Sutton, K. Weber, N. Dutt, and J. L. Krichmar, “Carlsim 6: An
open source library for large-scale, biologically detailed spiking neural
network simulation,” in 2022 International Joint Conference on Neural
Networks (IJCNN).
IEEE, 2022, pp. 1–10.
[2] S. H¨oppner, Y. Yan, A. Dixius, S. Scholze, J. Partzsch, M. Stolba,
F. Kelber, B. Vogginger, F. Neum¨arker, G. Ellguth, S. Hartmann,
S. Schiefer, T. Hocker, D. Walter, G. Liu, J. Garside, S. Furber,
and C. Mayr, “The spinnaker 2 processing element architecture for
hybrid digital neuromorphic computing,” 2022. [Online]. Available:
https://arxiv.org/abs/2103.08392
[3] S. H¨oppner, Y. Yan, B. Vogginger, A. Dixius, J. Partzsch, F. Neum¨arker,
S. Hartmann, S. Schiefer, S. Scholze, G. Ellguth et al., “Dynamic voltage
and frequency scaling for neuromorphic many-core systems,” in 2017
IEEE International Symposium on Circuits and Systems (ISCAS). IEEE,
2017, pp. 1–4.
[4] E. M. Izhikevich, “Simple model of spiking neurons,” IEEE Trans.
Neural Netw., vol. 14, no. 6, pp. 1569–1572, 2003.
[5] ——, “Which model to use for cortical spiking neurons?” IEEE Trans.
Neural Netw., vol. 15, no. 5, pp. 1063–1070, Sep. 2004.
