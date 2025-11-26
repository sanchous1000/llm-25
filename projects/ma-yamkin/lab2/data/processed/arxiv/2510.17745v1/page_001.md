---
source: "..\data\raw\arxiv\2510.17745v1.pdf"
arxiv_id: "2510.17745v1"
page: 1
total_pages: 4
date_converted: "2025-11-05"
---

A Multi-threading Kernel for Enabling
Neuromorphic Edge Applications
Lars Niedermeier∗‡, Vyom Shah†, and Jeffrey L. Krichmar†‡
∗Niedermeier Consulting, Zurich, ZH, Switzerland
†Department of Computer Science, University of California, Irvine, CA, USA
‡Department of Cognitive Sciences, University of California, Irvine, CA, USA
Correspondence Email: lars@niedermeier-consulting.ch
Abstract—Spiking Neural Networks (SNNs) have sparse, event-
driven processing that can leverage neuromorphic applications.
## In this work, we introduce a multi-threading kernel that enables
neuromorphic applications running at the edge, meaning they
process sensory input directly and without any up-link to or
dependency on a cloud service. The kernel shows speed-up gains
over single thread processing by a factor of four on moderately
sized SNNs and 1.7X on a Synfire network. Furthermore, it
load-balances all cores available on multi-core processors, such
as ARM, which run today’s mobile devices and is up to 70%
more energy efficient compared to statical core assignment. The
present work can enable the development of edge applications
that have low Size, Weight, and Power (SWaP), and can prototype
the integration of neuromorphic chips.
## Index Terms—Edge Computing, Neuromorphic Applications,
Spiking Neural Networks
I. INTRODUCTION
Spiking Neural Networks (SNNs) mimic natural nervous
systems with elements that replicate the sparse, all-or-none
neural activity. This makes them a good fit to take advantage of
low Size, Weight, and Power (SWaP) computing systems. The
downside of SNNs are the higher computational costs for the
numerical solution of the differential equations that determine
the neuron’s state. Software packages such as CARLsim have
evolved over the years to provide a mature framework for
computational neuroscientists, embedded systems engineers,
and roboticists [1].
## To measure, the potential of SNNs for neuromorphic edge
applications, methods are needed to quantitatively measure
performance. Recently, a multilayered recurrent SNN bench-
mark, called a Synfire chain, was used to measure the energy
efficiency of the SpiNNaker2 neuromorphic chip [2]. The
Synfire network, which consisted of leaky-integrated-and-fire
(LIF) spiking neurons, is used to measure the dynamic voltage
and frequency scaling (DVFS) developed for SpiNNaker [3].
## In the present work, we investigate the Synfire benchmark
with CARLsim and the Izhikevich neuron model [4], [5].
## Compared to LIF neurons, the Izhikevich neuron has a wider
range of dynamics and can mimic a variety of neuron types
found in the brain. Fig. 1 presents the Synfire network de-
veloped to benchmark CARLsim. The excitatory groups E
consist of regular spiking (RS) neurons found as pyramidal
cells in the cortex, the inhibitory groups I of fast spiking (FS)
interneurons. The network is segmented in four partitions that
are assigned to a dedicated core. We follow [6] in connecting
the last segment partition3 to the first paritition0 in contrast
to the original referenced Synfire network is a strict feed-
forward-inhibition (FFI) network [7].
(a) Synfire network as CARLsim kernel benchmark.
(b) Spikewave propagation with correlated inhibition.
(c) PThreads kernel (CARLsim 4).
(d) New multi-threading kernel.
Fig. 1.
## The CARLsim Synfire network with normal spike generator for
the stimulus. (a) Izhikevich neurons in four partitions recurrently linked.
(b) Neural activity visualized in CARLsim spike monitor. (c) Paritions with
fixed core affinity in pthreads kernel of CARLsim4. (d) The new kernel is
more energy efficient by assigning cores dynamically to free up SoC compute
resources.
arXiv:2510.17745v1  [cs.NE]  20 Oct 2025
