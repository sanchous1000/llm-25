---
source: "..\data\raw\arxiv\2510.17745v1.pdf"
arxiv_id: "2510.17745v1"
page: 2
total_pages: 4
date_converted: "2025-11-05"
---

## Depending on the network size, efficient operation may
require parallel processing on GPUs, multi-core CPUs, or
neuromorphic hardware. For instance, small networks up to
a few hundred neurons run most efficiently in a single thread
on a CPU. The efficient simulation of large-scale networks
with millions of neurons, as typically used in computational
neuroscience [8], are the core feature of CARLsim as its
PThreads based kernel scales over multiple GPUs [9]. In the
case of mid-size networks (e.g., 103 neurons), which are often
used for edge applications, the overhead for PThreads has poor
performance compared to an execution on a single thread.
## In the present work, we introduce a multi-threading kernel
that scales efficiently over the available cores of modern CPUs
used in SoCs such as ARM Cortex-76 in Raspberry Pi 5. This
addresses the performance limitation of PThreads in CARLsim
and other SNN simulators. All code and models are open-
source and available on [10]. The main contributions of this
work are:
1) Multi-threading for SNNs. A multi-threading kernel
that scales independently of the partitioning of the
network.
2) Dynamic load balancing. A load-balancing algorithm
that dynamically allocates computation to cores and
avoids synchronization bottlenecks of idle threads.
3) Performance monitoring. An SNN performance moni-
tor with ms precision.
4) Synthetic load network. A synthetic load network
named Chainfire that produces and measures neural and
synaptic activity.
5) SNN Benchmarks. Concrete benchmark results on Intel
and ARM processors.
6) Edge processing. SNNs that fulfill real-time require-
ments on off-the shelf mobile processors, without the
need of specialized neuromorphic hardware.
II. METHODS
A. Spiking neuron model
CARLsim efficiently implements spiking neural networks
such as LIF and the Izhikevich neuron model with 4 and 9
parameters [1]. In contrast to [2], we implemented the Synfire
network utilizing Izhikevich 4-parameter model described by
the following equations [4].
˙v
=
0.04v2 + 5v + 140 −u + I
(1)
˙u
=
a(bv −u)
(2)
if v ≥30
(
v = c
u = u + d
(3)
The present simulations use Forward Euler for numerical
integration. More complicated neuron models with multiple
compartments may require other numerical methods, such as
Runge-Kutta, to handle instabilities. The CARLsim kernel is
designed to handle these cases.
B. Synfire network architecture
We utilize the Synfire chain network as a benchmark to mea-
sure and compare performance of CARLsim and potentially
other neuromorphic chips. We follow H¨oppner et. al. in their
approach for SpiNNaker [2] and built a network with the same
structure and sizing, see Fig. 1. The excitatory groups E has
200 regular spiking (RS) neurons (a = 0.02, b = 0.2, c =
−65, d = 8), and the inhibitory groups consist of 50 fast
spiking (FS) neurons (a = 0.1, b = 0.2, c = −65, d = 2).
## The code and configuration is Open Source and available in
the CARLsim GitHub repository [10]. Using the CARLsim
multi-thread kernel, we replicated Synfire results for current
based (CUBA) and conductances based (COBA) synapses. See
Fig. 3db and GitHub repository [11] for the benchmark runs
and Subsection II-B.
C. Chainfire network architecture
In addition to using the Synfire chain SNN for benchmark-
ing, we created a Chainfire network, which could more readily
control parameters that affect CPU loads. Fig. 2 shows the
synthetic load network Chainfire with the minimal amount
of synapses that allows to produce and measure arbitrary
loads, specifically targeted at the compute-intensive status
updates of neurons, defined by equations 1 - 3. The network
is designed to generate loads similar to the Synfire network
and therefore has four clusters, that are marked in blue. This
allows validation, tuning, scaling of the parallelization, and
provides the maximum of performance improvement possible
by this kernel.
Fig. 2. Chainfire SNN for validation, tuning, and scaling of the parallelization.
## The blue rectangles contain groups of excitatory neurons. It can induce specific
loads on the multi-threading kernel. The green squares depict synchronization
neurons that dictate the fan-in and fan-out between neuron groups.
## The parallel rows are chains of excitatory RS neurons with
specific delays and weights (e.g. dexc = 5 ms, wexc = 0.432),
configured to propagate the spike wave with minimal delay
and without noise. The synchronization neurons, indicated
by a green frames, are the also excitatory RS neurons and
consolidate the fan-in for predecessor group or integrate the
fan-out.
## The feed-forward load generator SNN is configurable by the
following parameters: N total number of neurons per cluster,
d ms of delay for pre-synaptic to post-synaptic neuron in the
parallel chains that are sequential connected, and the span
in ms that determine the length of the chain, e.g. N = 100,
d = 20ms, span = 100ms results in the shown network groups
with 100 neurons, structured in 4 rows (parallel chains) and
five columns.
## The fan-in from synchronization neurons are configured, so
that a spike activates the input layer (first column) at the same
time. The spike-wave travels through the group and activates
