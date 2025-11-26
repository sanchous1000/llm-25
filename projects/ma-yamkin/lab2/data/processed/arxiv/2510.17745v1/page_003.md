---
source: "..\data\raw\arxiv\2510.17745v1.pdf"
arxiv_id: "2510.17745v1"
page: 3
total_pages: 4
date_converted: "2025-11-05"
---

each neuron of the adjacent column. The output layer (last
column) transmits the spikes to the fan-out which has reduced
weights, so that the synchronization neurons produce a single
spike. This in turn feeds back to the first column of the
Chainfire.
## The frequency of the input spike generator is 1 Hz. The
delays enforce a travel time of the spike-wave of around
500 ms (400 ms for the cluster plus inter-neurons, and some
processing time around 2 ms for the RS neurons). The spike-
wave travels through the network without interference to the
last synchronization neuron, before the next stimuli is induced.
D. Benchmark processors
SNNs running at the edge can take advantage of off-the-
shelf multi-core processors. The CARLsim multi-threaded
kernel utilizes available cores of modern energy efficient
CPUs, such as the ARM Cortex-76 in Raspberry Pi 5. These
processors are typically built in System-on-chips (SoCs), and
are used to operate devices at the edge. For a moderate sized
SNN to run efficiently at the edge, care must be taken to ensure
all cores have fully balanced loads. Algorithm 1 outlines
parallelization used in the multi-threaded kernel.
Algorithm 1: Multi-threading kernel for neuron state updates.
## Input: simulation time in ms, network partition netid, numerical integration
steps per ms S, core threads T
Output: spikes
1 while step ∈S do
/* numerical integration */
2
while group ∈partion do
/* initialize shared */
3
while neuron ∈group do
/* in parallel T */
4
Isum = CUBA/COBA;
/* synaptic fan-in */
5
v, u = EULER/RK4(dudt, dvdt, a, b, c, d, Isum)
6
if v > 30 then
/* Izhikevich */
7
spike = true;
8
runtimeData.recovery[lNId] = u;
/* update */
E. Load balancing by dynamic core assignment (DCA)
One reason SNNs are energy efficient is that their activity
is sparse with a firing rate of a few Hz with occasional spike
bursts. CARLsim utilizes the intrinsic Dynamic Voltage and
Frequency Scaling (DVFS) of modern CPUs, which provide
advanced energy policies. Intel no longer recommends direct
manipulation of the P-states to manipulate DVFS and rec-
ommends only to use power profiles for the sake of system
stability.
## The overhead for thread synchronization cannot be ne-
glected. Consequently, our load-balancer allocates only the
minimum cores necessary to fulfill the performance criteria,
for instance real-time, meaning 1 ms in the SNN corresponds
to 1 ms wall clock time.
III. RESULTS
We provide benchmark results for the Intel i9 with 8 cores
and several ARM Cortex processors with 4 cores. All results
can be reproduced with the open source of the CARLsim
repository. Furthermore, we provide supplemental material
such as videos and log of the run on GitHub [11].
A. Performance gain by multi-threading on Chainfire
We measure Chainfire performance on a release build and
a version with debugging enabled. Fig. 3a and Tables I and
II present the performance of a Chainfire network with 2000
neurons run on a Intel i9 desktop processor. The simulation
model time was 10s. The overall spike count is 20, 040 and
applied as a measure to determine that the neural activity is
same for all (parallel) runs. The texecution (treal) is the wall
clock time, which is implemented by the C++ standard library
(STL) std::chrono::steady clock. Performance is measured by
a speed factor, defined as tmodel/texecution. For example, if
the simulation of the SNN with eight threads runs in 2.28s,
it has a speed factor = 4.4. If the single threaded run takes
8.88s, it a the speed factor = 1.1. The resulting performance
gain is then 4.4/1.1 = 4.0.
TABLE I
PERFORMANCE IMPROVEMENTS CHAINFIRE (RELEASE BUILD)
Threads
Execution Time
Speed Factor
Performance Gain
1
8.88 s
1.1 x
2
5.20 s
1.9 x
1.7 x
4
3.09 s
3.2 x
2.9 x
8
2.38 s
4.2 x
3.8 x
16
2.28 s
4.4 x
4.0 x
32
5.87 s
1.7 x
1.5 x
TABLE II
PERFORMANCE IMPROVEMENTS CHAINFIRE (DEBUG BUILD)
Threads
Execution Time
Speed Factor
Performance Gain
1
14.82 s
67.5%
2
8.49 s
1.2 x
1.8 x
4
4.93 s
2.0 x
3.0 x
8
3.96 s
2.5 x
3.7 x
16
3.72 s
2.7 x
4.0 x
32
5.87 s
1.7 x
1.9 x
B. Performance gain by multi-threading on Synfire
Fig. 3b and Table III present the performance improvement
on the Synfire chain network with 1,200 neurons and 77K
synapses run on a Intel i9 desktop processor. As expected, the
performance gain is lower than on the synthetic SNN as the
optimization for the multi-threading kernel aimed primarily on
the compute intense numerical integration of the differential
equations for the neuron state. On the other hand, the network
is rather small and 70% performance gain is actually above
expectations. Future work will aim to parallelize the synaptic
processing.
TABLE III
PERFORMANCE IMPROVEMENTS SYNFIRE (RELEASE BUILD)
Threads
Execution Time
Speed Factor
Performance Gain
1
0.43 s
7 x
2
0.31 s
9.7 x
1.4 x
4
0.26 s
12.0 x
1.7 x
8
0.26 s
12.0 x
1.7 x
16
0.29 s
10.0 x
1.4 x
32
0.62 s
4.8 x
0.7 x
