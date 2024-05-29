# multi-time-step-integration

The manuscript "A Multi-Time Stepping Algorithm for the Explicit Finite Element Modelling of Dynamically Loaded Heterogeneous Structures" has been submitted and is under review for the publication in the International Journal for Numerical Methods in Engineering.

This repository allows you to solve for the wave propagation problems in 1D heterogeneous media with multi time step integration. 
Two domains can be integrated with different time steps, that may be driven by material properties or discretisation size.

Authors:
Kin Fung Chan | Nicola Bombace | Duygu Sap | David Wason | Nik Petrinic

Address:
Department of Engineering Science,
University of Oxford, Parks Road, Oxford,
OX1 3PJ, UK

## Why Multi-Time Step Integration?
- Solving a domain with a single time step is often inefficient for heterogeneous domains.
- Hence integrating with multiple time steps was introduced, also known as subcycling or asynchronous integration.
- This can lead to large speedups in the solution of a heterogeneous problem.
- Unlike previously proposed algorithms, we allow for non-integer and non-constant time step ratios between subdomains.
- However we want to ensure that using multiple time steps does not introduce instability into the solution of the problem. To do so, we use an energy balance check.
<p align="center">
    <img src="figures/two-domain-integration-diagram.png" alt="Multi Time Step Integration" width=100%>
</p>
<p align="center">
    <b>Fig 1: A full integration step for a coupling node between two multi time stepping subdomains </b>
</p>

## Getting Started
To Run:
- Download Python onto your local machine
- Install the following python libraries
<pre>
pip install numpy
pip install matplotlib
pip install imageio
</pre>
- git clone the repo to your local machine
- Now to test you can run a single domain with python main.py from the multi-time-step-integration folder. This code gives you the option of running multiple MTS methods
- To solve two subdomains with our novel multi time step algorithm we select the proposed method (1)
<pre>
python main.py
</pre>
This runs the Non-Integer 1D Numerical Example from Section 3 of the paper where a long bar is meshed with uniform linear FE discretisation, but two dissimilar materials.
<p align="center">
    <img src="figures/square-wave-1d-bar.png" alt="One-dimensional heterogeneous domain" width=100%>
</p>
<p align="center">
    <b>Fig 2: One-dimensional heterogeneous domain split into two subdomains with a square wave boundary condition </b>
</p>
You can expect folder(s) with the .gif outputs from each method ran.

## Implementations from Literature
- Comparison with the current state-of-the-art is made with two methods proposed from literature by Cho et al. and Dvorak et al.
- Our algorithm prevents the solution of subdomains at very small time steps, whilst progressing further in time with less steps overall 
<p align="center">
    <img src="figures/dtbars.png" alt="Multi Time Step Integration" width=100%>
</p>
<p align="center">
    <b>Fig 3: A comparison of subdomain integration steps for recent Multi Time Stepping algorithms </b>
</p>

## Running from Notebooks
For those who prefer the use of Jupyter Notebooks, or would like to delve further into the equations of the
algorithms, a Notebooks folder has been created that repeats the same functionality

## Further Questions
For any other questions on how to run the repo or the paper itself, please reach out at kin.chan@eng.ox.ac.uk
<p align="center">
    <img src="figures/elastic-stress-wave-t4.png" alt="Metaconcrete Wave Propagation" width=100%>
</p>
<p align="center">
    <b>Fig 4: Elastic Wave propagation in a metamaterial (metaconcrete) with single time step (monolithic) and multi time stepping solutions for a time step compared</b>
</p>