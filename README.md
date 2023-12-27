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
- Hence integrating with multiple time steps was introduced, also known as subcycling.
- Unlike previously proposed algorithms, we allow for non-integer and non-constant time step ratios between subdomains.
- However we want to ensure that using multiple time steps does not introduce instability into the solution of the problem. To do so, we use an energy balance check.

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
- Now to test you can start and run with python SimpleIntegrator.py from the multi-time-step-integration folder

<pre>
python MultiTimeStepIntegration.py 
</pre>
runs the 1D Numerical Example from Section 3 of the paper with the following .gif output

![MultiTimeStep GIF](Updated_Multi-time-step.gif)

## Implementations from Literature
For comparison an integer subcycling algorithm is also implemented, that uses interpolated velocities as seen in (Belytschko, 1979)

## Running from Notebooks
For those who prefer the use of Jupyter Notebooks, or would like to delve further into the equations of the
algorithms, a Notebooks folder has been created that repeats the same functionality

## Further Questions
For any other questions on how to run the repo or the paper itself, please reach out at kin.chan@eng.ox.ac.uk
