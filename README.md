# multi-time-step-integration

DRAFT in process: 
A Multi-Time Stepping Algorithm for the Modelling of Heterogeneous Structures with Explicit Time Integration

Authors:
Kin Fung Chan | Nicola Bombace | Duygu Sap | David Wason | Nik Petrinic

Address:
Department of Engineering Science,
University of Oxford, Parks Road, Oxford,
OX1 3PJ, UK

To Run:
- Download Python onto your local machine
- Install the following python libraries
- pip install numpy
- pip install matplotlib
- pip install imageio
- git clone the repo to your local machine
- Now to test you can start and run with python SimpleIntegrator.py from the multi-time-step-integration folder

<pre>
```python
python MultiTimeStepIntegration.py 
```
</pre>
runs the 1D Numerical Example from Section 3 of the paper with the following .gif output

![MultiTimeStep GIF](Updated_Multi-time-step.gif)

For comparison an integer subcycling algorithm is also implemented, that uses interpolated velocities as seen in (Belytschko, 1979)

For those who prefer the use of Jupyter Notebooks, or would like to delve further into the equations of the
algorithms, a Notebooks folder has been created that repeats the same functionality
