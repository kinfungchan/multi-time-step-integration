import sys
import os
import Sandbox
import proposed

# Add the top-level directory to the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

"""
This main module is for the Multi Time Stepping Integration Method

K.F. Chan, N. Bombace, D. Sap, D. Wason, and N. Petrinic (2024),
A Multi-Time Stepping Algorithm for the Modelling of Heterogeneous
Structures with Explicit Time Integration, I.J. Num. Meth. in Eng.

"""

def main():
    
    # Proposed Method
    run_proposed = input("Do you want to run the Proposed Method? (y/n): ")
    if run_proposed.lower() == "y":
        print("Running the Proposed Method")
        proposed.proposedCoupling()

    # Proposed Method with Stability
    run_proposed_stability = input("Do you want to run the Proposed Method with Stability? (y/n): ")
    if run_proposed_stability.lower() == "y":
        print("Running the Proposed Method with Stability")
        proposed.proposedCouplingStability(False, True)

    # Cho Method
    run_cho = input("Do you want to run the Cho Method? (y/n): ")
    if run_cho.lower() == "y":
        print("Running the Cho Method")
        Sandbox.ChoCoupling()

    # Dvorak Method
    run_dvorak = input("Do you want to run the Dvorak Method? (y/n): ")
    if run_dvorak.lower() == "y":
        print("Running the Dvorak Method")
        Sandbox.DvorakCoupling()

    # Finished Running the Methods
    print("Finished Running the Methods")
    print("Summary of Methods:")
    if run_proposed.lower() == "y":
        print("- Proposed Method")
    if run_proposed_stability.lower() == "y":
        print("- Proposed Method with Stability")
    if run_cho.lower() == "y":
        print("- Cho Method")
    if run_dvorak.lower() == "y":
        print("- Dvorak Method")


if __name__ == "__main__":
    main()