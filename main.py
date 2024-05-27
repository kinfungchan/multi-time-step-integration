import sys
import os
import literature
import proposed
import numpy as np

# Add the top-level directory to the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

"""
This main module is for the Multi Time Stepping Integration Method

K.F. Chan, N. Bombace, D. Sap, D. Wason, and N. Petrinic (2024),
A Multi-Time Stepping Algorithm for the Modelling of Heterogeneous
Structures with Explicit Time Integration, I.J. Num. Meth. in Eng.

"""

class Bar_1D:
    def __init__(self):
        # Large Domain
        self.E_L = 0.02 * 10**9 # 0.02GPa
        self.rho_L = 8000
        self.length_L = 50 * 10**-3 # 50mm
        self.area_L = 1 # 1m^2
        self.num_elem_L = 300
        # Small Domain
        self.E_S = (np.pi/0.02)**2 * self.rho_L # Non Integer Time Step Ratio = pi
        self.rho_S = self.rho_L
        self.length_S = 2 * 50 * 10**-3 
        self.area_S = self.area_L
        self.num_elem_S = 600
        # Safety Parameter
        self.safety_Param = 0.5

def main(bar):

    print("Choose a method to run:")
    print("1. Proposed Method")
    print("2. Proposed Method with Stability")
    print("3. Cho Method")
    print("4. Dvorak Method")
    print("5. Run All Methods") 

    while True:
        choice = input("Enter the number of your choice (1-5): ")
        if choice.isdigit() and 1 <= int(choice) <= 5:
            choice = int(choice)
            break
        else:
            print("Invalid input. Please enter a number between 1 and 5.")

    method_name = {
        1: "Proposed Method",
        2: "Proposed Method with Stability",
        3: "Cho Method",
        4: "Dvorak Method",
        5: "List of All Methods"
    }[choice]

    print(f"\nRunning the {method_name}...\n")

    if choice == 1:
        proposed.proposedCoupling(bar)
    elif choice == 2:
        proposed.proposedCouplingStability(False, True)
    elif choice == 3:
        literature.ChoCoupling(bar)
    elif choice == 4:
        literature.DvorakCoupling(bar)
    elif choice == 5:  # Run all methods
        proposed.proposedCoupling()
        proposed.proposedCouplingStability(False, True)
        literature.ChoCoupling(bar)
        literature.DvorakCoupling(bar)

    print(f"\nThe {method_name} has finished running.\n")

if __name__ == "__main__":
    Bar_1D = Bar_1D()
    main(Bar_1D)