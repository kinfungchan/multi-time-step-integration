import sys
import os
import literature
import proposed
from utils import Paper
from bar import Bar_1D, Bar_1D_HighHet, Bar_1D_HighHetUnstable

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

"""
This main module is for the Multi Time Stepping Integration Method

K.F. Chan, N. Bombace, D. Sap, D. Wason, and N. Petrinic (2024),
A Multi-Time Stepping Algorithm for the Modelling of Heterogeneous
Structures with Explicit Time Integration, I.J. Num. Meth. in Eng.

"""
def main(bar):

    print("Choose a method to run:")
    print("1. Proposed Method")
    print("2. Proposed Method with Stability")
    print("3. Cho Method")
    print("4. Dvorak Method")
    print("5. Run All Methods") 
    print("6. Monolithic Method")

    while True:
        choice = input("Enter the number of your choice (1-6): ")
        if choice.isdigit() and 1 <= int(choice) <= 6:
            choice = int(choice)
            break
        else:
            print("Invalid input. Please enter a number between 1 and 6.")

    method_name = {
        1: "Proposed Method",
        2: "Proposed Method with Stability",
        3: "Cho Method",
        4: "Dvorak Method",
        5: "List of All Methods",
        6: "Monolithic Method"
    }[choice]

    print(f"\nRunning the {method_name}...\n")

    if choice == 1:
        proposed.proposedCoupling(bar)
    elif choice == 2:
        prop = proposed.proposedCouplingStability(bar, False, True)
    elif choice == 3:
        cho = literature.ChoCoupling(bar)
    elif choice == 4:
        dvo = literature.DvorakCoupling(bar)
    elif choice == 5:  # Run all methods
        proposed.proposedCoupling(bar)
        prop = proposed.proposedCouplingStability(bar, False, True)
        cho = literature.ChoCoupling(bar)
        dvo = literature.DvorakCoupling(bar)
        # Comparison of Methods Outputs
        paper = Paper(prop, cho, dvo)
        paper.all_plots()
    elif choice == 6:
        proposed.monolithic()       

    print(f"\nThe {method_name} has finished running.\n")

if __name__ == "__main__":
    bar = Bar_1D()
    # bar = Bar_1D_HighHet()
    # bar = Bar_1D_HighHetUnstable()
    main(bar)
