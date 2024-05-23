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
    print("Choose a method to run:")
    print("1. Proposed Method")
    print("2. Proposed Method with Stability")
    print("3. Cho Method")
    print("4. Dvorak Method")

    while True:
        choice = input("Enter the number of your choice (1-4): ")
        if choice.isdigit() and 1 <= int(choice) <= 4:
            choice = int(choice)
            break
        else:
            print("Invalid input. Please enter a number between 1 and 4.")

    method_name = {
        1: "Proposed Method",
        2: "Proposed Method with Stability",
        3: "Cho Method",
        4: "Dvorak Method",
    }[choice]

    print(f"\nRunning the {method_name}...\n")

    if choice == 1:
        proposed.proposedCoupling()
    elif choice == 2:
        proposed.proposedCouplingStability(False, True)
    elif choice == 3:
        Sandbox.ChoCoupling()
    elif choice == 4:
        Sandbox.DvorakCoupling()

    print(f"\nThe {method_name} has finished running.\n") 

if __name__ == "__main__":
    main()