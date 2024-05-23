import sys
import os
import literature
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
        proposed.proposedCoupling()
    elif choice == 2:
        proposed.proposedCouplingStability(False, True)
    elif choice == 3:
        literature.ChoCoupling()
    elif choice == 4:
        literature.DvorakCoupling()
    elif choice == 5:  # Run all methods
        proposed.proposedCoupling()
        proposed.proposedCouplingStability(False, True)
        literature.ChoCoupling()
        literature.DvorakCoupling()

    print(f"\nThe {method_name} has finished running.\n")

if __name__ == "__main__":
    main()