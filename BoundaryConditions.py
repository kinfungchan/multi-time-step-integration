"""
This module is to apply Boundary conditions to the domain
"""


class BoundaryConditions:
    def __init__(self, indexes: list, velocities: list):
        self.indexes = indexes
        self.velocities = velocities