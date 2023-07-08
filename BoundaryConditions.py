"""
This module is to apply Boundary conditions to the domain
"""


class VelBoundaryConditions:
    def __init__(self, indexes: list, velocities: list):
        self.indexes = indexes
        self.velocities = velocities

class AccelBoundaryConditions:
    def __init__(self, indexes: list, accelerations: list):
        self.indexes = indexes
        self.accelerations = accelerations