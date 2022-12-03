"""
Assignment Title: Implementation of Standard Genetic Algorithm to solve the Traveling Salesman Problem
Purpose: Models a city that will construct at chosen (x, y) location and num to identify
Language: Implementation in Python

Author: Hemant Ramphul
Github: https://github.com/hemantramphul/
Date: 19 November 2022

Université des Mascareignes (UdM)
Faculty of Information and Communication Technology
Master Artificial Intelligence and Robotics
Official Website: https://udm.ac.mu
"""

import numpy as np
import random as rand


class City:
    """
        An individual city, located by coordinates.

        Attributes:
            x (float): X-coordinate of city.
            y (float): Y-coordinate of city.
            num (int): The city number (identifier).
    """

    def __init__(self, y, x, num):
        self.x = x  # Location x
        self.y = y  # Location y
        self.num = num  # Location identifier num

    """
        Get the distance between self and other cities using euclidean distance formula.

        Args:
            other (City): The other city to which the distance is measured.

        Returns: Distance between cities.
    """

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    """
        Built-in function: Method used to represent a class's objects as a string

        Returns: a string of cities.        
    """

    def __repr__(self):
        return str(self.num) + ": (" + str(self.x) + "," + str(self.y) + ")"

# # Uncomment for test purpose
# # Create city list in term array
# cityList = []
#
# # Create 25 city randomly
# for i in range(0, 20):
#     cityList.append(City(x=int(rand.random() * 200), y=int(rand.random() * 200), num=i))
#
# # Print result
# print(cityList)
