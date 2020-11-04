
import math
import numpy as np
from numpy import sqrt
from direct_kinematic import DirectKinematic


class FitnessFunctions(object):

    def __init__(self, parameters={}):
        self.energy_constants = [31.1,21.1,26.6,8.3,5.00,2.6]
        self.velocity_constants = [90,90,90,120,120,190]
        return

    def evaluate_energy(self, array_of_joints_coordinates):
        total_energy=0
        for joint in range(0, 6):
            degrees_tmp=0
            for coo in range(0, 10):
                # rotation=|joint_b-joint_a|
                degrees_tmp=degrees_tmp+abs(array_of_joints_coordinates[(coo+1)%10][joint]-array_of_joints_coordinates[coo][joint])
            total_energy=total_energy+math.degrees(degrees_tmp)*self.energy_constants[joint]
        return total_energy

    def evaluate_operation_time(self, array_of_joints_coordinates):
        total_operation_time = 0
        for coo in range(0, 10):
            degrees=[]
            for joint in range(0, 6):
                # rotation=|joint_b-joint_a|
                degrees.append(abs(array_of_joints_coordinates[(coo+1)%10][joint]-array_of_joints_coordinates[coo][joint]))
            total_operation_time=total_operation_time+max(degrees)*self.energy_constants[degrees.index(max(degrees))]
        return total_operation_time

    def evaluate_rotations(self, array_of_joints_coordinates):
        total_rotations = 0
        for joint in range(0, 6):
            for coo in range(0, 10):
                # rotation=|joint_b-joint_a|
                total_rotations=total_rotations+abs(array_of_joints_coordinates[(coo+1)%10][joint]-array_of_joints_coordinates[coo][joint])
        return total_rotations

    def evaluate_position_accuracy(self, array_of_joints_coordinates, points):

        directKinematics = DirectKinematic()

        total_accuracy = 0

        for i in range(0, 10):

            homogenousPred = directKinematics.evaluate(array_of_joints_coordinates[i])    # This is in homogenous coordinates
            predictedPosition = np.array([[homogenousPred[0][0] / homogenousPred[3][0]],
                                          [homogenousPred[1][0] / homogenousPred[3][0]],
                                          [homogenousPred[2][0] / homogenousPred[3][0]]])       #This is cartesian coordinates

            x_diff = (points[i][0] - predictedPosition[0])**2
            y_diff = (points[i][1] - predictedPosition[1])**2
            z_diff = (points[i][2] - predictedPosition[2])**2
            differences = x_diff + y_diff + z_diff
            total_accuracy += sqrt(differences)

        return total_accuracy[0]
