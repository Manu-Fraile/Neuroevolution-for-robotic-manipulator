
import math
import numpy as np
from numpy import sqrt
from direct_kinematic import DirectKinematic


class FitnessFunctions(object):

    def __init__(self, parameters={}):
        self.energy_constants = [31.1,21.1,26.6,8.3,5.00,2.6]
        self.velocity_constants = [90,90,90,120,120,190]
        return

    def evaluate_energy(self, array_of_joints_coordinates, verbose=False):
        total_energy=0
        degrees_tmp= [0] * 6
        all_data=[0]*10

        for coo in range(0, 10):
            for joint in range(0, 6):
                tmp=abs(array_of_joints_coordinates[coo][joint]-array_of_joints_coordinates[(coo-1+10)%10][joint])
                if verbose:
                    print(tmp)
                all_data[coo]=total_energy+math.degrees(tmp)*self.energy_constants[joint]
        total_energy=np.sum(all_data)
        if verbose:
            return total_energy, all_data
        return total_energy

    def evaluate_operation_time(self, array_of_joints_coordinates, verbose=False):
        total_operation_time = 0
        degrees_tmp= [0] * 6
        all_data=[0]*10

        for coo in range(0, 10):
            degrees=[]
            for joint in range(0, 6):
                # rotation=|joint_b-joint_a|
                degrees.append(abs(array_of_joints_coordinates[coo][joint]-array_of_joints_coordinates[(coo-1+10)%10][joint]))
            all_data[coo]=max(degrees)*self.energy_constants[degrees.index(max(degrees))]
        total_operation_time=np.sum(all_data)
        if verbose:
            return total_operation_time, all_data
        return total_operation_time

    def evaluate_rotations(self, array_of_joints_coordinates, verbose=False):
        total_rotations = 0
        all_data=[0]*10
        for joint in range(0, 6):
            for coo in range(0, 10):
                # rotation=|joint_b-joint_a|
                all_data[coo]=all_data[coo]+total_rotations+abs(array_of_joints_coordinates[coo][joint]-array_of_joints_coordinates[(coo-1+10)%10][joint])
        total_rotations=np.sum(all_data)
        if verbose:
            return total_rotations, all_data
        return total_rotations

    def evaluate_position_accuracy(self, array_of_joints_coordinates, points, verbose=False):

        directKinematics = DirectKinematic()

        total_accuracy = 0

        all_data=[]

        for i in range(0, 10):

            homogenousPred = directKinematics.evaluate(array_of_joints_coordinates[i])    # This is in homogenous coordinates
            #homogenousPred = directKinematics.evaluate([0.345,0.720,-0.153, 2.120,0.874,1.620])    # This is in homogenous coordinates

            predictedPosition = np.array([[homogenousPred[0][0] / homogenousPred[3][0]],
                                          [homogenousPred[1][0] / homogenousPred[3][0]],
                                          [homogenousPred[2][0] / homogenousPred[3][0]]])       #This is cartesian coordinates

            x_diff = (points[i][0] - predictedPosition[0])**2
            y_diff = (points[i][1] - predictedPosition[1])**2
            z_diff = (points[i][2] - predictedPosition[2])**2
            differences = x_diff + y_diff + z_diff
            all_data.append(sqrt(differences))
            total_accuracy += all_data[i]
        if verbose:
            return total_accuracy, all_data
        return total_accuracy[0]

if __name__ == '__main__':
    trajectory_points = [[2.25, 1.1, 0.25],
                        [0.9, 1.5, 0.25],
                        [-0.85, 1.14, 2.22],
                        [-1.8, 1.25, 1.17],
                        [1.8, 1.25, 1.17],
                        [-1.25, -1.1, 0.25],
                        [-2.25, -1.48, 0.25],
                        [0.45, -1.14, 2.22],
                        [0.8, -1.25, 2.35],
                        [0.8, -1.25, -1.35]]
    outputs =  [[0.473, 0.592, -0.230, 0.130, 0.008,-0.617],
                            [1.026, 0.293, -0.008, 0.132, 1.155, -0.617],
                            [2.086, -0.014, -0.270, 2.890, 1.155, -0.617],
                            [2.523, 0.179, -0.270, 2.890, -0.440, -0.617],
                            [0.597, 0.179, -0.270, 2.890, -0.440, -0.617],
                            [-2.417, 0.179, 0.434, 2.887, -0.665, -0.617],
                            [-2.465, 0.794, -0.459, 1.342, -0.665, -0.617],
                            [-1.087, -0.189, -0.462, 0.324, -0.665, -0.617],
                            [-0.966, 1.188, 0.215, 0.130, 0.008, -0.617],
                            [-0.951, -0.100, -0.462, 0.130, -0.526, -0.617]]
    fitnessFunctions = FitnessFunctions()
    total_accuracy, accuracies = fitnessFunctions.evaluate_position_accuracy(outputs, trajectory_points, True)
    print(total_accuracy)
    print(accuracies)
