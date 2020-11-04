
import math as m
import numpy as np
from numpy import cos, sin


class DirectKinematic(object):

    def __init__(self, parameters={}):
        return

    def evaluate(self, array_of_joints_coordinates):

        o = array_of_joints_coordinates[0]
        t = array_of_joints_coordinates[1]
        s = array_of_joints_coordinates[2]
        f = array_of_joints_coordinates[3]
        r = array_of_joints_coordinates[4]
        e = array_of_joints_coordinates[5]

        transformMat = transformM(o, t, s, f, r, e)

        efMatrix = np.array([[0.3],
                             [0],
                             [0.1],
                             [1]])

        efPosition = np.dot(transformMat, efMatrix)

        return efPosition


# omega: -180 ; 180
# theta: -70 ; 70
# psi: -28 ; 105
# fi: -300 ; 300
# ro: -120 ; 120
# epsilon: -300 ; 300
def transformM(o, t, s, f, r, e, R1=0.188, R2=1.175, R3=1.3, R4=0.2):  # omega, theta, psi, fi, ro, epsilon

    A = np.array([[cos(o), -sin(o), 0, 0],
                  [sin(o), cos(o), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    B = np.array([[1, 0, 0, R1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    C = np.array([[cos(t), 0, sin(t), 0],
                  [0, 1, 0, 0],
                  [-sin(t), 0, cos(t), 0],
                  [0, 0, 0, 1]])

    D = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, R2],
                  [0, 0, 0, 1]])

    E = np.array([[cos(s), 0, sin(s), 0],
                  [0, 1, 0, 0],
                  [-sin(s), 0, cos(s), 0],
                  [0, 0, 0, 1]])

    F = np.array([[1, 0, 0, 0],
                  [0, cos(f), -sin(f), 0],
                  [0, sin(f), cos(f), 0],
                  [0, 0, 0, 1]])

    G = np.array([[1, 0, 0, R3],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    H = np.array([[cos(r), 0, sin(r), 0],
                  [0, 1, 0, 0],
                  [-sin(r), 0, cos(r), 0],
                  [0, 0, 0, 1]])

    I = np.array([[1, 0, 0, R4],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    J = np.array([[1, 0, 0, 0],
                  [0, cos(e), -sin(e), 0],
                  [0, sin(e), cos(e), 0],
                  [0, 0, 0, 1]])

    TM = A * B * C * D * E * F * G * H * I * J

    return TM