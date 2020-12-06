
import math
import numpy as np
from numpy import sqrt
from direct_kinematic import DirectKinematic

import os
import pickle
import pandas as pd

if __name__ == '__main__':

    data=[]
    directKinematic = DirectKinematic()

    for a in range(-180,180 ,20):
        print(a)
        print("len "+ str(len(data)))
        for b in range(-70,70 ,15):
            print("b " + str(b))
            for c in range(-28,105 ,15):
                #print("c " + str(c))
                for d in range(-300,300 ,40):
                    #print("d " + str(d))
                    for e in range(-120,120 ,20):
                        for f in range(-300,300 ,40):
                            input=[a,b,c,d,e,f]
                            input_rad=[]
                            for el in input:
                                input_rad.append(math.radians(el))
                            homogenousCoo = directKinematic.evaluate(input_rad)    # This is in homogenous coordinates

                            coo =[homogenousCoo[0][0] / homogenousCoo[3][0],
                                      homogenousCoo[1][0] / homogenousCoo[3][0],
                                      homogenousCoo[2][0] / homogenousCoo[3][0]]
                            data.append([input_rad, coo])

    print(data[0])
    with open('points.csv', 'wb') as f:
        pickle.dump(data, f)
