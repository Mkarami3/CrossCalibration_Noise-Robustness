'''
Study on the noise robustness of a closed-form solution to the problem
of cross-calibration between stereoscopic vision systems and 3D remote eye and gaze
trackers. In many applications, it is essential to discover where the gaze of a person locates in
a 3D stereoscopic depth map, and while published algorithms are iterative, there is a closedform
solution which employs an efficient algorithm for solving the Perspective-n-Point (PnP)
problem.

In our case, the stereoscopic vision system provides 3D points Pi in its own coordinate
system. However, the eye tracker provides only an eye position ei and a gaze vector ⃗gi for
each Pi fixated by the observer. We do not have 2D image points corresponding to the 3D
stereoscopic points. An effective way of solving this problem is to perspective project the
vectors ⃗gi onto a virtual projective plane perpendicular to the line of sight of the eye tracker
at a distance of 1 from the projection center of the tracker.


Author: Mohammad Karami
Affiliation: CS department at Western University, Canada
'''

from Calibration import *
import numpy as np
import matplotlib.pyplot as plt

GazeVector = np.array([ (0,3,30), 
                        (-3,-5,28),
                        (10,7,31),
                        (-10,1,25),
                        (-5,2,30),
                        (-6,-4,27)])

WorldCoord = np.array([(0,3,50),
                       (2,-5,47),
                       (-1,7,60),
                       (5,-1,40),
                       (0,2,45),
                       (3,-4,44)],dtype=np.float32)

Number_Iter = 20
PnP = Calibration(GazeVector, WorldCoord)
error_PnP,error_PnPRansac = PnP.run(Number_Iter)


plt.scatter(np.arange(0,Number_Iter),error_PnPRansac)
plt.scatter(np.arange(0,Number_Iter),error_PnP)
plt.xticks(range(Number_Iter))
plt.legend(("PnPRansac","PnP"))
plt.xlabel('$\sigma$')
plt.ylabel('Reprojection Error')
plt.show()