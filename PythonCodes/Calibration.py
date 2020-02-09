import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

class Calibration(object):
    
    def __init__(self,GazeVector, WorldCoord):
        '''
        Initilization of GazeVector and World Coordinate and camera matrix
        '''
        self.GazeVector = GazeVector
        self.WorldCoord = WorldCoord
        self.camera_matrix = np.array([ [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]], dtype = "double")
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
    def CrossSectionPoints(self, GazeVector):
        '''
        Compute intersection of whole gaze vectors with the virtual plane and return numpy array
        '''
        ImagePoints = np.zeros((GazeVector.shape[0],2),dtype="double")
        for i in range(GazeVector.shape[0]):
            ImagePoints[i,:] = self.CrossSectionPoint(GazeVector[i,:])
        
#        print("ImagePoints", ImagePoints)
        return ImagePoints
    
    def CrossSectionPoint(self,GazeVec):
        '''
        Computer intersection of only one gaze vector and return to a single point (x,y)
        '''
        t= 1/GazeVec[2]
        X_sec = GazeVec[0] * t
        Y_sec = GazeVec[1] * t
        return X_sec,Y_sec
    
    def PnP(self, WorldCoord,ImagePoints):
        '''
        Apply PnP routine and return rotation and translation matrices
        '''
        (_,rvecs, tvecs) = cv2.solvePnP(WorldCoord,ImagePoints, self.camera_matrix, 
                                        self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        Rt, _ = cv2.Rodrigues(rvecs)
        return Rt, tvecs
    
    def PnPRansac(self, WorldCoord,ImagePoints):
        '''
        Apply PnPRansac routine and return rotation and translation matrices
        PnPRansac is less sensitive to errors
        '''
        (_, rvecs, tvecs, inliers) = cv2.solvePnPRansac(WorldCoord,ImagePoints, self.camera_matrix, 
                                                        self.dist_coeffs)
        Rt, _ = cv2.Rodrigues(rvecs)
        print("tvecs= ", tvecs)
        print("Rt=", Rt)
        return Rt, tvecs
    
    def cal_normal_random(self,mean,sigma):
        '''
        Calculate Gaussian noise to be added to the gaze vector elements
        '''
        rand_num = np.zeros(12)
        max_num = 2147483647 
        
        for i in range(12):
            rand_num[i] = random.randint(0, 2147483647)/max_num
    
        c1 = 0.029899776 ;
        c2 = 0.008355968 ;
        c3 = 0.076542912 ;
        c4 = 0.252408784 ;
        c5 = 3.949846138 ;
        
        r = 0.0 ;
        for i in range(12):
            r+= rand_num[i] ;
        
        r = (r-6.0)/4.0 ;
        r2 = r*r ;
        gauss_rand=((((c1*r2+c2)*r2+c3)*r2+c4)*r2+c5)*r
        
        return ((mean+sigma*gauss_rand))

    def Get_ScaleFactor(self,Rt,translation_vector):
        '''
        Compute the unknown scale factor 
        '''
        
        scalefactor = np.zeros((self.WorldCoord.shape[0],1))
        for i in range(self.WorldCoord.shape[0]):
            Sample_3Dpoint = self.WorldCoord[i].reshape(3,1)
            R_s = np.dot(Rt,Sample_3Dpoint).reshape(3,1) + translation_vector.reshape(3,1)
            Pixel_coord = np.dot(self.camera_matrix,R_s)
            scalefactor[i] = Pixel_coord[2]
#            print(scalefactor[i])
        return scalefactor
    
    def Reprojection(self,ImagePoints,S,Rt,translation_vector):
        '''
        Reproject Image points into 3D world coordinate systems to measure error
        '''
        ReprojectionPoints = np.zeros((6,3))
        ImagePoints_conc = np.concatenate((ImagePoints, np.array([1] * 6).reshape(6,1) ), axis=1) 
        ImagePoints_scale = np.multiply(ImagePoints_conc,S)
        
        for i in range(6):
            first_part = np.dot(np.linalg.inv(Rt),np.linalg.inv(self.camera_matrix))
            secon_part = np.dot(first_part, ImagePoints_scale[i])
            third_part = np.dot(np.linalg.inv(Rt),translation_vector)
            ReprojectionPoints[i,:] =  np.transpose((secon_part.reshape(3,1) - third_part))
            
        return ReprojectionPoints
    
    def ImageNoise(self, k):
        '''
        Add noise to images
        '''
        GazeVector_noise = np.zeros((6, 3))
        for i in range(self.GazeVector.shape[0]):
            m = np.linalg.norm(self.GazeVector[i])
            random_quantity = self.cal_normal_random(0,(k+1)*0.01*m)
            print("random=", random_quantity)
            GazeVector_noise[i,:] = (self.GazeVector[i] + random_quantity).reshape(1,3)
            
        ImagePoints_noise = self.CrossSectionPoints(GazeVector_noise) 
        return ImagePoints_noise 
    
    def error(self,ImagePoints_noise,Rt,translation_vector):
        '''
        Compute the reprojection error
        '''
        S = self.Get_ScaleFactor(Rt,translation_vector) 
        ReprojectionPoints = self.Reprojection(ImagePoints_noise, S,Rt,translation_vector)
        reprojection_error = self.WorldCoord - ReprojectionPoints
        
        return reprojection_error
        
    def run(self,Number_Iter):
        '''
         Measure the error as sum of square root of the difference 
         between 3D projected and noisless 3D points
        '''
        error_PnP = np.zeros(Number_Iter)
        error_PnPRansac = np.zeros(Number_Iter)
        
        for k in range(Number_Iter):

            ImagePoints_noise = self.ImageNoise(k)
            Rt_pnp,translation_vector_pnp = self.PnP(self.WorldCoord,ImagePoints_noise)
            Rt_Ransac,translation_vector_Ransac = self.PnPRansac(self.WorldCoord,ImagePoints_noise)
            PnP_diff = self.error(ImagePoints_noise,Rt_pnp,translation_vector_pnp)
            Ransac_diff = self.error(ImagePoints_noise,Rt_Ransac,translation_vector_Ransac)
            error_PnP[k] = np.average(np.linalg.norm(PnP_diff, axis=1))
            error_PnPRansac[k] = np.average(np.linalg.norm(Ransac_diff, axis=1))
        
        return error_PnP,error_PnPRansac
    

        