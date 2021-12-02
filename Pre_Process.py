# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 00:55:59 2020

@author: Tanzeel
"""

import numpy as np


class MSC(object):
    """
    Class for performing Multiplicative Scatter Correction.

    Methods contain Calibration and Apply utilities.
    """
    
    def __init__(self,reference=None):
        #self.input_data=input_data
        self.reference=reference
        # Whether model has been trained
        self.is_trained = False
        # Dimensionality of training data
        self.train_data_dim = ''
    
    def fit (self,input_data,ref):
        # Define a new array and populate it with the corrected data later    
        data_msc = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            # Run regression
            fit = np.polyfit(ref, input_data[i,:], 1, full=True)
            # Apply correction
            data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
        
        return data_msc
    
    def Calibrate(self,input_data):
        # Get the reference spectrum. If not given, estimate it from the mean    
        ref = np.mean(input_data, axis=0)
        #Fit the msc on the training data
        data_msc=self.fit(input_data,ref)
        
        #Get the shape of input data
        m,n=np.shape(input_data)
        self.reference=ref
        self.train_data_dim=n
        # Mark model as trained
        self.is_trained = True

        return data_msc
    
    def Apply(self,New_data):
        
        M,N=np.shape(New_data)
        if self.reference is None:    
            print("MSC has not been applied on training data. First apply on training data")
          
        else:
            if self.train_data_dim == N:
                ref = self.reference
                new_data_msc=self.fit(New_data,ref)
            else:
                print('''Test data is of different dimensionality
                                 than training data.''')
                    
            return new_data_msc   
     
    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained     
    
    def Get_reference(self):
        """Return the reference (mean) spectrum from training data."""
        if self.reference is None:    
            print("MSC has not been applied on training data. First apply on training data")
        else:
            return self.reference     

class MeanCenter(object):
    """
    Class for performing Mean Center correction.

    Methods contain Calibration and Apply utilities.
    """
    
    def __init__(self,reference=None):
        #self.input_data=input_data
        self.reference=reference
        # Whether model has been trained
        self.is_trained = False
        # Dimensionality of training data
        self.train_data_dim = ''
    
    def fit (self,input_data,ref):
        ''' Perform Mean Center correction'''
        m,n=np.shape(input_data)
        
        rvect=np.ones((m,1))
        mcX    = (input_data-rvect*ref)
        return mcX
    
    def Calibrate(self,input_data):
        # Get the reference spectrum. If not given, estimate it from the mean    
        ref = np.mean(input_data, axis=0)
        #Fit the msc on the training data
        data_mean_center=self.fit(input_data,ref)
        
        #Get the shape of input data
        m,n=np.shape(input_data)
        self.reference=ref
        self.train_data_dim=n
        # Mark model as trained
        self.is_trained = True

        return data_mean_center
    
    def Apply(self,New_data):
        
        M,N=np.shape(New_data)
        if self.reference is None:    
            print("Mean Center has not been applied on training data. First apply on training data")
          
        else:
            if self.train_data_dim == N:
                ref = self.reference
                new_data_mc=self.fit(New_data,ref)
            else:
                print('''Test data is of different dimensionality
                                 than training data.''')
                    
            return new_data_mc   
     
    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained     
    
    def Get_reference(self):
        """Return the reference (mean) spectrum from training data."""
        if self.reference is None:    
            print("MSC has not been applied on training data. First apply on training data")
        else:
            return self.reference     

class Autoscale(object):
    """
    Class for performing Multiplicative Scatter Correction.

    Methods contain Calibration and Apply utilities.
    """
    
    def __init__(self,meanX=None,stdX=None):
        #self.input_data=input_data
        self.reference=meanX
        self.stdX=stdX
        # Whether model has been trained
        self.is_trained = False
        # Dimensionality of training data
        self.train_data_dim = ''
    
    def fit (self,input_data,meanX,stdX):
        
        m,n=np.shape(input_data)
        rvect=np.ones((m,1))
        ax    = (input_data-rvect*meanX)/(rvect*stdX)
        return ax
    
    def Calibrate(self,input_data):
        # Get the reference spectrum. If not given, estimate it from the mean    
        meanX=np.mean(input_data,axis=0)
        stdX=np.std(input_data,axis=0,ddof=1)
    
        #Fit the msc on the training data
        data_autoscale=self.fit(input_data,meanX,stdX)
        
        #Get the shape of input data
        m,n=np.shape(input_data)
        self.meanX=meanX
        self.stdX=stdX
        self.train_data_dim=n
        print(self.meanX,self.stdX)
        # Mark model as trained
        self.is_trained = True

        return data_autoscale
    
    def Apply(self,New_data):
        
        M,N=np.shape(New_data)
        if self.meanX is None:    
            print("Autoscale has not been applied on training data. First apply on training data")
          
        else:
            if self.train_data_dim == N:
                new_data_auto=self.fit(New_data,self.meanX,self.stdX)
            else:
                print('''Test data is of different dimensionality
                                 than training data.''')
                    
            return new_data_auto   
     
    def inv_fit (self,New_data):
        
        meanX=self.meanX
        stdX=self.stdX
        
        m,n=np.shape(New_data)
        rvect=np.ones((m,1))
        ax_inv=New_data*(rvect*stdX)+(rvect*meanX)
        #ax    = (New_data-rvect*meanX)/(rvect*stdX)
        return ax_inv
        
    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained     
    
    def Get_reference(self):
        """Return the reference (mean) spectrum from training data."""
        if self.meanX is None:    
            print("MSC has not been applied on training data. First apply on training data")
        else:
            return (self.meanX,self.stdX)     


def trans2absr(input_data):
    ''' Perform transmission to absorption correction'''
    X_T=np.log10(1/input_data)
    return X_T
