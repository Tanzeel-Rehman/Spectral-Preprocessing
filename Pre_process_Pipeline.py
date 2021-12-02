# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 08:53:14 2021

@author: Tanzeel
"""
from Pre_Process import Autoscale,MSC,MeanCenter,trans2absr

# Example Function to pre-process the training data and then apply the pre-processing means/std on the test data.
def Preprocess_pipeline(X_train,Y_train,X_test,Y_test):
    
    #Pre-process the training X data
    Xp=trans2absr(X_train)
    msc = MSC()
    Xp = msc.Calibrate(Xp)
    mc= MeanCenter()
    Xp = mc.Calibrate(Xp)

    #Pre-process training the Y_data
    auto=Autoscale()
    Yp=auto.Calibrate(Y_train)
    
    #Apply the Pre-processing to  the test X data
    Zp=trans2absr(X_test)
    Zp = msc.Apply(Zp)
    Zp = mc.Apply(Zp)
    
    #Apply the Pre-processing to  the test X data
    Yp_test=auto.Apply(Y_test)
    
    return (Xp,Yp,Zp,Yp_test,auto)