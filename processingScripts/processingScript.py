
# coding: utf-8

# ### Part 2. Time series analysis

# ##### Package imports

# In[1]:

#Miscellanous
#from IPython.core.display import display #display inline package
#from ipywidgets import interact, interactive, fixed, FloatProgress
#import ipywidgets as widgets
import pdb #debugging tool
import pandas as pd # pandas dataframe package
import pandas
from datetime import datetime #date conversion tool
from xlrd.xldate import xldate_as_tuple #xldate converter
from urllib2 import urlopen #get data from web tool
#from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image
#import utm #lat long to utms
import zipfile #unzip tool
from IPython.display import clear_output
from scipy.signal import correlate #rainfall correlation

#numpy packages and tools
import numpy as np
from numpy import linspace
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

#Spatial data packages
import rasterio #io raster data 
#from osgeo import gdal

#Plotly tools
import plotly.tools as tls
tls.set_credentials_file(username='NTPlotly', api_key='48kd2al3f2') #my plotly credentials (please dont use they cost me $$)
import plotly.plotly as py
#from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly import tools
#import cufflinks as cf
import matplotlib

#scipy packages
from scipy.misc import derivative #derivates of functions tool
from scipy.interpolate import UnivariateSpline #basic spline fitting tool
from scipy.signal import gaussian #gaussian filtering tool
from scipy.ndimage import filters #anotehr filtering tool
from scipy import stats #core scipy stats package
from scipy.integrate import simps #simpsons rule package for trapezoid calcualtions in area under curve calcs

#Basic math tools
from math import log
from math import factorial
import random
import argparse


# ### Spline fitting and crossing tests to detect and map peaks and troughs

# In[2]:

#simple function to call moving average function
def doParameters(y):
        
    avg, param = movingAverage(y)
           
    return param    


# In[3]:

#Gaussian type filter to find both the average and the variance of the raw data.
def movingAverage(series, sigma=3):

    b = gaussian(39, sigma)
    
    average = filters.convolve1d(series, b/b.sum())
    
    var = filters.convolve1d(np.power(series-average,2), b/b.sum())
    
    return average, var


# ##### Spline model

# A univariate spline model

# In[4]:

def doSpline(x,y,paramVar,paramMan):
    
    #pdb.set_trace()
    
    spl1 = UnivariateSpline(x, y, w=0.9/np.sqrt(paramVar)) # weighting factor method
    #spl1 = UnivariateSpline(x, y) # weighting factor method   
    
    spl2 = UnivariateSpline(x, y,s=paramMan) #manual smoothing method
    
        
    return spl1,spl2


# ##### Derivative estimation

# Estimates the 1st derivative / slope of the spline function at each data point and appends to list

# In[5]:

def findDerivatives(spl,x):
    
    splDeriv = []
       
    for i in x:

        a = spl.derivatives(i)

        splDeriv.append(a[1])
    
    
    return splDeriv


# ##### Crossing axis test 

# Simple test to find changes in the sign of the derivate estimations as crossing points.

# In[6]:

def crossingTest(first):

    peakCrossing = np.zeros(len(first))

    troughCrossing = np.zeros(len(first))

    for i in range(len(first)):

        if i >0:

            d1 = first[i-1]

            d2 = first[i]
            
            d1s = sign(d1)

            d2s = sign(d2)

            if d1s > d2s:

                peakCrossing[i]=1

            elif d1s < d2s:

                troughCrossing[i]=1

    return peakCrossing, troughCrossing


# In[7]:

#basic sign test
def sign(x): 

    return 1 if x >= 0 else -1


# ##### Main peak to trough function

# Function that takes the raw data inputs applies the spline and derivate functions, returns peaks, troughs for both manual and weighted parameterisation methods and returns the zero filtered raw data for further analyses and plotting.

# In[8]:

def doDerivativePeakTrough(data,dates,nBands):

    zeros = data > 0 #create a mask of the zeros to remove them

    y = data[zeros] #apply the zero mask to the data
            
    dates = np.array(dates) #turn dates into a numpy array
    
    newDates = dates[zeros]#apply the same zero mask to the data

    x = linspace(1,len(y),len(y)) #create a new x axis from the data with the zeros removed
   
    paramVar = doParameters(y) # call a function that sets weighting factor based on the variance of the raw data.
    
    paramMan = 0.1 # set the weighting factor manually
    
    splVar,splMan= doSpline(x,y,paramVar,paramMan) #call fitted spline function and return models for both manual and variance fitting methods
        
    splDataV = splVar(x) #returns y spline fitted data (variance fit method)
    
    splDataM = splMan(x) #returns y spline fitted data (manual fit method)
 
    splDerivVar = findDerivatives(splVar,x) #return the first derivate of the spline function (variance fit method)
    
    splDerivMan = findDerivatives(splMan,x) #return the first derivate of the spline function (manual fit method)
    
    peakCrossingVar,troughCrossingVar = crossingTest(splDerivVar) # performs a crossing test to determine peaks and troughs  (variance fit method)
    
    peakCrossingMan,troughCrossingMan = crossingTest(splDerivMan) # performs a crossing test to determine peaks and troughs (manual fit method)

    peaksVar = peakCrossingVar == 1 #filter crossing test result to index the peaks (variance fit method)

    troughsVar = troughCrossingVar == 1 #filter crossing test to index the troughs (variance fit method)
    
    peaksMan = peakCrossingMan == 1  #filter crossing test result to index the peaks (manual fit method)

    troughsMan = troughCrossingMan == 1 #filter crossing test to index the troughs (manual fit method)
    
    return peaksVar,troughsVar,peaksMan,troughsMan,splDataV, splDataM,newDates,y,splDerivVar


# ### Determining the utilisation periods

# This section aims to identify the utilisation periods in the time series.

# ##### Organise data into a pandas dataframe

# Takes the results from the peak and trough detection and organises into a new data frame

# In[9]:

def doPandasDf(peaksVar,troughsVar,splDataV,newDates,newData,firstDeriv):
    
    #pdb.set_trace()
                    
    df = pandas.DataFrame(newDates,columns=['dates'])
            
    df['dates'] = pandas.to_datetime(newDates)
    
    df['peaks'] = peaksVar

    df['troughs']= troughsVar

    df['npv'] = newData

    df['fittedSpline'] = splDataV
    
    df['firstDeriv'] = firstDeriv
    
    return df 


# ##### Masking

# Function that checks for exceptions to expected ground cover dynamic's. For example heavilly timbered areas, persistant bare areas such as scalds, pans, dry lakes, timbered and non-timbered hills and mountains. A null value is returned where these areas are encountered. 
# 
# Utilisation modelling is then applied to areas not masked.
# 
# Note: this is very brutal at this stage and will likely need refinement!!

# In[10]:

def doChecking(peakVals,troughVals,df,nBands):
    
    sDate = []

    eDate = []
    
    slopes = []

    auc = []

    rSqr = []
    
    slopesF = []

    aucF = []
    
    aucS = []

    rSqrF = []

    imageCount = []
    
    npv = []
    
    fitted = []

    troughFlag = 0

    peakFlag = 0   
    
    #pdb.set_trace()
    
    ## Filtering process
    
    if len(troughVals)>2 and len(peakVals) >2: #this filters only where more than one peak and trough exist.

        startDif = troughVals.index[0] - peakVals.index[0] 

        if startDif <0:# this checks and corrects for when a trough is encountered prior to a peak at the start of the time series

            troughVals = troughVals.iloc[1:]#this slices the ts from the second item to the end.

            troughFlag =1

        endDif =  troughVals.index[-1] - peakVals.index[-1]
        
        if endDif <0:#this checks and corrects for when a peak at the end is not proceeded by a trough. The previous peak is used.

            peakVals = peakVals.iloc[:-1]#this slices the trough list from the beginning to the second last.

            peakFlag = 1

        for i in range(len(peakVals)):

                start = peakVals.index[i]

                startD = peakVals['dates'].iloc[i]

                startD = startD.to_pydatetime()
                
                #pdb.set_trace()

                #startD = startD.toordinal() #Commented this out for graphing purposes to use in output image need to uncomment

                end = troughVals.index[i]

                endD = troughVals['dates'].iloc[i]

                endD = endD.to_pydatetime()

                #endD = endD.toordinal() #Commented this out for graphing purposes to use in output image need to uncomment
                
                npvTemp = peakVals['npv'].iloc[i]
                
                fitTemp = npvTemp = peakVals['fittedSpline'].iloc[i]

                if start < end:
                    
                    slope,r_value,area,t = doModellingRaw(df,start,end)
                    
                    slopeF,r_valueF,areaF,areaSmall = doModellingFitted(df,start,end)

                    sDate.append(startD)

                    eDate.append(endD)

                    slopes.append(slope)

                    auc.append(area)

                    rSqr.append(r_value**2)
                    
                    slopesF.append(slopeF)

                    aucF.append(areaF)
                    
                    aucS.append(areaSmall)

                    rSqrF.append(r_valueF**2)

                    imageCount.append(len(t))
                    
                    npv.append(npvTemp)
                    
                    fitted.append(fitTemp)

    else:

        #pdb.set_trace()
        
        #cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

        #df.iplot(kind='scatter', mode='lines', x='dates', y='fittedSpline', filename='cufflinks/simple-scatter')
        
        zeros = [0] * 300

        sDate = zeros

        eDate = zeros

        slopes = zeros

        auc = zeros

        rSqr = zeros
        
        slopesF = zeros

        aucF = zeros
        
        aucS = zeros

        rSqrF = zeros

        imageCount = zeros
        
        npv = zeros
        
        fitted = zeros
        
    return sDate,eDate,slopes,auc,rSqr,slopesF,aucF,aucS,rSqrF,imageCount,npv,fitted


# #####  Model slope and area under the curve

# ###### Area under curve and slope model (actual data)

# In[11]:

def doModellingRaw(df,start,end):
        
    npvRaw = df['npv'].loc[start:end]
    
    t = np.linspace(0,3,len(npvRaw))

    y= np.array(npvRaw,dtype=float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(t,y)

    area = simps(y, t)   
    
    return slope,r_value,area,t


# ###### Area under curve and slope model (fitted data)

# In[12]:

def doModellingFitted(df,start,end):
        
    npvFit = df['fittedSpline'].loc[start:end]

    t = np.linspace(0,3,len(npvFit))

    y= np.array(npvFit,dtype=float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(t,y)
    
    ySmallIntegral = y-y[-1]
    
    #pdb.set_trace()

    area = simps(y, t)   
    
    areaSmall = simps(ySmallIntegral,t)
    
    return slope,r_value,area,areaSmall


# ##### Rescale

# Outputs from modelling are rescaled from 1 to 100 for the purposes of image output.

# In[13]:

def doRescale(slopes,slopesF,auc,aucF,aucS):
    
    slopeRescaled = remap(slopes, min(slopes), max(slopes), 1, 100)

    aucRescaled = remap(auc, min(auc), max(auc), 1, 100)
        
    slopeRescaledFit = remap(slopesF, min(slopesF), max(slopesF), 1, 100)

    aucRescaledFit = remap(aucF, min(aucF), max(aucF), 1, 100)   
    
    aucSmallRescaledFit = remap(aucS, min(aucS), max(aucS), 1, 100)
    
    
    return slopeRescaled,aucRescaled,slopeRescaledFit,aucRescaledFit,aucSmallRescaledFit


# In[14]:

def remap(x, oMin, oMax, nMin, nMax):

    #check reversed input range
    
    reverseInput = False
    
    #pdb.set_trace()
    
    oldMin = min( oMin, oMax )
    
    oldMax = max( oMin, oMax )
    
    if not oldMin == oMin:
    
        reverseInput = True

    #check reversed output range
    
    reverseOutput = False   
    
    newMin = min( nMin, nMax )
    
    newMax = max( nMin, nMax )
    
    if not newMin == nMin :
    
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    
    if reverseInput:
        
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    
    if reverseOutput:
        
        result = newMax - portion

    return result


# ##### Main utilisation function

# In[15]:

def doSlicing(df,nBands):
         
    #Get the peak locations

    mask = df['peaks']== True

    peakVals = df[mask]

    #get the trough locations

    mask = df['troughs']== True

    troughVals = df[mask]
    
    #pdb.set_trace()
    
    sDate,eDate,slopes,auc,rSqr,slopesF,aucF,aucS,rSqrF,imageCount,npv,fitted = doChecking(peakVals,troughVals,df,nBands)
    
    modellingDf = pandas.DataFrame()
   
    modellingDf['sDate']= sDate

    modellingDf['eDate']= eDate

    modellingDf['rSqr']= rSqr
    
    modellingDf['rSqrF']= rSqrF

    modellingDf['imageCount']= imageCount
    
    modellingDf['npv']= npv
    
    modellingDf['fitted']= fitted
    
    

    if sum(imageCount)>0 and sum(slopes)is not 0 :

        slopeRescaled,aucRescaled,slopeRescaledFit,aucRescaledFit,aucRescaledFitSmall = doRescale(slopes,slopesF,auc,aucF,aucS)

        #with rescaling
        
        modellingDf['slope']= slopeRescaled

        modellingDf['auc']= aucRescaled

        modellingDf['slopeF'] = slopeRescaledFit

        modellingDf['aucF'] = aucRescaledFit
        
        modellingDf['aucS'] = aucRescaledFitSmall
        
        #without rescaling
        
        modellingDf['slopeUsc']= slopes

        modellingDf['aucUsc']= auc

        modellingDf['slopeFUsc'] = slopesF

        modellingDf['aucFUSc'] = aucF
        
        modellingDf['aucSUSc'] = aucS
        
           

    else:

        #add zeros if no data
        
        modellingDf['slope']= 0

        modellingDf['auc'] = 0

        modellingDf['slopeF'] = 0

        modellingDf['aucF']= 0
        
        modellingDf['aucS'] = 0
        
        
        modellingDf['slopeUsc']= 0

        modellingDf['aucUsc'] = 0

        modellingDf['slopeFUsc'] = 0

        modellingDf['aucFUSc']= 0
        
        modellingDf['aucSUsc'] = 0

    return modellingDf


# In[16]:

#simple function to find the closest date in a list of dates, i.e. find the closest image date to a monthly rainfall date
def nearestDate(base, dates):
    
    #pdb.set_trace()
        
    nearness = { abs(base - date) : date for date in dates }
    
    return nearness[min(nearness.keys())]


# In[17]:

def doPixelLoop(npv,dates,profile,siteName):
    
    #display(f)#progress bar
    
    (nBands,nRows,nCols) = npv.shape
    
    noOutputBands = 300 #this is an arbitory number for output arrays
    
    #output arrays
    outputSlope = np.zeros((noOutputBands,nRows,nCols))
    
    outputAuc = np.zeros((noOutputBands,nRows,nCols))
    
    outputSlopeF = np.zeros((noOutputBands,nRows,nCols))
    
    outputAucF = np.zeros((noOutputBands,nRows,nCols))
    
    outputDataFrame = pd.DataFrame()
         
    outputDataFrame2 = pd.DataFrame()    
    
    #looping over image pixel by pixel (slow - bottle neck)
    
     

    for i in range(nRows):
        
            
        for j in range(nCols):
            
                                    
            drill = npv[:,i,j] #drill down thorugh image and return 1d array

            #call peak trough function - see func for details            
            peaksVar,troughsVar,peaksMan,troughsMan,splDataV, splDataM,newDates,newData,firstDeriv = doDerivativePeakTrough(drill,dates,nBands)

            #Note: the manual parametrisation method has not been used any further, need to test this at some stage and compare to weighted method

            #call func that organises output from peak trough into a pandas df
            dfV = doPandasDf(peaksVar,troughsVar,splDataV,newDates,newData,firstDeriv)

            dfV['x']=int(i)

            dfV['y']=int(j) 

                    #call func that slices the time series up into peak to trough components and stores in pandas dataframe
            modelledData = doSlicing(dfV,nBands)

            modelledData['x']=int(i)

            modelledData['y']=int(j)            

            outputDataFrame = pd.concat([outputDataFrame,modelledData])   

            outputDataFrame2 = pd.concat([outputDataFrame2,dfV])
            
    fName1 = 'data/output_' + siteName + '.csv'
    
    outputDataFrame.to_csv(fName1,index_label="index")  
    
    fName2 = 'data/outputTimeSeries_' + siteName + '.csv'
    
    #outputDataFrame2.to_pickle('data/outputTimeSeriesPk.pkl')
                
    outputDataFrame2.to_csv(fName2,index_label="index")             


# In[18]:

#Function that reads the download image into memory
def openTif(filename):
    
    '''
    npv = []
    profile = gdal.Open('chip.tif', gdal.GA_ReadOnly)
    for i in xrange(1, profile.RasterCount+1):
        npv.append(profile.GetRasterBand(i).ReadAsArray())
    
    '''
    with rasterio.open(filename) as src:
        npv = src.read()
        profile = src.profile
        
    return npv,profile
    


# In[19]:

def doProcess(siteName):
    
    print 'Processing started.....'
    
    #pdb.set_trace()  
    
    npv,profile = openTif('data/chip.tif')

    temp = pandas.read_csv('data/dates.csv',header =0,parse_dates=['0'],dayfirst=True) #reads the csv into a pandas array
        
    
    dates = pandas.DataFrame()
    
    dates['dates']=temp['0']
    
    b,r,c = np.shape(npv)
       
    
    doPixelLoop(npv,dates,profile,siteName)
    
    print '...complete.'
    



