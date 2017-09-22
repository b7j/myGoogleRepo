#!/usr/bin/env python
# coding: utf-8


#interactive web map package
import folium 
from folium import features
import shapefile
from json import dumps
from folium import plugins

#Miscellanous
from IPython.core.display import display #display inline package
from ipywidgets import interact, interactive, fixed, FloatProgress
import ipywidgets as widgets
import pdb #debugging tool
import pandas as pd # pandas dataframe package
import pandas
from datetime import datetime #date conversion tool
from xlrd.xldate import xldate_as_tuple #xldate converter
from urllib2 import urlopen #get data from web tool
#from collections import OrderedDict
#import warnings
#warnings.filterwarnings('ignore')
from IPython.display import Image
#import utm #lat long to utms
import zipfile #unzip tool
from IPython.display import clear_output
from scipy.signal import correlate #rainfall correlation

#numpy packages and tools
import numpy as np
from numpy import linspace

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

import processingScript as process

# #### Data import

# ##### Import site coords

# Import a list of the ~500 sites across the study area with each sites latitude and longitude.

# In[15]:

#Function to get a number of lists and shapefiles about the study area.

def getdata():
    
    
        #reads in a prepared list of sites asat September 2016 from dropbox
        url = "https://dl.dropboxusercontent.com/u/53285700/csvs/siteDetails.csv"  # dl=1 is important
        u = urlopen(url)
        data = u.read()
        u.close()
        filename = "data/siteDetails.csv"

        with open(filename, "wb") as f :
            f.write(data)

        df = pd.read_csv(filename) 
        
        '''

        #reads in a prepared list of images for downloading, note the images have been created, stored and extracted into dropbox folder from the dsiti system using the following script on the hpc: UQ_createNPVChips.py
        url = "https://dl.dropboxusercontent.com/u/53285700/images/images.txt"  # dl=1 is important
        u = urlopen(url)
        data = u.read()
        u.close()
        filename = "data/imageNames.csv"

        with open(filename, "wb") as f :
            f.write(data)

        images = pd.read_csv(filename,header=None)
        
        '''
        images = pd.read_csv('images.txt',header=None)
        
        

        #read in zip file containing shapefile of sub ibra bioregions
        url = "https://www.dropbox.com/s/d63c5m0jac6jvct/NTSubIbra.zip?dl=1"  # dl=1 is important
        u = urlopen(url)
        data = u.read()
        u.close()
        filename = "NTSubIbra.zip"

        with open(filename, "wb") as f :
            f.write(data)
            
        return df, images




# #### Enter site details and download image

# First four functions are called when the site id is entered and downlaod button clicked (see text box and botton below)

# In[19]:

#Function to download selected image and associated list of image dates csv file into local directory

def getDropboxImage(url): 
    
    #image download
    u = urlopen(url)
    data = u.read()
    u.close()
    filename = "data/chip.tif"
 
    with open(filename, "wb") as f :
        f.write(data)
    
    #pdb.set_trace()
    #dates csv download
    urlTemp = url.replace('images','csvs')
    urlTemp = urlTemp.replace('.tif','.csv')
    u = urlopen(urlTemp)
    data = u.read()
    u.close()
    filename = "data/dates.csv"
 
    with open(filename, "wb") as f :
        f.write(data)





def mainRoutine():

    #call the getdata function - inline
    #df,images = getdata()
    
    images = pd.read_csv('images.txt',header=None)
    
    for i in range(len(images)):
        
        print str(i)
        
        #pdb.set_trace()
        
        imageName = images.loc[i].tolist()
        
        imageName = imageName[0]
        
        url = 'https://dl.dropboxusercontent.com/u/53285700/images/' + imageName
        
        obs = imageName[7:25]
        
        getDropboxImage(url)
        
        process.doProcess(obs)
    
    

if __name__ == "__main__":
    mainRoutine()



