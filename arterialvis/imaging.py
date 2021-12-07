"""Data analytics functions for processing neuroimaging data

Functions:

    load_scan(path) -> array
    get_pixels_hu -> array
    extract_pixels -> array

    flatten -> array
    flat_wrapper -> array
    
    show_dist
    show_cluster_dist
    
    cluster -> dataframe
    cluster_wrapper -> dataframe
    cluster_modes -> dataframe
    find_middle_cluster -> integer
    filter_by_cluster -> dataframe
    get_HUrange -> tuple
    
    compare_scans
    
    mask -> array
    mask_wrapper -> array
    binary_mask -> array
    remove_islands -> array
    
    render_volume -> figure

"""

import os
import pydicom
from math import *
import numpy as np 
import pandas as pd
import pickle
import copy
import matplotlib.pyplot as plt
from itertools import chain
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.cluster import KMeans
import cv2
import ipyvolume as ipv


def load_scan(path):
    """Load DICOM data from a local directory.
    
    Keyword arguments:
    path -- the directory where the DICOM (.dcm) images are located
    """
    slices = [pydicom.dcmread(path + '/' + s) for s in               
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):
    """Extract an array of Hounsfield Unit values from a stack of scans
    
    Keyword arguments:
    scans -- an array of images
    """
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] = 0 # Set outside-of-scan pixels to 0
    intercept = scans[0].RescaleIntercept  # Convert to Hounsfield units (HU)
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def extract_pixels(path, debug=True):
    """Extract an array of pixels from a DICOM directory
    
    Keyword arguments:
    path -- the directory where the DICOM (.dcm) images are located
    """
    dicom = load_scan(path)
    pixels = get_pixels_hu(dicom)
    if debug==True: # If debugging, show the image
        plt.imshow(pixels[int(np.floor(len(pixels)/2))], cmap=plt.cm.binary)
    return pixels


def flatten(pixels, output=None, min_filter=-1024):
    """Flattens a 3D numpy array of pixels for each slice of the DICOM image into a single 1D array,
    excluding any values below the min_filter, and optionally saves the array as a pickle file
    
    Keyword arguments:
        pixels -- a 3D numpy array of pixels for each slice of the DICOM image
        output -- (optional) default: None; path of a location to save the pickle
    """
    flatten_list = list(chain.from_iterable(chain.from_iterable(pixels)))
    result = [x for x in flatten_list if x > min_filter]
    if output:
        pickle.dump(result, open(output, "wb"))
    return result

def flat_wrapper(pixels, output=None):
    """Wraps the flatten() function so as to check if the data has already been flattened
    and not repeat the process if it is not necessary.
    
    Keyword arguments:
        pixels -- a 3D numpy array of pixels for each slice of the DICOM image
        output -- (optional) default: None; the possible pre-existing save location to check
    """
    if output:
        try:
            flat = pd.read_pickle(output)
            return flat
        except:
            pass
    flat = flatten(pixels, output=output)
    return flat

def show_dist(flat, viewer='matplotlib', output=None):
    """Display the distribution of values in a 1D array
    
    Keyword arguments:
        viwer -- 'plotly' for interactive ; anything else (default: 'matplotlib') returns a static matplotlib histogram
        output -- (optional) default: None; (optionally, path and) filename *without extension* where to save the histogram
    """
    if viewer =='plotly':
        fig = go.Figure(data=[go.Histogram(x=flat)])
        if output: fig.write_html(output+'.html')
        fig.show()
    else:
        plt.clf()
        plt.hist(flat, 100, facecolor='blue', alpha=0.5)
        if output: plt.savefig(output+'.png')
        plt.show()
        
def show_cluster_dist(df, output=None):
    """Display the distribution of values by cluster
    
    Keyword arguments:
        df -- the pandas dataframe that stores the values, where (df.x is value) and (df.y is cluster_id)
        output -- (optional) default: None; (optionally, path and) filename *without extension* where to save the histogram
    """
    dfArr = []
    for l in list(set(df['y'])):
        dfArr.append(df[df['y']==l])
    colors = ['red', 'green', 'blue','orange','purple','yellow','brown','gray']
    i = 0
    plt.clf()
    for c in dfArr:
        plt.hist(c['x'], 100, facecolor=colors[i], alpha=0.5, label=i)
        i+=1
    if output:
        plt.savefig(output+'.png')
    plt.show()
    
def cluster(flat, k=3, output=None):
    """Run k-means pixel clustering on a 1D array and (optionally) save as CSV
    
    Keyword arguments:
        flat -- 1D array of values
        k -- number of clusters (default: 3)
        output -- (optional) default: None; location to save the CSV
    """
    km = KMeans(n_clusters=k)
    npArr = np.array(flat).reshape(-1,1)
    km.fit(npArr)
    label = km.fit_predict(npArr)
    df = pd.DataFrame(data={'x':flat, 'y':label})
    if output:
        df.to_csv(output, index=False)
        show_cluster_dist(df)
    return df

def cluster_wrapper(output=None, k=3, flat=None, pixels=False):
    """Wraps the flatten() and cluster() functions
    
    Keyword arguments:
        pixels -- (optional) a 3D numpy array of pixels for each slice of the DICOM image 
        flat -- (optional; required if 'pixels' not provided) 1D array of values
        k -- number of clusters (default: 3)
        output -- (optional) default: None; location to save the CSV
    """
    if flat is None:
        try:
            flat = flatten(pixels)
        except:
            print('Error! If no flattened array is provided, you must supply a pixels 3D array to flatten')
    if output:
        try:
            clustered = pd.read_csv(output)
            return clustered
        except:
            pass
    clustered = cluster(flat, k=k, output=output)
    return clustered

def cluster_modes(df):
    """Find the most common value in each cluster
    
    Keyword arguments:
        df -- the dataframe generated by cluster(), where (df.x is value) and (df.y is cluster_id)
    """
    clusters = list(set(df['y']))
    modes = []
    for k in clusters:
        modes.append(df[df['y']==k]['x'].mode()[0])
    mdf = pd.DataFrame(data={'cluster':clusters, 'mode':modes})
    return mdf

def find_middle_cluster(df):
    """Select the cluster with the median mode
    (use the higher median instead of averaging when the number of clusters is even)
    
    Keyword arguments:
        df -- the dataframe generated by cluster(), where (df.x is value) and (df.y is cluster_id)
    """
    mdf = cluster_modes(df)
    median = mdf['mode'].median_high()
    k = int(mdf[mdf['mode']==median]['cluster'])
    return k

def filter_by_cluster(df, cluster=None):
    """Filter the dataframe to only include a single cluster;
    if cluster is not specified, use the middle cluster determined by find_middle_cluster()
    
    Keyword arguments:
        df -- the dataframe generated by cluster(), where (df.x is value) and (df.y is cluster_id)
        cluster -- (optional) the specific cluster for which to filter
    """
    if cluster is None:
        cluster = find_middle_cluster(df)
    filtered = df[df['y']==cluster]['x']
    return filtered

def get_HUrange(df, cluster=None):
    """Extract the Hounsfield Unit (HU) range of the cluster;
    if cluster is not specified, use the middle cluster determined by find_middle_cluster()
    
    Keyword arguments:
        df -- the dataframe generated by cluster(), where (df.x is value) and (df.y is cluster_id)
        cluster -- (optional) the specific cluster for which to filter
    """
    if cluster is None:
        cluster = find_middle_cluster(df)
    minHU = df[df['y']==cluster]['x'].min()
    maxHU = df[df['y']==cluster]['x'].max()
    return (minHU, maxHU)

def compare_scans(original, mask, output=None, viewer="plotly"):
    """Show a slice through the middle of two brain scans (3D numpy arrays) side-by-side
    and (optionally) save the comparison image
    
    Keyword arguments:
        original -- the first 3D numpy array to compare
        mask -- the second 3D numpy array to compare
        output -- (optional) default: None; (optionally, path and) filename *without extension* where to save the comparison image
        viewer --  'plotly' for interactive; anything else returns a static matplotlib histogram
         output -- (optional) default: None; (optionally, path and) filename *without extension* where to save the scan comparison image
    """
    midbrain = int(np.floor(len(original)/2))
    print(f'Midbrain: {midbrain}')
    if viewer=="plotly":
        fig = make_subplots(1, 2, subplot_titles=("Original",'Mask'))
        fig.add_trace(go.Heatmap(z=original[midbrain],colorscale = 'gray'), 1, 1)
        fig.add_trace(go.Heatmap(z=mask[midbrain],colorscale = 'gray'), 1, 2)
        fig.update_layout(height=400)
        fig.show()
        if output:
            fig.write_html(output+'.html')
    else:
        plt.figure()
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(original[midbrain], cmap=plt.cm.binary)
        axarr[1].imshow(mask[midbrain], cmap=plt.cm.binary)
        if output:
            plt.savefig(output+'.png')
        plt.show()

def mask(pixels, HUrange, output=None):
    """Mask a 3D numpy array by a specified range by pushing pixels outside of the range
    1000HU below the minimum of the range
    
    Keyword arguments:
        pixels -- 3D numpy array
        HUrange -- a tuple of (min, max) HUrange
        output -- (optional) default: None; location + filename where to save the masked data
    """
    mask = copy.deepcopy(pixels)
    i=0
    for img in mask:
        j=0
        for row in img:
            mask[i][j] = [HUrange[0]-1000 if ((x < HUrange[0]) or (x > HUrange[1])) else x for x in row]
            j+=1
        i+=1
    if output:
        pickle.dump(mask, open(output, "wb"))
    return mask

def mask_wrapper(pixels, output=None, HUrange=None, df=False):
    """Wrapper for the mask() function which extracts an HUrange from the dataframe if there is no range specified
    and optionally checks to see if the mask data has already been pre-computed and saved in a specified location
    
    Keyword arguments:
        pixels -- 3D numpy array of values from the DICOM image stack
        output -- (optional) default: None; location + filename where to check for (or save) the masked data
        HUrange -- (optional) a custom Hounsfield Units range for masking
        df -- (optional, required if HUrange is 'None') dataframe from which to extract HUrange
    """
    if HUrange is None:
        if df is not None:
            HUrange = get_HUrange(df)
        else:
            print('Error! Must supply HUrange OR df')
    try:
        masked = pd.read_pickle(output)
    except:   
        masked = mask(pixels, HUrange, output=output)
    return masked

def binary_mask(pixels, maskRange):
    """Generate a binary mask from a 3D numpy array according to a specified range
    
    Keyword arguments:
        pixels -- 3D numpy array
        maskRange -- tuple of (min, max) Hounsfield unit range
    """
    binary = copy.deepcopy(pixels)
    i=0
    for img in pixels:
        j=0
        for row in img:
            binary[i][j] = [1 if (maskRange[0] < x < maskRange[1]) else 0 for x in row]
            j+=1
        i+=1
    compare_scans(pixels, binary, viewer='plotly')
    return binary


def remove_islands(pixels, output=None, k=3):
    """Generate a new 3D numpy array which removes islands using OpenCV's 'opening' function
    
    Keyword arguments:
        pixels -- 3D numpy array
        k -- square kernel size in pixels for opening; default: 3, which returns a kernel of size 3x3
        output -- (optional) where to save the pickle of the generated array
    """
    kernel = np.ones((k, k), np.float32)
    opening = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, kernel)
    compare_scans(pixels, opening, viewer='plotly')
    if output:
        pickle.dump(opening, open(output, "wb"))
    return opening

def render_volume(pixels):
    """Render a volume visualization of the 3D numpy array using iPyVolume
    
    Keyword arguments:
        pixels -- 3D numpy array
    """
    ipv.figure()
    ipv.volshow(pixels, opacity=0.03)
    ipv.view(-30, 40)
    ipv.show()
    return ipv.figure