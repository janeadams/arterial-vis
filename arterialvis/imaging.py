"""Data analytics functions for processing neuroimaging data

Functions:

    parse_volumes -> dataframe
    find_largest_volume -> tuple
    load_scan(path) -> array
    get_pixels_hu -> array
    show_scan
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
from dotenv import dotenv_values
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

def parse_volumes(dicom_path=None, debug=True):
    """Extract all volumes from the selected DICOM directory and return the file paths
    
    Keyword arguments:
        path -- the directory where *ALL* the DICOM (.dcm) images are located
     
    Returns: Dataframe of metadata about the volumes
    """
    if (dicom_path==None):
        config = dotenv_values(".env")
        dicom_path = config['DICOM_SAVE']
    volumes = {}
    for path, subdirs, files in os.walk(dicom_path):
        for name in files:
            file_path = os.path.join(path,name)
            splitfile = os.path.splitext(name)
            vol = ('/').join(file_path.split('/')[:-1])
            if splitfile[1]=='.dcm':
                if vol not in volumes:
                    volumes[vol]=[]
                else:
                    volumes[vol].append(name)
    df = pd.DataFrame()
    df['path']=list(volumes.keys())
    df['files']=list(volumes.values())
    df.index = [(path.split('/'))[-1] for path in df['path']]
    df['count']=[len(files) for files in df['files']]
    if debug:
        print(df.drop(columns=['files']))
        print(f'\nThe volume with the highest slice count can be found at: \n {df[df["count"] == df["count"].max()]["path"][0]}')
    return df

def find_largest_volume(dicom_path=None, debug=True):
    """Find the volume with the greatest number of slices (for demonstration)
    
    Keyword arguments:
        dicom_path -- the directory where *ALL* the DICOM (.dcm) images are located
    
    Returns: Tuple of (path, name) of first largest volume in the dicom path
    """
    volumes = parse_volumes(dicom_path=dicom_path, debug=debug)
    path = volumes[volumes["count"] == volumes["count"].max()]["path"][0]
    name = list(volumes[volumes["count"] == volumes["count"].max()].index)[0]
    return path, name


def load_scan(path):
    """Load DICOM data from a local directory.
    
    Keyword arguments:
        path -- the directory where the target volume's DICOM (.dcm) images are located
        
    Returns: 3D pixel array of DICOM slices
    """
    slices = [pydicom.dcmread(os.path.join(path,s)) for s in               
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    return slices

def get_pixels_hu(slices):
    """Extract an array of Hounsfield Unit values from a stack of scans
    Source: https://hengloose.medium.com/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed
    Author: Franklin Heng
    
    Keyword arguments:
        slices -- an array of images
    
    Returns: 3D numpy array of HU values
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0 # Set outside-of-scan pixels to 0
    intercept = slices[0].RescaleIntercept if 'RescaleIntercept' in slices[0] else -1024  # Convert to Hounsfield units (HU)
    slope = slices[0].RescaleSlope if 'RescaleSlope' in slices[0] else 1
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def show_scan(pixels, viewer='plotly', output=False):
    """Render an image of the scan by slicing through the middle of the 3D array
    
    Keyword arguments:
        pixels -- a 3D numpy array of pixels for each slice of the DICOM image
        output --  (optional) default: None; path of a location to save the image
        
    Returns: Nothing; shows image
    """
    midbrain = int(np.floor(len(pixels)/2))
    fig = go.Figure(
        data=go.Heatmap(z=pixels[midbrain],
                        colorscale = 'gray'),
    )
    if output:
        head, tail = os.path.split(output)
        htmlsavepath = os.path.join(head, tail+'.html')
        fig.write_html(htmlsavepath)
        
        pngsavepath = os.path.join(head, tail+'.png')
        plt.imsave(pngsavepath, pixels[midbrain], cmap=plt.cm.binary)
    if viewer.lower() =='plotly':
        fig.show()
        plt.ioff()
        plt.close()
    else: 
        plt.imshow(pixels[midbrain], cmap=plt.cm.binary)
        plt.show()

def extract_pixels(path, debug=True):
    """Extract an array of pixels from a DICOM directory
    
    Keyword arguments:
        path -- the directory where the target volume's DICOM (.dcm) images are located
        
    Returns: 3D numpy array of HU values
    """
    dicom = load_scan(path)
    pixels = get_pixels_hu(dicom)
    if debug==True: # If debugging, show the image
        show_scan(pixels, output=path)
    return pixels


def flatten(pixels, output=False, min_filter=-1024):
    """Flattens a 3D numpy array of pixels for each slice of the DICOM image into a single 1D array,
    excluding any values below the min_filter, and optionally saves the array as a pickle file
    
    Keyword arguments:
        pixels -- a 3D numpy array of pixels for each slice of the DICOM image
        output -- (optional) default: None; path of a location to save the pickle
        
    Returns: flattened 1D array of HU values greater than the min_filter value
    """
    flatten_list = list(chain.from_iterable(chain.from_iterable(pixels)))
    result = [x for x in flatten_list if x > min_filter]
    if output:
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output, f'flattened.pkl'), 'wb') as f:
            pickle.dump(result, f)
    return result

def flat_wrapper(pixels, output=False):
    """Wraps the flatten() function so as to check if the data has already been flattened
    and not repeat the process if it is not necessary.
    
    Keyword arguments:
        pixels -- a 3D numpy array of pixels for each slice of the DICOM image
        output -- (optional) default: None; the possible pre-existing save location to check
    
    Returns: flattened 1D array of HU values
    """
    if output:
        try:
            flat = pd.read_pickle(os.path.join(output, f'flattened.pkl'))
            return flat
        except:
            pass
    flat = flatten(pixels, output=output)
    return flat

def show_dist(flat, viewer='plotly', output=False):
    """Display the distribution of values in a 1D array
    
    Keyword arguments:
        flat -- flattened 1D array of values
        viewer -- 'plotly' (case-insensitive) for interactive ; anything else (default: 'mpl') returns a static matplotlib histogram
        output -- (optional) default: None; (optionally, path and) filename *without extension* where to save the histogram
    
    Returns: Nothing; shows image
    """
    fig = go.Figure(data=[go.Histogram(x=flat)])
    plt.clf()
    plt.hist(flat, 100, facecolor='blue', alpha=0.5) 
    if output:
        try: os.makedirs(output)
        except: pass
        fig.write_html(os.path.join(output,'pixel_distribution.html'))
        plt.savefig(os.path.join(output,'pixel_distribution.png'))
    if viewer.lower() =='plotly':
        fig.show()
        plt.ioff()
        plt.close()
    else: plt.show()
        
def show_cluster_dist(df, viewer='plotly', output=False):
    """Display the distribution of values by cluster
    
    Keyword arguments:
        df -- the pandas dataframe that stores the values, where (df.x is value) and (df.y is cluster_id)
        viewer -- 'plotly' (case-insensitive) for interactive ; anything else returns a static matplotlib histogram
        output -- (optional) default: None; (optionally, path and) filename *without extension* where to save the histogram
    
    Returns: Nothing; shows image
    """
    
    fig = px.histogram(df, x="x", color="y")
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
        try: os.makedirs(output)
        except: pass
        fig.write_html(os.path.join(output,'cluster_distribution.html'))
        plt.savefig(os.path.join(output,'cluster_distribution.png'))
    if viewer.lower()=='plotly':
        fig.show()
        plt.ioff()
        plt.close()
    else: plt.show()
    
def cluster(flat, k=3, output=None):
    """Run k-means pixel clustering on a 1D array and (optionally) save as CSV
    
    Keyword arguments:
        flat -- 1D array of values
        k -- number of clusters (default: 3)
        output -- (optional) default: None; location to save the CSV
        
    Returns: Dataframe of metadata about the cluster index for each pixel
    """
    km = KMeans(n_clusters=k)
    npArr = np.array(flat).reshape(-1,1)
    km.fit(npArr)
    label = km.fit_predict(npArr)
    df = pd.DataFrame(data={'x':flat, 'y':label})
    if output:
        try: os.makedirs(output)
        except: pass
        df.to_csv(os.path.join(output,f'cluster_k{k}.csv'), index=False)
        show_cluster_dist(df, output=output)
    return df

def cluster_wrapper(pixels=False, flat=None, k=3, output=False):
    """Wraps the flatten() and cluster() functions
    
    Keyword arguments:
        pixels -- (optional) a 3D numpy array of pixels for each slice of the DICOM image 
        flat -- (optional; required if 'pixels' not provided) 1D array of values
        k -- number of clusters (default: 3)
        output -- (optional) default: None; location to save the CSV
        
    Returns: Dataframe of metadata about the cluster index for each pixel
    """
    if flat is None:
        try:
            flat = flatten(pixels)
        except:
            print('Error! If no flattened array is provided, you must supply a pixels 3D array to flatten')
    if output:
        try:
            clustered = pd.read_csv(os.path.join(output,f'cluster_k{k}.csv'))
            return clustered
        except:
            pass
    clustered = cluster(flat, k=k, output=output)
    return clustered

def cluster_modes(df):
    """Find the most common value in each cluster
    
    Keyword arguments:
        df -- the dataframe generated by cluster(), where (df.x is value) and (df.y is cluster_id)
        
    Returns: Dataframe of metadata about the modes for each cluster
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
    
    Returns: Index of the cluster with the modal value that is the median out of all the cluster modes
    """
    mdf = cluster_modes(df)
    median = mdf['mode'].median()
    k = int(mdf[mdf['mode']==median]['cluster'])
    return k

def filter_by_cluster(df, cluster=None):
    """Filter the dataframe to only include a single cluster;
    if cluster is not specified, use the middle cluster determined by find_middle_cluster()
    
    Keyword arguments:
        df -- the dataframe generated by cluster(), where (df.x is value) and (df.y is cluster_id)
        cluster -- (optional) the specific cluster for which to filter
        
    Returns: Filtered dataframe of pixel values and their cluster index
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
    
    Returns: Tuple of (minHU, maxHU)
    """
    if cluster is None:
        cluster = find_middle_cluster(df)
    minHU = df[df['y']==cluster]['x'].min()
    maxHU = df[df['y']==cluster]['x'].max()
    return (minHU, maxHU)

def compare_scans(baseline, compare, viewer="plotly", output=False):
    """Show a slice through the middle of two brain scans (3D numpy arrays) side-by-side
    and (optionally) save the comparison image
    
    Keyword arguments:
        original -- the first 3D numpy array to compare
        mask -- the second 3D numpy array to compare
        viewer --  'plotly' for interactive; anything else returns a static matplotlib histogram
         output -- (optional) default: None; (optionally, path and) filename *without extension* where to save the scan comparison image
         
    Returns: Nothing; shows image
    """
    midbrain = int(np.floor(len(baseline)/2))
    print(f'Midbrain: {midbrain}')

    fig = make_subplots(1, 2, subplot_titles=("Baseline",'Compare'))
    fig.add_trace(go.Heatmap(z=baseline[midbrain],colorscale = 'gray'), 1, 1)
    fig.add_trace(go.Heatmap(z=compare[midbrain],colorscale = 'gray'), 1, 2)
    fig.update_layout(height=400, width=800)
    plt.figure()
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(baseline[midbrain], cmap=plt.cm.binary)
    axarr[1].imshow(compare[midbrain], cmap=plt.cm.binary)
    if output:
        try: os.makedirs(output)
        except: pass
        fig.write_html(os.path.join(output, 'compare_scans.html'))
        plt.savefig(os.path.join(output, 'compare_scans.png'))
    if viewer=="plotly":
        fig.show()
        plt.ioff()
        plt.close()
    else:
        plt.show()

def mask(pixels, HUrange, output=False, debug=True):
    """Mask a 3D numpy array by a specified range by pushing pixels outside of the range
    1000HU below the minimum of the range
    
    Keyword arguments:
        pixels -- 3D numpy array
        HUrange -- a tuple of (min, max) HUrange
        output -- (optional) default: None; location + filename where to save the masked data
        
    Returns: 3D numpy array of masked pixel values
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
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output, f'mask_{HUrange[0]}-{HUrange[1]}.pkl'), 'wb') as f:
            pickle.dump(mask, f)
    if debug:
        show_scan(mask, output=output)
    return mask

def mask_wrapper(pixels, output=None, HUrange=None, df=True, debug=True):
    """Wrapper for the mask() function which extracts an HUrange from the dataframe if there is no range specified
    and optionally checks to see if the mask data has already been pre-computed and saved in a specified location
    
    Keyword arguments:
        pixels -- 3D numpy array of values from the DICOM image stack
        output -- (optional) default: None; location + filename where to check for (or save) the masked data
        HUrange -- (optional) a custom Hounsfield Units range for masking
        df -- (optional, required if HUrange is 'None') dataframe from which to extract HUrange
                
    Returns: 3D numpy array of masked pixel values
    """
    if HUrange is None:
        if df is not None:
            HUrange = get_HUrange(df)
        else:
            print('Error! Must supply HUrange OR df')
    try:
        masked = pd.read_pickle(os.path.join(output, f'mask_{HUrange[0]}-{HUrange[1]}.pkl'))
    except:
        masked = mask(pixels, HUrange, output=output, debug=debug)
    return masked

def binary_mask(pixels, maskRange, output=False, debug=True):
    """Generate a binary mask from a 3D numpy array according to a specified range
    
    Keyword arguments:
        pixels -- 3D numpy array
        maskRange -- tuple of (min, max) Hounsfield unit range
                
    Returns: 3D numpy array (binary) of masked pixel values
    """
    binary = copy.deepcopy(pixels)
    i=0
    for img in pixels:
        j=0
        for row in img:
            binary[i][j] = [1 if (maskRange[0] < x < maskRange[1]) else 0 for x in row]
            j+=1
        i+=1
    if output:
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output, f'binary-mask_{maskRange[0]}-{maskRange[1]}.pkl'), 'wb') as f:
            pickle.dump(mask,f)
    if debug: compare_scans(pixels, binary, viewer='plotly')
    return binary


def remove_islands(pixels, output=False, k=3):
    """Generate a new 3D numpy array which removes islands using OpenCV's 'opening' function
    
    Keyword arguments:
        pixels -- 3D numpy array
        k -- square kernel size in pixels for opening; default: 3, which returns a kernel of size 3x3
        output -- (optional) where to save the pickle of the generated array
            
    Returns: 3D numpy array (binary) of masked pixel values
    """
    kernel = np.ones((k, k), np.float32)
    opening = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, kernel)
    compare_scans(pixels, opening, viewer='plotly')
    if output:
        try: os.makedirs(output)
        except: pass
        with open(os.path.join(output, f'remove_islands_k{k}.pkl'), 'wb') as f:
            pickle.dump(opening, f)
    return opening