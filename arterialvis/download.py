"""Basic functions for reading in and processing various types of neuroimaging and neuromorphology data

Functions:
    download_zip
    download_swc
    download_dicom
"""
import os
import sys
from pathlib import Path
from dotenv import dotenv_values
import requests
from zipfile import ZipFile

def download_zip(zip_path, save_path, debug=True):
    """Download a zip file and extract into a directory
    
    Keyword arguments:
        zip_path -- the web URL where the zip is located (default uses .env config 'SWC_SOURCE')
        save_path -- the directory the unzipped files should be saved (default uses .env config 'SWC_SAVE')
        
    Thank you to Roman Podlinov for the streaming requests portion of this code, as detailed here:
    https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    """
    if debug:
        print(f'Downloading zip file from URL {zip_path} and unzipped to {save_path}')
    response = requests.get(zip_path, headers={'User-Agent': 'Arterial-Vis'}, stream=True) # Use the stream option to avoid overloading memory for large datasets
    try:
        os.mkdir(save_path) # Create the directory for saving the unzipped .swc files
    except:
        raise FileExistsError(f'WAIT! There is already a directory {save_path}. You should specify in the save_path in the function argument as something new; if you want to overwrite your existing data, please remove the existing save directory {save_path}.') # If the save directory already exists, stop and alert the user
    handle = open(os.path.join(save_path,'.zip'), "wb")
    for chunk in response.iter_content(chunk_size=512):
        if chunk:  # filter out keep-alive new chunks
            handle.write(chunk)
    handle.close()
    with ZipFile(os.path.join(save_path, '.zip'), 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall(save_path)
    os.remove(os.path.join(save_path,'.zip')) # Remove the zip file after it has been unzipped
        
def download_swc(zip_path=None, save_path=None, debug=True):
    """Download a zip file of neuromorphology (.swc files) and extract into a directory
    
    Keyword arguments:
        zip_path -- the web URL where the zip is located (default uses .env config 'SWC_SOURCE')
        save_path -- the directory the unzipped files should be saved (default uses .env config 'SWC_SAVE')
    """
    if (zip_path==None) or (save_path==None):
        config = dotenv_values(".env")
        if zip_path==None:
            zip_path = config['SWC_SOURCE']
        if save_path==None:
            save_path = Path(config['SWC_SAVE'])
    download_zip(zip_path, save_path, debug=debug)
    
        
def download_dicom(zip_path=None, save_path=None, debug=True):
    """Download a zip file of neuroimaging (.dcm files) and extract into a directory
    
    Keyword arguments:
        zip_path -- the web URL where the zip is located (default uses .env config 'DICOM_SOURCE')
        save_path -- the directory where the unzipped files should be saved (default uses .env config 'DICOM_SAVE')
    """
    if (zip_path==None) or (save_path==None):
        config = dotenv_values(".env")
        if zip_path==None:
            zip_path = config['DICOM_SOURCE']
        if save_path==None:
            save_path = Path(config['DICOM_SAVE'])
    download_zip(zip_path, save_path, debug=debug)
    
def append_dcm(dicom_path=None, debug=True):
    """Append ".dcm" DICOM file extension to files lacking any extension,
    searching recursively through the path directory
    (this is useful for some datasets which are exported without file extensions)
    
    Keyword arguments:
        dicom_path -- the directory where the unzipped files are saved (default uses .env config 'DICOM_SAVE')
    """
    if dicom_path==None:
        config = dotenv_values(".env")
        dicom_path = Path(config['DICOM_SAVE'])
        
    renamed_files = 0
    total_files = 0
    total_subdirs = 0
    
    for path, subdirs, files in os.walk(dicom_path):
        total_subdirs+=len(subdirs)
        for name in files:
            total_files+=1
            file_path = os.path.join(path,name)
            if name[:1] != ".":
                splitfile = os.path.splitext(name)
                if splitfile[1]=='':
                    new_name = os.path.join(path,name.lower()+'.dcm')
                    os.rename(file_path, new_name)
                    renamed_files +=1
    if debug:
        print(f'Renamed {renamed_files} out of {total_files} total files found in {dicom_path}')
        
def make_output_dir(name):
    """Create an output directory called 'output' and a subdirectory with the name specified
    
    Keyword arguments:
        name -- the name of the subdirectory to create in 'output/'
    
    Returns: Path of output directory
    """
    print(f'Absolute path: {os.path.abspath(os.curdir)}')
    try:
        os.mkdir('output')
    except:
        print('output directory already exists')
    try:
        outputdir = os.path.join('output',(name.split('.'))[0])
        os.mkdir(outputdir)
        print(f'Created an output directory at {outputdir}')
    except:
        print(f'{outputdir} directory already exists')
    return os.path.join(os.curdir, outputdir)