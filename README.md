# ArterialVis

*Public-facing repository of work-in-progress library for image processing, segmentation, 3D to 2D network abstraction of cerebrovasculature, and interactive visualization of planar graph embedding for detection of neurological anomalies*

There are two ends to this project which have yet to be connected: the first, to segment neuroimaging data and extract morphology; the second, to provide interactive animations for 2D graph embedding of the extracted 3D morphology. We include both halves of this workflow for documentary purposes, but the first half (segmentation) remains incomplete, while the pilot of the project and the focus of this repository uses pre-existing morphologized data for 2D embedding and interactive demonstration.

Please note that at any time, you can access module descriptions by running
`print($MODULE.__doc__)` or function descriptions with `print($FUNCTION.__doc__)`.

## Getting Started

We recommend using a virtual environment to keep packages organized. There are two recommended ways to do this from the command line; either using `conda` or `venv` and `pip`:

### Setting up a virtual environment with `conda`

Set up your virtual environment with:
`conda create --name $ENV_NAME -f environment.yml`

Activate the environment with:
`conda activate $ENV_NAME`

In order to view your virtual environment in Jupyter Notebooks:
`python -m ipykernel install --user --name $ENV_NAME --display-name "Virtual environment"`

### OR: Setting up a virtual environment with `venv` and `pip`

If you haven't already, install `venv` with:
`pip install virtualenv`

Set up your virtual environment with:
`python -m venv $ENV_DIR`

Activate the environment with:
`source $ENV_DIR/bin/activate`

## Downloading sample data

For the purposes of demonstration, we use publicly available data, which can be downloaded using the steps described [here](#demodown).

To download and unzip web-hosted data from a custom zip route and/or save path, use: `download.download_zip($ZIP_PATH, $SAVE_PATH)`.

All data processing functions are stored in `arterialvis/download.py`.

### <a name="demodown"></a>Download all
The functions to download neuroimaging and neuromorphology demo datasets, and append the ".dcm" file extension to the proper DICOM images in the neuroimaging dataset, are bundled together in the `import_data.py` file. To collect all necessary demo data, simply run `python import_data.py` from the command line inside the virtual environment.

**NOTE:** *This is the only downloading step necessary in demo execution of this package. Continue on to README instructions for imaging and segmentation [here](#imaging) or morphology and graphing [here](#graphing).

### Neuromorphology
For network visualization, we use extracted The Brain Vasculature (BraVa) neuromorphology data provided by the Kraskow Institute at George Mason University. Information about this dataset can be found [here](http://cng.gmu.edu/brava/home.php). The URL to the zip of the morphology data is located in the package `.env` file as `SWC_SOURCE` and is unzipped by default into the location specified `SWC_SAVE`, which is `swc_files`.

To download the default neuromorphology data, use:
`fetch.download_swc()`

### Neuroimaging
For image rendering and segmentation, we use an MRI DICOM data set of the head of a normal male human aged 52 provided by the Radiology Department at the Macclesfield General Hospital and documented [here](https://zenodo.org/record/16956). *Note that this workflow is incomplete and exists only to show the initial research steps towards segmentation and extraction of neuromorphology. You can skip ahead to README instructions for morphology and graphing [here](#graphing).*

To download the default neurimaging data, use:
`download.download_dicom()`

### Custom data
Specify a custom URL and/or save path with:
`download.download_zip(zip_path=$URL, save_path=$NEW_LOCAL_DIRNAME)`

## <a name="imaging"></a>Imaging & Segmentation Module

**Note:** *This portion of the library is incomplete.* Functions exist for reading, clustering, masking, removing islands, and rendering volumes. Because masking and island removal alone are insufficient for acceptable segmentation, functions do not yet exist for centerline extraction and exporting of morphology. Please refer to the following section, "Morphology", for 2D embedding of morphological structures from pre-existing segmented and morphologized data from [BraVa](http://cng.gmu.edu/brava/home.php).

The ArterialVis imaging and segmentation workflow is designed to use DICOM images. By convention, DICOM images are stored in directories, where each sequentially enumerated image corresponds to an adjacent slice in the brain. ArterialVis reads DICOM images into 3D arrays, wherein the first level of the array corresponds to each slice, and the subsequent two levels correspond to the X and Y coordinates of each image.

All image processing and segmentation functions are stored in `arterialvis/imaging.py` and can be imported using `from arterialvis import imaging` to use the function call format `imaging.$FUNCTION()` or `from arterialvis.imaging import *` to import all functions and simply use the function call format `$FUNCTION()`.

Ultimately, the goal of the ArterialVis imaging and segmentation workflow is to convert DICOM image stacks (directories of `.dcm` files) into a single morphology file (ending in `*swc`).

## <a name="morphology"></a>Morphology & Graphing Module

The ArterialVis morphology and graphing module takes `*.swc` files as input, and outputs interactive interfaces for exploring 3D morphological structure and animation from 3D spatial positioning to 2D abstracted graph embedding using multiple layout algorithms.

All morphological and graphing functions are stored in `arterialvis/morphology.py` and can be imported using `from arterialvis import morphology` to use the function call format `morphology.$FUNCTION()` or `from arterialvis.morphology import *` to import all functions and simply use the function call format `$FUNCTION()`.