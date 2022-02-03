# ArterialVis

*Public-facing repository of work-in-progress library for image processing, segmentation, 3D to 2D network abstraction of cerebrovasculature, and interactive visualization of planar graph embedding for detection of neurological anomalies*

![3D to 2D Animation Example](https://github.com/janeadams/arterial-vis/blob/main/documentation/full_graph_animation.gif?raw=true)

There are two ends to this project which have yet to be connected: the first, to segment neuroimaging data and extract morphology; the second, to provide interactive animations for 2D graph embedding of the extracted 3D morphology. We include both halves of this workflow for documentary purposes, but the first half (segmentation) remains incomplete, while the pilot of the project and the focus of this repository uses pre-existing morphologized data for 2D embedding and interactive demonstration.

Please note that at any time, you can access module and function descriptions by accessing the docstring using the format
`print(<MODULE OR FUNCTION>.__doc__)`.

## Getting Started

We recommend using a virtual environment to keep packages organized. There are two recommended ways to do this from the command line; either using `conda` or `venv` and `pip`:

### Setting up a virtual environment with `conda`
(if using Windows, ensure these commands are run in conda shell, not PowerShell)

Set up your virtual environment with:
`conda create --name arterialenv -f environment.yml python=3.8.3`

If you run into issues with the `environment.yml` file, you may need to use pip:
try just running `conda create --name arterialenv python=3.8.3` and then use pip to install requirements with `pip install -r requirements.txt`

**Note:** *There is a known issue with iPyVolume in Python 3.10; we are using Python 3.8.3 because it appears to be stable.*

Activate the environment with:
`conda activate arterialenv`

If you're having trouble viewing your virtual environments, run
- `conda info --envs` on Linux with conda
- `lsvirtualenv -l` on Windows

In order to view your virtual environment in Jupyter Notebooks:
`python -m ipykernel install --user --name arterialenv --display-name "ArterialVis environment"`

### OR: Setting up a virtual environment with `venv` and `pip`

If you haven't already, install `venv` with:
`pip install virtualenv`

Set up your virtual environment with:
`python -m venv arterialenv`

Activate the environment with:
- `activate arterialenv`, or, if this doesn't work, try:
- `source arterialenv/bin/activate` on Linux machines
- `venv\Scripts\activate arterialenv` on Windows machines
- `. .\arterialenv\Scripts\activate.ps1` in PowerShell

Install required packages with:
`pip install -r requirements.txt`

### Run jupyter notebook
Either by launching from the Anaconda GUI, or running `jupyter notebook` from the command line in the project directory


## Downloading sample data

For the purposes of demonstration, we use publicly available data, which can be downloaded using the steps described [here](#demodown).

To download and unzip web-hosted data from a custom zip route and/or save path, use: `download.download_zip(<ZIP_PATH>, <SAVE_PATH>)`.

All data processing functions are stored in `arterialvis/download.py`.

### <a name="demodown"></a>Download all
The functions to download neuroimaging and neuromorphology demo datasets, and append the ".dcm" file extension to the proper DICOM images in the neuroimaging dataset, are bundled together in the `import_data.py` file. To collect all necessary demo data, simply run `python download_sample_data.py` from the command line inside the virtual environment.

**If you are having trouble downloading sample data using this method, you can also manually download the data using the information saved in the `.env` file:**

|            | URL | Save Location |
|------------|-----|---------------|
| Imaging    |   [Zenodo](https://zenodo.org/record/16956/files/DICOM.zip)  | ./dicom_files/   |
| Morphology |   [BraVa](http://cng.gmu.edu/brava/files/swc_files.zip)  | ./swc_files/     |

or change the .env file accordingly.

**NOTE:** *This is the only downloading step necessary in demo execution of this package. Continue on to README instructions for imaging and segmentation [here](#imaging) or morphology and graphing [here](#graphing).

### Neuromorphology
For network visualization, we use extracted The Brain Vasculature (BraVa) neuromorphology data provided by the Kraskow Institute at George Mason University. Information about this dataset can be found [here](http://cng.gmu.edu/brava/home.php). The URL to the zip of the morphology data is located in the package `.env` file as `SWC_SOURCE` and is unzipped by default into the location specified `SWC_SAVE`, which is `swc_files`.

To download the default neuromorphology data, use:
`fetch.download_swc()`

### Neuroimaging
For image rendering and segmentation, we use an MRI DICOM data set provided by the Radiology Department at the Macclesfield General Hospital documented [here](https://zenodo.org/record/16956).

To download the default neurimaging data, use:
`download.download_dicom()`

### Custom data
Specify a custom URL and/or save path with:
```
download.download_zip(
    zip_path = $URL,
    save_path = $NEW_LOCAL_DIRNAME)
```

## <a name="imaging"></a>Imaging & Segmentation Module

**Note: This portion of the library is incomplete.** Functions exist for reading, clustering, masking, removing islands, and rendering volumes. Because masking and island removal alone are insufficient for acceptable segmentation, functions do not yet exist for centerline extraction and exporting of morphology. Please refer to the following section, "Morphology", for 2D embedding of morphological structures from pre-existing segmented and morphologized data from [BraVa](http://cng.gmu.edu/brava/home.php). You can skip ahead to README instructions for morphology and graphing [here](#graphing).

![Masking example](https://github.com/janeadams/arterial-vis/blob/main/documentation/compare_scans_masked.png?raw=true)

### Data Format
The ArterialVis imaging and segmentation workflow is designed to use DICOM images. By convention, DICOM images are stored in directories, where each sequentially enumerated image corresponds to an adjacent slice in the brain. ArterialVis reads DICOM images into 3D arrays, wherein the first level of the array corresponds to each slice, and the subsequent two levels correspond to the X and Y coordinates of each image.

### Running the Notebook
Ensuring your `arterialenv` environment is active, start Jupyter Notebooks with `jupyter notebook`. Step through each cell in `segmentation.ipynb` to view outputs.

### Module Structure
All image processing and segmentation functions are stored in `arterialvis/imaging.py` and can be imported using `from arterialvis import imaging` to use the function call format `imaging.<FUNCTION>()` or `from arterialvis.imaging import *` to import all functions and simply use the function call format `<FUNCTION>()`.

### Future Work
Ultimately, the goal of the ArterialVis imaging and segmentation workflow is to convert DICOM image stacks (directories of `.dcm` files) into a single morphology file (ending in `*swc`).

## <a name="morphology"></a>Morphology & Graphing Module

![Morphology comparison example](https://github.com/janeadams/arterial-vis/blob/main/documentation/comparison_dashboard.png?raw=true)

### Data Format
The ArterialVis morphology and graphing module takes `*.swc` files as input, and outputs interactive interfaces for exploring 3D morphological structure and animation from 3D spatial positioning to 2D abstracted graph embedding using multiple layout algorithms.

### Module Structure
All morphological and graphing functions are stored in `arterialvis/morphology.py` and can be imported using `from arterialvis import morphology` to use the function call format `morphology.<FUNCTION>()` or `from arterialvis.morphology import *` to import all functions and simply use the function call format `<FUNCTION>()`.

![Sparse animation](https://github.com/janeadams/arterial-vis/blob/main/documentation/sparse_animation.gif?raw=true)

We recommend testing all morphological embeddings on sparsified graphs before moving to animation of the complete graph; generating a network layout for the complete vascular tree can take an average of 15 minutes on an ordinary personal computer.

![Complete animation](https://github.com/janeadams/arterial-vis/blob/main/documentation/full_graph_animation.gif?raw=true)
