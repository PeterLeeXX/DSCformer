# 🚀DSCformer
 DSCformer: Lightweight model for predicting soil nitrogen content using VNIR-SWIR spectroscopy
## Dataset
 The following two datasets can be used:
|DataSet|URL|
| ----------- | ----------- |
|LUCAS 2015 TOPSOIL data|https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data|
|LUCAS 2009 TOPSOIL data **(Recommended)**|https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data|

## Requirements
```
einops==0.8.0
matplotlib==3.2.2
numpy==1.19.5
pandas==1.1.5
ptflops==0.7
scikit_learn==0.23.2
scipy==1.5.4
tensorboardX==2.4
tensorboardX==2.6.2.2
thop==0.1.1.post2209072238
torch==1.9.1+cu111
torchsummary==1.5.1
tqdm==4.61.2
```
## Usage
- After downloading the code locally, you can run the example by directly running the **train.py** file, the dataset used in the example is in the test data folder.
- The **model.py** file contains the DSCformer model. You can change the model configuration by configuring the incoming parameters of the **dsc** function, including Token Mixer, Patch Size, DSCers_num, and so on.
- The file **model_analysis.py** contains a series of evaluations of the model, which can be run to obtain a series of information about the model, including the specific structure, run time, etc..
