# About
This directory contains reference Jupyter notebooks and utility code for the ["Distilling Interpretable Models into Human-Readable Code"](https://arxiv.org/abs/2101.08393) paper.
A release has been created for each version of the paper with the code and notebook state at that time.
Please reference the appropriate release and its accompanying datafiles for reproducibility.
After downloading the release you can follow the instruction below to run the notebooks.

# Install requirements

You may wish to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to simplify installing requirements.

```
cd <RELEASE_DIR>/pwlfit
pip install .
cd publications/Distilling_Interpretable_Models_into_Human_Readable_Code
pip install -r requirements.txt
```

# Prepare input directory
Each notebook takes an input directory that contains the datasets (which we do
not host) and the model scores as parquet files (found in DATA.zip).
For the COMPAS and FICO datasets we used the same datasets as the [NAM paper](https://arxiv.org/pdf/2004.13912.pdf). They provide a copy via data_utils.py from [their github](https://github.com/google-research/google-research/tree/master/neural_additive_models). Instead of using this interface we directly access the files via [gsutil](https://cloud.google.com/storage/docs/gsutil). Below are example commands for setting up the input directory for each dataset after downloading DATA.zip.



### COMPAS

```
unzip DATA.zip
gsutil cp -R gs://nam_datasets/data/recidivism/* DATA/COMPAS/
```

## FICO

```
unzip DATA.zip
gsutil cp -R gs://nam_datasets/data/fico/* DATA/FICO
```


## MSLR-WEB30K

[Download MSLR-WEB30K.zip](https://www.microsoft.com/en-us/research/project/mslr/).

```
unzip DATA.zip
unzip MSLR-WEB30K.zip
cp Fold1/* DATA/MSLR-WEB30K
```


# Run Jupyter and open notebooks

```
jupyter notebook
```
