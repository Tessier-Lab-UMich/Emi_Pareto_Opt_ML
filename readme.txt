1. Download the anaconda distribution of python from: https://www.anaconda.com/

2. Open the anaconda command prompt, navigate to where the code for this paper is stored on your computer using:

cd PATH\TO\CODE

and use the following commands to activate the environment:

conda env create -f environment.yml
conda activate emi_env

3. The attached folder includes scripts to import data, fit LDA models to binary labeled data, and correlated those models to continuous measurements of sequences. 
Each model type (onehot, physchem, or unirep) is included as its own script. The physchem and unirep models scripts additionally correlate LDA models with out-of-library sequences. 

Scripts can be run using any development environment (ex: Spyder, included with anaconda) or by typing

python SCRIPTNAME.py

into the anaconda command prompt. For example:

python onehot_models.py