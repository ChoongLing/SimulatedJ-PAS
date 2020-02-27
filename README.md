# Simulated J-PAS
First paper code

Code to deal with simulated J-PAS data (from CALIFA) and using ANN/CNNs to learn the relationship between J-PAS-like SEDs and spectroscopically derived ages/metallicities and then derivation of metallicity and age gradients. 

# Curent contents:
JoinData.py: Code to unite the spectral and positional data at each spaxel into a single file.

HyperasZ.py and HyperasAge.py: Code to run the 'Hyperas' package on a given dataset to choose the best architecture for a CNN for the dataset. Results are printed but not saved anywhere.

BestANN_Z.py and BestANN_Age.py: Code to train an ANN on a given training set. Saves model in a .h5 file and creates some plots showing quality of fit and fit history.

TrainZModel.py and TrainAgeModel.py: As with BestANN*.py above, but trains a CNN on a dataset and saves the trained network.

TestingModels.py: Code that takes a trained model saved as a .h5 file and tests it on a given dataset. Creates plots of the quality of predictions, predicted and spectroscopic code against radius and a gradient (with errors) determined for each galaxy and then a summary for the full dataset.

# References
Liew-Cain et al. 2020 (submitted; preprint at https://arxiv.org/abs/2002.08278)
