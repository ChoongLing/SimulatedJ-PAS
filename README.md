# SimulatedJ-PAS
First paper code

Code to deal with simulated J-PAS data (from CALIFA) and using ANN/CNNs to learn the relationship between J-PAS-like SEDs and spectroscopically derived ages/metallicities and then derivation of metallicity and age gradients. 

# Curent contents:
TrainZModel.py and TrainAgeModel.py: Code to train an ANN on a given training set. Saves model in a .h5 file and creates some plots showing quality of fit and fit history.
HyperasZ.py and HyperasAge.py: Code to run the 'Hyperas' package on a given dataset to choose the best architecture for a CNN for the dataset. Results are printed but not saved anywhere.

# References
Liew-Cain et al. (in prep)
