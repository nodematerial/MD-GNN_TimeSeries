## Training MD-GNN Model

In this experiment_directory, as the supplementary experiment, we train the MD-GNN model by using the high-temperature training dataset. 
(1850K, 1900K, 1950K, 2000K, 2050K, 2100K)

### Command

poetry run python3 train.py # Train MD-GNN model
poetry run python3 infer.py # predict the potential energy of the test dataset by using the trained MD-GNN model
poetry run python3 mapping.py # mapping the ground truth and predicted potential energy of the test dataset
