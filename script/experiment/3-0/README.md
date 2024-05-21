# 3.0. Training MD-GNN model

### Command

poetry run python3 train.py # Train MD-GNN model
poetry run python3 infer.py # predict the potential energy of the test dataset by using the trained MD-GNN model
poetry run python3 mapping.py # mapping the ground truth and predicted potential energy of the test dataset
cp best_gnn.pth ../3-2
cp prediction.csv ../3-2
cp prediction.csv ../3-3/predictions
cp prediction.csv ../3-4/longtime_predictions