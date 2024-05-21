
cd 3-0
poetry run python3 train.py # Train MD-GNN model
poetry run python3 infer.py # predict the potential energy of the test dataset by using the trained MD-GNN model
poetry run python3 mapping.py # mapping the ground truth and predicted potential energy of the test dataset
cp best_gnn.pth ../3-1
cp prediction.csv ../3-1
cp best_gnn.pth ../3-2
cp prediction.csv ../3-2
cp prediction.csv ../3-3/predictions
cp prediction.csv ../3-4/longtime_predictions
 
cd ../3-1
poetry run python3 build_vector.py
poetry run python3 lstm.py
poetry run python3 vec2pot.py
poetry run python3 mapping.py
 
cd ../3-2
poetry run python3 build_vector.py
poetry run python3 lstm.py
poetry run python3 vec2pot.py
poetry run python3 mapping.py
poetry run python3 build_fullyconnected_pth.py
cp fully_connected.pth ../3-3
cp fully_connected.pth ../3-4
cp best_lstm.pth ../3-3
cp best_lstm.pth ../3-4


cd ../3-3
poetry run python3 recursive.py
(cd predictions; poetry run python3 split_prediction_csv.py)
poetry run python3 mapping.py
 
cd ../3-4
poetry run python3 longtime_recursive.py
(cd longtime_predictions; poetry run python3 split_prediction_csv.py)
poetry run python3 longtime_mapping.py