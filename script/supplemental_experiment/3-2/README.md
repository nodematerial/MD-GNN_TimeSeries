# 3.2. Modified model for one-step-ahead prediction with residual connection 

### Command

poetry run python3 build_vector.py
poetry run python3 lstm.py
poetry run python3 vec2pot.py
poetry run python3 mapping.py
poetry run python3 build_fullyconnected_pth.py
cp fully_connected.pth ../3-3
cp fully_connected.pth ../3-4
cp best_lstm.pth ../3-3
cp best_lstm.pth ../3-4