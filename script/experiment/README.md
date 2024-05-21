## Training MD-GNN TimeSeries Model and Evaluation.

In this experiment directory, we trained and evaluated the MD-GNN TimeSeries model. 

The dataset features MD simulation data under NPT ensemble conditions, with temperatures set to 1000K, 1100K, 1200K, 1300K, 1400K, 1500K, 1600K, 1700K, 1800K, and 1900K.

The calculation is divided in 5steps

1. Training MD-GNN model (referenced as [Prediction of potential energy profiles of molecular dynamic simulation by graph convolutional networks](https://www.sciencedirect.com/science/article/pii/S0927025623004421) -> 3.0.
2. One-step-ahead prediction using LSTM -> 3.1.
3. Modified model for one-step-ahead prediction with residual connection -> 3.2.
4. Multi-step ahead prediction by recursive time evolution of latent vectors -> 3.3.
5. Long time prediction by recursive time evolution of latent vectors -> 3.4.

### Command

You can train and evaluate the MD-GNN model by running the following pipeline command in the same directory as this README file.

```
bash pipeline.sh
```
