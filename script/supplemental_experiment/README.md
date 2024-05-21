## Training MD-GNN TimeSeries Model with High-Temperature Dataset (supplemental experiment)

In this experiment_directory, as the supplementary experiment, we train the MD-GNN model by using the high-temperature training dataset. 
(1850K, 1900K, 1950K, 2000K, 2050K, 2100K)
Then we found that the MD-GNN TimeSeries model is not suitable for the high-temperature dataset.

We estimate that our feature extraction way couldn't extract the important features to express
the physical properties of the high-temperature dataset.

### Command

You can train and evaluate the MD-GNN model by running the following pipeline command in the same directory as this README file.

```
bash pipeline.sh
```
