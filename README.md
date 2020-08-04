# Diagnosis
This is a diagnosis project which includes the diagnosis and analysis.

## Data

1. Transpose the log files to data matrix including alignment and encoding.
2. IF neccessary, Normalize the data

- Nearest neighbor: Set one state as Anchor and the others state are the nearest time based on the anchor time.
- Conv Box: Set the sliding box (e.g. 10s) and get the mean value of this sliding box.
- Latest update: Set one vector if any feature has updated, The difference of each vector is very small

The Diagnosis is labeled as time period.


## Model

- XGBoost
- LightGBM

