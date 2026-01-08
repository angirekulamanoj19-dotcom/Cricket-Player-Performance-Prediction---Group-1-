Cricket Player Performance Prediction

Data Processing (Up to Normalization)

This repository contains data preparation work for the IPL cricket dataset.

1)Completed Tasks
- Loaded IPL ball-by-ball data
- Cleaned the dataset (handled missing values, corrected formats)
- Created new useful features (total runs, wicket indicators)
- Normalized numerical columns
- Saved cleaned and normalized datasets

2)Folder Structure
- data/
  - IPL_ball_by_ball_updated.csv (raw data)
  - processed_deliveries.csv (cleaned data)
  - processed_deliveries_normalized.csv (normalized data)
- src/
  - data_cleaning.py (data cleaning and normalization script)
- notebooks/
  - 01_EDA.ipynb (initial exploration, optional)
- README.md

3)How to Run
```bash
python src/data_cleaning.py
