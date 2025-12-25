# Cricket-Player-Performance-Prediction---Group-1-
## Week 1 & Week 2 – Data Acquisition, Cleaning, and EDA

## Overview
Weeks 1 and 2 of this project focus on understanding the IPL datasets, analyzing data quality, handling missing values, and performing exploratory data analysis (EDA). The objective is to prepare clean, reliable datasets that can be used for feature engineering and machine learning in later stages.

---

## Datasets Used
- **matches.csv** – Match-level details such as season, teams, venue, toss information, and winner.
- **deliveries.csv** – Ball-by-ball data containing runs, wickets, players involved, and over/ball information.

---

## Week 1 & 2 Tasks

### 1. Data Loading and Inspection
- Loaded both datasets using pandas.
- Examined dataset shapes, column names, and data types.
- Verified consistency of `match_id` between matches and deliveries datasets.

### 2. Missing Value Analysis
- Performed column-wise missing value analysis.
- Identified expected missing values:
  - Dismissal-related fields when no wicket occurred.
  - Extras-related fields when no extras were conceded.
- Identified and handled problematic missing values in core delivery columns.

### 3. Data Cleaning

#### Matches Dataset
- Filled missing `method` values with `Normal`.
- Filled missing `city` values with `Unknown`.
- Converted `date` column to datetime format.
- Standardized team names across seasons.
- Removed matches with no result (missing `winner`).

#### Deliveries Dataset
- Filled missing `fielder` values with `Not Applicable`.
- Filled missing `extras_type` values with `None`.
- Removed rows with missing critical fields such as:
  - over, ball, batter, bowler, run values, and wicket indicator.
- Standardized team names to match the matches dataset.

Cleaned datasets were saved as:
- `matches_clean.csv`
- `deliveries_clean.csv`

### 4. Exploratory Data Analysis (EDA)
Performed EDA to understand overall patterns and distributions:
- Matches played per season
- Top venues by number of matches
- Distribution of runs scored per ball
- Distribution of wickets per match
- Top batsmen by total runs
- Top bowlers by total wickets

These analyses highlight the high variability of T20 cricket and justify the use of machine learning models for performance prediction.

---

## Files Included
- `01_EDA.ipynb` – Contains data loading, cleaning steps, EDA visualizations, and observations.
- `data_cleaning.py` – Standalone, reproducible script for applying finalized data cleaning steps.

---

## Key Outcomes
- Cleaned and standardized IPL datasets.
- Clear understanding of data quality and distributions.
- Reproducible data preprocessing pipeline established.
- Strong foundation prepared for feature engineering and modeling in upcoming weeks.

---
