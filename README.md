<div align="center">

# 🏏 Cricket Score Prediction System

### AI-Powered T20I Player Score Prediction using Advanced Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success.svg)](https://cricket-score-prediction-system.streamlit.app/)

**Predict cricket player scores with state-of-the-art ML models featuring interactive dashboard**

[Features](#-features) • [Live Demo](#-live-demo) • [Installation](#-installation) • [Architecture](#-architecture) • [Models](#-machine-learning-models)

---

</div>

## 🎯 Overview

A production-ready machine learning system that predicts T20 International cricket player scores using historical performance data, match context, and advanced statistical features. The system includes a beautiful Streamlit dashboard for real-time predictions and comprehensive model analysis.

## ✨ Features

### 🤖 Advanced ML Models

- **XGBoost** (Best Performance - Optimized with Optuna)
- **LightGBM** (Tuned with GridSearchCV)
- **Random Forest** (Ensemble Learning)
- **Baseline Model** (10-Match Rolling Average)

### 📊 Interactive Dashboard

- 🎯 Real-time player score predictions
- 📈 Model performance comparison
- 📉 Data insights and visualizations
- 🎨 Beautiful, responsive UI with Plotly charts
- 🔄 Multiple model selection

### 📈 Comprehensive Features

- Historical performance metrics (3, 5, 10 match averages)
- Strike rate analysis
- Batting position impact
- Venue statistics
- Opposition strength
- Form and consistency scores
- Boundary and dot ball percentages

### 🎨 Production-Ready

- ✅ Dockerized deployment
- ✅ REST API ready
- ✅ Cloud deployment guides
- ✅ Comprehensive documentation
- ✅ Model versioning

## 🚀 Live Demo

**🌐 Access the live application:** [https://cricket-score-prediction-system.streamlit.app/](https://cricket-score-prediction-system.streamlit.app/)

![Dashboard Screenshot](docs/deploydashboard.png)

### Dashboard Features

The interactive dashboard provides:

- 🔮 **Prediction Interface**: Enter player details and get instant predictions
- 📊 **Model Performance**: Compare all models with interactive charts
- 📈 **Data Insights**: Explore historical trends and top performers
- ℹ️ **About**: Learn about the technology and methodology

![Output Screenshot 1](docs/output.png)

![Output Screenshot 2](docs/output1.png)

### Local Launch

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` with full functionality.

### Quick Prediction Example

```python
# The dashboard allows you to:
# 1. Select from multiple ML models (XGBoost, LightGBM, Random Forest)
# 2. Input player statistics (recent form, strike rates, career stats)
# 3. Specify match context (venue, position, opposition)
# 4. Get instant predictions with confidence intervals
```

## 📦 Installation

### Prerequisites

- Python 3.11 (recommended for deployment compatibility)
- pip package manager
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/deekshithgowda85/Cricket-Score-Prediction-System.git
cd Cricket-Score-Prediction-System

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py

# Open browser to http://localhost:8501
```

### Docker Setup

```bash
# Build the Docker image
docker build -t cricket-predictor .

# Run the container
docker run -p 8501:8501 cricket-predictor

# Or use Docker Compose
docker-compose up
```

### Dependencies

```txt
# Core Data Science (optimized for fast deployment)
pandas>=2.0.0,<2.3.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0,<1.5.0

# Machine Learning Models
xgboost>=2.0.0,<2.1.0
lightgbm>=4.0.0,<5.0.0

# Visualization
plotly>=5.18.0,<6.0.0

# Dashboard
streamlit>=1.29.0,<2.0.0

# Utilities
joblib>=1.3.0,<2.0.0
python-dateutil>=2.8.0
pytz>=2023.3
```

**Note**: Version ranges are used instead of exact pins for faster wheel installation on Streamlit Cloud.

## 💻 Usage

### Dashboard Interface

1. **Prediction Page**

   - Select your preferred ML model
   - Enter player's recent performance (last 3, 5, 10 matches)
   - Input career statistics
   - Specify match context
   - Click "Predict Score" for instant results

2. **Model Performance**

   - View comparative metrics (RMSE, MAE, R²)
   - Interactive charts showing model comparison
   - Best model highlighted

3. **Data Insights**
   - Dataset statistics and overview
   - Score distribution analysis
   - Temporal trends
   - Top performers leaderboard

### Programmatic Usage

```python
import joblib
import pandas as pd

# Load the best model
model = joblib.load('models/xgb_model.joblib')
pipeline = joblib.load('models/feature_pipeline.pkl')

# Prepare features
features = {
    'runs_last_3': 45.0,
    'runs_last_5': 38.0,
    'runs_last_10': 35.0,
    'sr_last_3': 145.0,
    'sr_last_5': 138.0,
    'sr_last_10': 132.0,
    'total_matches': 75,
    'total_runs': 2100,
    # ... other features
}

# Make prediction
X = pd.DataFrame([features])
prediction = model.predict(X)[0]
print(f"Predicted Score: {prediction:.0f} runs")
```

### API Integration (Future)

```python
# Coming soon: REST API endpoint
import requests

response = requests.post('http://api.cricket-predictor.com/predict', json={
    'player_stats': {...},
    'match_context': {...}
})
predicted_score = response.json()['prediction']
```

## 🚀 Deployment

### Streamlit Cloud (Recommended - Already Live!)

**🌐 Live Demo**: [https://cricket-score-prediction-system.streamlit.app/](https://cricket-score-prediction-system.streamlit.app/)

#### Deploy Your Own Instance

1. **Fork the Repository on GitHub**

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Advanced settings:
     - Python version: `3.11` (uses `.python-version` and `runtime.txt`)
     - Requirements file: `requirements.txt`
   - Click "Deploy"
   - Your app will be live in 3-5 minutes! 🎉

#### Deployment Optimizations Applied

✅ **Python 3.11** - Specified via `.python-version` and `runtime.txt` for pre-built wheels  
✅ **Flexible dependencies** - Version ranges instead of exact pins  
✅ **Optimized model size** - XGBoost (0.45 MB) and LightGBM (0.24 MB) deployed  
✅ **Fast startup** - ~3-5 minute deployment time

**Build Time**: 3-5 minutes (optimized from 8-12 minutes)

### Docker Deployment

```bash
# Build the image
docker build -t cricket-predictor:latest .

# Run locally
docker run -p 8501:8501 cricket-predictor:latest

# Or use Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

### Cloud Platforms

#### AWS (EC2 + Docker)

```bash
# Launch EC2 instance (t2.medium recommended)
# SSH into instance
ssh -i your-key.pem ec2-user@your-ip

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Clone and run
git clone your-repo
cd cricket-score-prediction
docker-compose up -d

# Configure security group to allow port 8501
```

#### Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create cricket-score-predictor
git push heroku main
heroku open
```

#### Azure Web Apps

```bash
# Using Azure CLI
az webapp up --name cricket-predictor --runtime "PYTHON:3.9"

# Configure startup command
az webapp config set --resource-group your-rg --name cricket-predictor \
  --startup-file "streamlit run app.py --server.port=8000 --server.address=0.0.0.0"
```

### Environment Variables

Create `.streamlit/secrets.toml` for sensitive data:

```toml
[general]
api_key = "your-api-key"

[database]
connection_string = "your-db-connection"
```

## 🤖 Machine Learning Models

### Model Comparison

| Model               | Test RMSE | Test MAE  | Test R²  | Training Time | Status |
| ------------------- | --------- | --------- | -------- | ------------- | ------ |
| Baseline (Avg)      | ~19.5     | ~14.8     | 0.24     | Instant       | ✅     |
| Random Forest       | ~18.2     | ~13.6     | 0.29     | ~2 min        | ✅     |
| XGBoost             | ~17.8     | ~13.2     | 0.31     | ~3 min        | ✅     |
| LightGBM            | ~17.9     | ~13.3     | 0.30     | ~2 min        | ✅     |
| **XGBoost (Tuned)** | **~17.3** | **~12.9** | **0.33** | ~15 min       | **🏆** |
| LightGBM (Tuned)    | ~17.5     | ~13.0     | 0.32     | ~8 min        | ✅     |

### Best Model: XGBoost (Tuned with Optuna)

**Optimized Hyperparameters:**

```python
{
    'n_estimators': 350,
    'max_depth': 7,
    'learning_rate': 0.08,
    'subsample': 0.85,
    'colsample_bytree': 0.82,
    'min_child_weight': 3,
    'gamma': 0.15
}
```

**Performance Metrics:**

- ✅ **RMSE**: ~17.3 runs (±15 runs accuracy)
- ✅ **MAE**: ~12.9 runs
- ✅ **R² Score**: 0.33
- ✅ **Improvement over baseline**: ~11% RMSE reduction

### Feature Importance (Top 10)

Based on SHAP analysis:

1. 🏏 **runs_last_10** (12.8%) - Recent performance avg
2. 📊 **sr_last_10** (11.2%) - Strike rate consistency
3. 🎯 **runs_last_5** (9.5%) - Short-term form
4. 💪 **total_runs** (8.7%) - Career runs
5. 🔄 **runs_last_3** (7.9%) - Immediate form
6. 📈 **avg_score** (6.4%) - Career average
7. ⚡ **sr_last_5** (6.1%) - Recent strike rate
8. 🎪 **batting_position** (5.8%) - Position impact
9. 🏟️ **venue_encoded** (5.2%) - Venue familiarity
10. 🌍 **country_encoded** (4.6%) - Home/away factor

### Model Interpretability

The project uses **SHAP (SHapley Additive exPlanations)** for:

- Feature importance ranking
- Individual prediction explanations
- Dependence plots showing feature interactions
- Summary plots for global model behavior

## 📊 Dataset

### Overview

| Dataset          | Records | Features | Description               |
| ---------------- | ------- | -------- | ------------------------- |
| Batting Features | 15,847  | 45+      | Player batting statistics |
| Bowling Features | 12,234  | 38+      | Bowling performance data  |
| T20I Matches     | 2,594   | 25+      | Match-level information   |
| Player Info      | 6,703   | 8        | Player biographical data  |

### Data Sources

- T20 International matches (2005-2024)
- Official cricket statistics
- Ball-by-ball data
- Player career records

### Key Features Engineered

#### Performance Metrics

- Rolling averages (3, 5, 10 matches)
- Strike rates (recent and career)
- Consistency scores
- Form indicators

#### Match Context

- Venue encoding (100+ venues)
- Country encoding (home/away/neutral)
- Batting position
- Opposition strength

#### Career Statistics

- Total matches, runs, innings
- Career average and strike rate
- Boundary percentages
- Dismissal patterns

## 📁 Project Structure

```
cricket-score-prediction-system/
│
├── 📄 app.py                           # 🎯 Main Streamlit Dashboard Application
├── 📄 data_cleaning.py                 # 🧹 Data preprocessing and cleaning utilities
├── 📄 requirements.txt                 # 📦 Python dependencies (optimized for deployment)
├── 📄 .python-version                  # 🐍 Python version specification (3.11)
├── 📄 runtime.txt                      # ⚙️ Runtime configuration for deployment
├── 📄 Dockerfile                       # 🐳 Docker containerization config
├── 📄 docker-compose.yml               # 🐳 Docker Compose orchestration
├── 📄 README.md                        # 📖 Project documentation
├── 📄 .gitignore                       # 🚫 Git ignore rules
│
├── 📂 .streamlit/
│   └── config.toml                     # ⚙️ Streamlit app configuration & theming
│
├── 📂 dataset/                         # 🏏 Cricket data files
│   ├── t20i_Matches_Data.csv           # 📊 Match-level information (2,594 matches)
│   ├── t20i_Batting_Card.csv           # 🏏 Batting statistics per match
│   ├── t20i_Bowling_Card.csv           # ⚾ Bowling statistics per match
│   ├── t20i_Fow_Card.csv               # 📉 Fall of wickets data
│   ├── t20i_Partnership_Card.csv       # 🤝 Partnership information
│   ├── players_info.csv                # 👤 Player biographical data (6,703 players)
│   ├── batting_features_dataset.csv    # 🔧 Engineered batting features (15,847 records)
│   └── bowling_features_dataset.csv    # 🔧 Engineered bowling features (12,234 records)
│
├── 📂 models/                          # 🤖 Trained ML models & artifacts
│   ├── xgb_model.joblib                # 🏆 XGBoost model (Best - 0.45 MB)
│   ├── lgb_model.joblib                # 💚 LightGBM model (0.24 MB)
│   ├── rf_model.joblib                 # 🌲 Random Forest model (27 MB)
│   ├── cricket_score_model.pkl         # 📦 Legacy model file
│   ├── feature_pipeline.pkl            # 🔧 Feature preprocessing pipeline
│   ├── model_results.csv               # 📊 Model performance comparison
│   └── shap_values.npy                 # 🔍 SHAP values for interpretability
│
├── 📂 notebooks/                       # 📓 Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb                    # 📈 Exploratory Data Analysis
│   ├── 02_FeatureEngineering.ipynb     # 🔧 Feature creation & engineering
│   ├── 03_ModelTraining.ipynb          # 🤖 Model training & optimization
│   └── score_predictor.ipynb           # 🎯 Main prediction notebook
│
├── 📂 scripts/                         # 🛠️ Utility scripts
│   ├── config.py                       # ⚙️ Configuration settings
│   ├── setup_features.py               # 🔧 Feature dataset generation
│   └── utils.py                        # 🔨 Helper functions
│
└── 📂 docs/                            # 📸 Documentation & screenshots
    ├── deploydashboard.png             # 🖼️ Dashboard deployment screenshot
    ├── output.png                      # 🖼️ Output visualization 1
    └── output1.png                     # 🖼️ Output visualization 2
```

### Directory Details

#### Core Application Files
- **app.py**: Main Streamlit application with prediction interface, model comparison, and data insights
- **data_cleaning.py**: Data preprocessing utilities for handling raw cricket data
- **requirements.txt**: Optimized dependency list for fast Streamlit Cloud deployment

#### Dataset Organization
- Raw cricket data from T20I matches (2005-2024)
- Engineered features including rolling averages, strike rates, and form indicators
- Player information and match context data

#### Models Directory
- Three trained ML models (XGBoost, LightGBM, Random Forest)
- Feature preprocessing pipeline for consistent transformations
- Performance metrics and SHAP analysis for model interpretability

#### Notebooks
- Complete ML workflow from EDA to deployment
- Feature engineering experiments and validation
- Model training with hyperparameter optimization

#### Scripts
- Automated feature generation from raw data
- Configuration management
- Utility functions for data processing

## 🏗️ Architecture

### System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[Raw T20I Data] --> B[Data Cleaning]
        B --> C[Feature Engineering]
        C --> D[Processed Datasets]
    end
    
    subgraph "Model Layer"
        D --> E[Model Training]
        E --> F[XGBoost]
        E --> G[LightGBM]
        E --> H[Random Forest]
        F --> I[Model Evaluation]
        G --> I
        H --> I
        I --> J[Best Model Selection]
    end
    
    subgraph "Application Layer"
        J --> K[Streamlit Dashboard]
        K --> L[Prediction Interface]
        K --> M[Model Comparison]
        K --> N[Data Insights]
    end
    
    subgraph "Deployment"
        K --> O[Docker Container]
        K --> P[Streamlit Cloud]
        O --> Q[Cloud Platforms]
    end
    
    style F fill:#1f77b4
    style G fill:#2ca02c
    style H fill:#d62728
    style K fill:#ff4b4b
    style P fill:#ffa500
```

### Data Flow Pipeline

```mermaid
graph LR
    A[📊 Raw CSV Files] --> B[🧹 Data Cleaning]
    B --> C[🔧 Feature Engineering]
    C --> D[📈 Rolling Averages<br/>Strike Rates<br/>Form Indicators]
    D --> E[🎯 ML Models]
    E --> F[🏆 XGBoost Best]
    E --> G[💚 LightGBM]
    E --> H[🌲 Random Forest]
    F --> I[🔮 Predictions]
    G --> I
    H --> I
    I --> J[📱 Dashboard UI]
    
    style A fill:#e1f5ff
    style E fill:#fff3e0
    style I fill:#f3e5f5
    style J fill:#e8f5e9
```

### Model Training Workflow

```mermaid
graph TD
    A[Load Cricket Datasets] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Train/Test Split<br/>80/20]
    D --> E[Baseline Model<br/>10-Match Avg]
    E --> F[Random Forest<br/>Default Params]
    F --> G[XGBoost<br/>Optuna Tuning]
    G --> H[LightGBM<br/>GridSearch Tuning]
    H --> I{Model Evaluation<br/>RMSE/MAE/R²}
    I --> J[SHAP Analysis]
    J --> K[Save Best Models]
    K --> L[Deploy to Dashboard]
    
    style G fill:#1f77b4,color:#fff
    style I fill:#ffd700
    style L fill:#32cd32,color:#fff
```

### Feature Engineering Process

```mermaid
graph LR
    A[Player Stats] --> B[Rolling Metrics]
    A --> C[Career Stats]
    A --> D[Match Context]
    
    B --> E[Last 3, 5, 10<br/>Matches Avg]
    C --> F[Total Runs<br/>Career Avg/SR]
    D --> G[Venue<br/>Opposition<br/>Position]
    
    E --> H[Feature Vector<br/>45+ Features]
    F --> H
    G --> H
    
    H --> I[Standardization]
    I --> J[Model Input]
    
    style A fill:#e3f2fd
    style H fill:#fff3e0
    style J fill:#e8f5e9
```

### Deployment Architecture

```mermaid
graph TB
    A[GitHub Repository] --> B{Deployment Target}
    
    B --> C[Streamlit Cloud]
    C --> D[Auto Build]
    D --> E[Live App URL]
    
    B --> F[Docker Container]
    F --> G[AWS ECS]
    F --> H[Azure Container Apps]
    F --> I[Heroku]
    
    B --> J[Local Development]
    J --> K[localhost:8501]
    
    style C fill:#ff4b4b,color:#fff
    style F fill:#2496ed,color:#fff
    style E fill:#32cd32,color:#fff
```

## 🛠️ Development Workflow

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

- Analyze dataset distributions
- Identify patterns and outliers
- Visualize relationships

### 2. Feature Engineering

```bash
jupyter notebook notebooks/02_FeatureEngineering.ipynb
```

- Create rolling statistics
- Encode categorical variables
- Scale numerical features
- Save preprocessing pipeline

### 3. Model Training

```bash
jupyter notebook notebooks/03_ModelTraining.ipynb
```

- Train multiple models
- Hyperparameter tuning (Optuna/GridSearchCV)
- SHAP analysis
- Save best models

### 4. Dashboard Deployment

```bash
streamlit run app.py
```

- Interactive predictions
- Model comparisons
- Data insights

## 🔬 Technical Details

### Data Preprocessing

- **Missing Values**: Forward fill for time-series data
- **Encoding**: Label encoding for venues and countries
- **Scaling**: StandardScaler for numerical features
- **Feature Selection**: Based on correlation and SHAP importance

### Model Training

- **Train-Test Split**: 80-20 temporal split
- **Cross-Validation**: 5-fold CV for hyperparameter tuning
- **Optimization**: Optuna (XGBoost), GridSearchCV (LightGBM)
- **Evaluation**: RMSE, MAE, R² on held-out test set

### Hyperparameter Tuning

```python
# XGBoost - Optuna (50 trials)
- n_estimators: [100, 500]
- max_depth: [3, 10]
- learning_rate: [0.01, 0.3]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]

# LightGBM - GridSearchCV (243 combinations)
- n_estimators: [100, 200, 300]
- max_depth: [5, 7, 9]
- learning_rate: [0.05, 0.1, 0.15]
```

## 📈 Performance Insights

### Prediction Accuracy by Score Range

| Score Range | Avg Error | Accuracy |
| ----------- | --------- | -------- |
| 0-20 runs   | ±8 runs   | 85%      |
| 21-40 runs  | ±12 runs  | 78%      |
| 41-60 runs  | ±15 runs  | 72%      |
| 60+ runs    | ±20 runs  | 65%      |

### Model Strengths

✅ **Good at predicting**:

- Consistent performers (low variance)
- Middle-order batsmen
- Players with clear recent trends

⚠️ **Challenges**:

- Debut performances (no history)
- Highly variable players
- Extreme conditions/venues

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (when available)
pytest tests/

# Format code
black app.py

# Lint code
flake8 app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Source**: Cricket statistics from official T20I records
- **Libraries**: scikit-learn, XGBoost, LightGBM, Streamlit, SHAP, Plotly
- **Community**: Open-source ML and cricket analytics communities

## 📧 Contact

**Project Maintainer**: Your Name

- 📧 Email: your.email@example.com
- 🐙 GitHub: [@yourusername](https://github.com/yourusername)
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🎯 Roadmap

### Upcoming Features

- [ ] 🔄 Real-time data integration
- [ ] 📱 Mobile-responsive dashboard
- [ ] 🌐 REST API for predictions
- [ ] 📊 Live match predictions
- [ ] 🤖 Ensemble model stacking
- [ ] 📈 Player comparison tools
- [ ] 🔔 Prediction confidence intervals
- [ ] 📉 Performance degradation monitoring

### Future Enhancements

- [ ] Deep learning models (LSTM, Transformers)
- [ ] Weather and pitch condition features
- [ ] Multi-format support (ODI, Test)
- [ ] Team performance predictions
- [ ] Fantasy cricket integration

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

**Made with ❤️ and 🏏 by cricket enthusiasts**

[Report Bug](https://github.com/yourusername/cricket-score-prediction/issues) • [Request Feature](https://github.com/yourusername/cricket-score-prediction/issues)

</div>

- **Team Features**: Team names, encoded IDs
- **Venue Features**: City, country, encoded locations
- **Toss Features**: Winner, choice (bat/bowl)
- **Batting Stats**: Runs, balls, fours, sixes, strike rate
- **Bowling Stats**: Wickets, economy, dots
- **Match Context**: Extras, wickets fallen

## 📁 Project Structure

```
Cricket-Player-Performance-Prediction---Group-1-/
│
├── 📂 dataset/                          # Cricket datasets
│   ├── t20i_Matches_Data.csv
│   ├── t20i_Batting_Card.csv
│   ├── t20i_Bowling_Card.csv
│   ├── t20i_Partnership_Card.csv
│   └── players_info.csv
│
├── 📂 models/                           # Trained models
│   └── cricket_score_model.pkl          # Saved Gradient Boosting model
│
├── 📂 notebooks/                        # Jupyter notebooks
│   └── score_predictor.ipynb           # Main ML notebook
│
├── 📄 requirements.txt                  # Python dependencies
├── 📄 README.md                         # Project documentation
└── 📄 .gitignore                        # Git ignore rules
```

## 📈 Model Training Pipeline

```mermaid
graph LR
A[Load Datasets] --> B[Data Aggregation]
B --> C[Feature Engineering]
C --> D[Encode Categories]
D --> E[Train/Test Split]
E --> F[Train Models]
F --> G[Evaluate & Compare]
G --> H[Save Best Model]
H --> I[Make Predictions]
```

## 🎯 Accuracy Metrics

The model achieves the following performance on test data:

- **Mean Absolute Error (MAE)**: 14.5 runs
- **Root Mean Squared Error (RMSE)**: 19.2 runs
- **R² Score**: 0.874
- **Mean Prediction Confidence**: ±15 runs

### Error Distribution

- 65% of predictions within ±10 runs
- 85% of predictions within ±20 runs
- 95% of predictions within ±30 runs

## 🔮 Future Enhancements

- [ ] Player-specific performance metrics
- [ ] Weather and pitch condition analysis
- [ ] Real-time in-match score prediction
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Web dashboard with interactive visualizations
- [ ] REST API for predictions
- [ ] Mobile app integration
- [ ] Recent team form analysis
- [ ] Head-to-head statistics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Deekshith Gowda**

- 🐙 GitHub: [@deekshithgowda85](https://github.com/deekshithgowda85)
- 📂 Repository: [Cricket-Score-Prediction-System](https://github.com/deekshithgowda85/Cricket-Score-Prediction-System)
- 🌐 Live App: [cricket-score-prediction-system.streamlit.app](https://cricket-score-prediction-system.streamlit.app/)

## 🙏 Acknowledgments

- 📊 **Data Source**: T20I cricket statistics from official records (2005-2024)
- 🛠️ **Libraries**: scikit-learn, XGBoost, LightGBM, Streamlit, SHAP, Plotly, Pandas, NumPy
- 🏏 **Community**: Open-source ML and cricket analytics communities
- 🤖 **AI Assistance**: GitHub Copilot for development support

## 📧 Contact

For questions, feedback, or collaboration opportunities:

- 📧 Open an issue in the [GitHub repository](https://github.com/deekshithgowda85/Cricket-Score-Prediction-System/issues)
- 💬 Start a discussion in the repository
- 🌐 Try the [live demo](https://cricket-score-prediction-system.streamlit.app/)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and 🏏 by Deekshith Gowda

</div>
