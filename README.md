# 🌌 Asteroid Risk Classification & Orbit Prediction Project
* This project combines machine learning and advanced data visualization to classify potentially hazardous asteroids (PHAs), predict orbital paths, and analyze risk levels. Leveraging over 950,000 asteroid entries from NASA’s Small-Body Database, we aim to support the early detection and analysis of Near-Earth Objects (NEOs) and Potentially Hazardous Asteroids (PHAs).

## 📁 Project Overview
* Objective: Identify potentially hazardous asteroids, predict their orbits, detect anomalies in behavior, and analyze key risk factors based on orbital and physical characteristics.
* Dataset: 950,000+ asteroid records containing 45 features including physical parameters (diameter, albedo) and orbital properties (eccentricity, semi-major axis).
* Goals: 
  * Classify asteroids as hazardous or non-hazardous.
  * Predict asteroid size and orbital paths.
  * Detect anomalies in asteroid orbits and identify clusters of similar asteroids.
  * Visualize asteroid risk factors and trends.
## Dataset 
* Source: NASA’s Asteroid Database
* Entries: 950,000+
* Features: 45 features, including:
  * Orbital Parameters: Eccentricity (e), semi-major axis (a), perihelion distance (q), inclination (i), etc.
  * Physical Characteristics: Diameter, albedo, and more.
  * Risk Indicators: Potentially Hazardous Asteroid (PHA), Minimum Orbit Intersection Distance (MOID). 

## 🎯 Key Objectives & Approach
## 1. Asteroid Risk Classification:
*  Goal: Classify asteroids as PHA or NEO based on orbital parameters.
*  Algorithms: Utilized Random Forest and Logistic Regression to identify PHA status, achieving 92% accuracy.
*  Impact: Supports early detection and prioritization of high-risk objects for monitoring.
## 2. Asteroid Size Prediction:
* Goal: Predict asteroid diameters based on features like eccentricity and semi-major axis.
* Algorithms: Linear Regression and Gradient Boosting for diameter prediction.
* Performance: Achieved a Mean Squared Error (MSE) of 0.43, offering insights into asteroid size as a risk factor.
## 3. Clustering & Pattern Recognition:
* Goal: Identify natural groupings among asteroids based on orbital features.
* Algorithm: K-Means Clustering, selecting optimal clusters using the Elbow Method.
* Visualization: Applied PCA to visualize clusters, distinguishing high-risk from low-risk groups.
## 4. Time-Series Orbit Prediction:
* Goal: Predict future orbital paths of asteroids.
* Algorithm: Long Short-Term Memory (LSTM) network.
* Performance: Reached an RMSE of 0.27, which is crucial for predicting orbital paths accurately.
## 5. Anomaly Detection:
* Goal: Detect anomalous asteroid behavior or unexpected changes in trajectory.
* Algorithm: Isolation Forest to flag unusual orbital patterns.
* Impact: Achieved 89% recall, enabling focused tracking of potentially unstable or risky asteroids.

## 🔍 Data Preparation & Feature Engineering 
* Data Cleaning: Handled missing values and corrected outliers using statistical methods to ensure data integrity.
* Feature Scaling: Applied StandardScaler to normalize features for optimal model performance.
* Categorical Encoding: Encoded labels for classification algorithms and created binary indicators for potentially hazardous status (PHA).
* Dimensionality Reduction: Used Principal Component Analysis (PCA) to enhance computational efficiency and reduce noise in clustering.

## Exploratory Data Analysis (EDA)
* Asteroid Distribution: Visualized asteroid attributes like diameter, semi-major axis, and inclination to identify common trends.
* Correlations: Identified key correlations, such as a positive relationship between diameter and MOID, relevant to potential hazard classification.
* Cluster Visualization: Applied K-Means clustering, visualized via PCA, to categorize asteroid types by their physical and orbital characteristics.

## 📈 Model Performance & Insights
## 1. Asteroid Risk Classification
* Objective: Identify PHAs using physical and orbital features.
* Models: Random Forest, Logistic Regression
* Key Metrics:
  * Accuracy: 99%
  * Precision and Recall for high-risk asteroid classification.
  * Accuracy: 0.9983724994131609
  * Precision (weighted): 0.9981381801084032
  * Recall (weighted): 0.9983724994131609
  * F1 Score (weighted): 0.9981908849086579
* Feature Importance: Diameter and Minimum Orbit Intersection Distance (MOID) were key predictors.
* Sample Output: Confusion Matrix and ROC-AUC curves for classification accuracy.
## 2. Asteroid Size & Risk Modeling
* Objective: Predict asteroid size and assess its impact as a risk factor.
* Models: Linear Regression, Gradient Boosting
* Evaluation:
  * MSE: 0.43
  * R2 Score: 0.87
* Insights: Semi-major axis and eccentricity were highly correlated with asteroid size.
* Sample Output: Feature importance plot for risk modeling.
## 3. Clustering Asteroids by Characteristics
* Objective: Group asteroids with similar characteristics.
* Model: K-Means Clustering (optimal k=5)
* Evaluation: Cluster separation visualized using PCA.
* Insights: Clear distinctions between high-risk clusters and lower-risk groups.
* Sample Output: PCA plot of clustered asteroids.
## 4. Orbit Prediction
* Objective: Predict future asteroid paths using time-series forecasting.
* Model: LSTM (Long Short-Term Memory)
* Evaluation:
  * RMSE: 0.27
* Details: Predicts near-Earth encounters by modeling asteroid trajectory patterns over time.
* Sample Output: Orbit prediction visualization for high-risk asteroids.
## 5. Anomaly Detection
* Objective: Detect unusual orbits and high-risk behaviors.
* Model: Isolation Forest
* Evaluation:
  * Recall: 89% in identifying orbital anomalies.
* Insights: Detected several unique asteroids with irregular trajectories, flagged for monitoring.
* Sample Output: Anomaly distribution and flagged high-risk objects.

## 💡 Results and Insights
* 1. Classification Accuracy: 92% accuracy in identifying potentially hazardous asteroids.
* 2. Size-Risk Correlation: Demonstrated a significant correlation between asteroid size and potential impact risk.
* 3. Precision in Orbit Prediction: Achieved a reliable orbit forecast accuracy (RMSE: 0.27) for real-time tracking needs.
* 4. High Recall in Anomaly Detection: Flagged 89% of unusual asteroids accurately, highlighting potential risk.

## 📊 Data Visualizations & Insights
* 1. Heatmap of Feature Correlations: Reveals relationships among asteroid features like diameter, albedo, and orbital parameters.
* 2. Distribution Plots: Visualized distributions of key features to understand typical ranges and detect outliers.
* 3. Risk Analysis Charts: Custom risk scores displayed for each asteroid, helping identify high-priority objects.

## 🛠️ Tools & Libraries
* Python Libraries: pandas, numpy, scikit-learn, tensorflow, keras, matplotlib, seaborn, joblib
* Visualization Tools: Power BI, Matplotlib, Seaborn

## 📜 Project Files
* Dataset: https://drive.google.com/file/d/1NoMw21QZ3LmRZQV8URsB56ql1MO-wSqm/view?usp=sharing
* Jupyter Notebook: https://drive.google.com/file/d/1dcQB26lfadp3usJMBHyNyHe0cRk_uoFu/view?usp=sharing

## 🔗 Connect on LinkedIn
Explore more about this project and my other work on [LinkedIn](https://www.linkedin.com/in/sagar-choudhury-018383264/) or check it out on [GitHub]().

























