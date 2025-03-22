# PSO-ELM-Journal


## Enhancing DDoS Detection: A Comparative Analysis of Machine Learning Models with an Optimized PSO-ELM Approach
This project focuses on improving the detection of Distributed Denial of Service (DDoS) attacks using a comparative study of traditional machine learning models alongside an optimized Particle Swarm Optimization-based Extreme Learning Machine (PSO-ELM) approach.

### Project Overview
The research involves comprehensive data preprocessing, handling of class imbalance with SMOTE, feature selection using statistical techniques (Mutual Information, ANOVA), dimensionality reduction via Incremental PCA, and the development of various machine learning models including Random Forest, MLP, KNN, and the proposed PSO-optimized ELM. The project evaluates each model using multiple performance metrics and visualizations to compare their effectiveness in detecting DDoS attacks.

### Dataset
The dataset includes labeled network traffic data representing both normal and DDoS attack scenarios. Preprocessing includes normalization, class balancing using SMOTE, and splitting the data into training and testing sets. The dataset is suitable for binary classification: DDoS Attack vs Normal Traffic. You can download the dataset from here: https://drive.google.com/file/d/1ssdrm80E2zk43NbLVkdUh5eBd78UkRls/view?usp=sharing

### Technologies Used
Python, TensorFlow / Keras, Scikit-learn, imbalanced-learn (SMOTE), Matplotlib & Seaborn (for visualization), PySwarm (for PSO), Pandas, NumPy

### Model Architecture
Baseline Models: Random Forest, K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP). Proposed Model: Extreme Learning Machine (ELM) with single-layer feedforward neural network and Optimized using Particle Swarm Optimization (PSO) for selecting input weights and biases, Feature selection and dimensionality reduction techniques are applied to enhance training efficiency and model accuracy


![Untitled Diagram-Page-2 drawio (3)](https://github.com/user-attachments/assets/8df4aa56-207e-47a2-b59f-d3f1e0d05f82)

### Cloud Architecture:
![Untitled Diagram-Page-1 drawio](https://github.com/user-attachments/assets/a2af1a4f-5d10-43ec-8ef6-4c1eccb4f66d)


### Results and Evaluation Metrices
The PSO-ELM model outperformed traditional classifiers in terms of accuracy, F1-score, and ROC-AUC, Dimensionality reduction and feature selection significantly improved detection speed and reduced overfitting, Visualization tools such as ROC curves and confusion matrices confirmed strong model performance, The comparative analysis validates the strength of the optimized ELM approach for real-time DDoS detection


### ROC Curve: 
![final_roc_curve_ddos (1)](https://github.com/user-attachments/assets/be7511a6-7796-4749-8546-9a9095859f06)

### Accuracy Curve
![accuracy](https://github.com/user-attachments/assets/06a0efb6-ddbd-4ce9-b3f6-b4d70e038dac)

### Loss Curve:
![train_test_loss](https://github.com/user-attachments/assets/7c0116ad-3abf-44a0-8cd4-edae0a73cb4d)

### Five Fold Cross Validation
![Copy of Five_Fold_SCV1 (1)](https://github.com/user-attachments/assets/10917234-f014-4169-a575-d83bfca85ebe)




