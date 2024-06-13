# EDA--Breast-Cancer-Wisconsin-Diagnostic
Breast Cancer Wisconsin Diagnostic - Exploratory Data Analysis (EDA)
Overview
This project performs an Exploratory Data Analysis (EDA) on the Breast Cancer Wisconsin (Diagnostic) dataset. The primary objective is to understand the dataset's structure, uncover patterns, and provide insights that could be beneficial for further analysis or modeling.

Dataset
The dataset used in this analysis is the Breast Cancer Wisconsin (Diagnostic) dataset, which is publicly available from the UCI Machine Learning Repository. It includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset is used to classify whether the cancer is benign or malignant.

Number of Instances: 569
Number of Features: 30 (plus label)
Label: Diagnosis (M = malignant, B = benign)
Features
The features are computed for each cell nucleus and include:

Radius (mean of distances from center to points on the perimeter)
Texture (standard deviation of gray-scale values)
Perimeter
Area
Smoothness (local variation in radius lengths)
Compactness (perimeter^2 / area - 1.0)
Concavity (severity of concave portions of the contour)
Concave points (number of concave portions of the contour)
Symmetry
Fractal dimension ("coastline approximation" - 1)
Each feature is computed for the mean, standard error, and worst (mean of the three largest values) of each cell nucleus, resulting in a total of 30 feature columns.

Objectives
Data Cleaning: Handling missing values, if any, and ensuring the dataset is ready for analysis.
Descriptive Statistics: Summarizing the central tendency, dispersion, and shape of the dataset’s distribution.
Data Visualization: Using plots to understand the distribution, relationships, and outliers in the data.
Correlation Analysis: Identifying correlations between features to understand dependencies.
Installation
To run this project, you need to have Python installed along with the following libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
You can install the required libraries using pip:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Clone the repository and navigate to the project directory:

bash
Copy code
Run the Jupyter Notebook to explore the EDA:

bash
Copy code
jupyter notebook 
Analysis Summary
Data Cleaning
Verified no missing values in the dataset.
Checked for duplicates and removed any, if present.
Descriptive Statistics
Summary statistics provided for each feature.
Distribution of the diagnosis classes (Benign vs Malignant).
Data Visualization
Histograms and density plots for understanding the distribution of features.
Box plots to identify outliers.
Pair plots to visualize relationships between features.
Correlation Analysis
Heatmap of feature correlations.
Analysis of highly correlated features.
Results
The EDA revealed significant differences between malignant and benign cases across various features. Key insights include:

Certain features such as mean radius, texture, perimeter, area, and smoothness show clear separations between benign and malignant cases.
Strong correlations were observed between features like mean radius, perimeter, area, and compactness.
Conclusion
The EDA provides a comprehensive understanding of the Breast Cancer Wisconsin (Diagnostic) dataset. The insights gained are crucial for building predictive models and understanding the underlying patterns in the data.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
UCI Machine Learning Repository
This README provides a detailed overview of the project, guiding users on how to understand, install, and utilize the EDA. It also summarizes the findings and offers insights into the dataset.
