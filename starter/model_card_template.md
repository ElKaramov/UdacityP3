# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Training a Classification Model to determine if a person makes over 50K a year. We use a Logistic Regression using Hyperparameters tuning of the regularization parameter 'C' in logistic Regression was realized using GridSearchCV.
The Model is saved in the model folder. 
All training steps and metrics are logged in the file "LogFile.log".

## Intended Use
This model can be used to predict the salary level of an individual based off a handful of attributes. The usage is meant for students, academics or research purpose.
## Training Data
The Census Income Dataset was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) as a csv file. The original data set has 32,561 rows and 15 columns composed of the target label "salary", 8 categorical features and 6 numerical features. Details on each of the features ae available at the UCI link above. Target label "salary" has two classes ('<=50K', '>50K') and shows class imbalance with a ratio of 75% / 25%. 
A simple data cleansing was performed on the original dataset to remove leading and trailing whitespaces. See pre-processing.ipynb notebook for data exploration and cleansing step.

A 80%/20% split was used to break this dataset into a train and test set. Stratification on target label "salary" was applied. To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the target.
## Evaluation Data
20% of the dataset was set aside for model evaluation. Transformation was applied on the categorical features and the target label respectively using the One Hot Encoder and label binarizer fitted on the train set.

## Metrics
The classification performance is evaluated using precision, recall and fbeta metrics:

- precision: 0.713
- recall: 0.263
- fbeta: 0.384

## Ethical Considerations
The dataset should not be considered as a fair representation of the salary distribution and should not be used to assume salary level of certain population categories.
## Caveats and Recommendations
The dataset is a outdated sample, it was extracted from the 1994 Census DataBase. It is recommended to use the dataset for training purpose on ML classification or related problems.