# Optimizing Feature Selection for Network Intrusion Detection Using Meta-Heuristic Algorithms on the UNSW-NB15 Dataset

This project focuses on feature selection using different Meta-Heuristic (MH) algorithms and classification of the UNSW-NB15 dataset. The selected features from these algorithms are used to train machine learning models, which are then evaluated based on their performance metrics.

## Directory and Imports

The code initializes by mounting Google Drive and changing the directory to the specific folder containing the MH algorithms and datasets.

```python
from google.colab import drive
drive.mount('/content/drive')

import os
folder_path = "/content/drive/MyDrive/MH_Algorithms"
os.chdir(folder_path)
os.listdir()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
```

## Training Data Processing

The training data is loaded, and specific columns are dropped. The categorical columns (`proto`, `state`, and `service`) are converted to numeric values. The data is then normalized.

```python
data_read = pd.read_csv('/content/drive/MyDrive/datasets/UNSW_NB15_training-set.csv')
data_1 = data_read.drop(['id', 'attack_cat', 'label'], axis = 1)

le = LabelEncoder()
proto = le.fit_transform(data_1['proto'])
data_1['proto'] = proto

state = le.fit_transform(data_1['state'])
data_1['state'] = state

service = le.fit_transform(data_1['service'])
data_1['service'] = service

train_feat = np.asarray(data_1.values[:, :])

data_2 = data_read.values
train_label = np.asarray(data_2[:, -1]).astype('int')

scaler = preprocessing.StandardScaler().fit(train_feat)
train_minmax_feat = scaler.transform(train_feat)
```

## Test Data Processing

Similarly, the test data is processed by dropping specific columns, converting categorical data to numeric, and normalizing the features.

```python
test_data_read = pd.read_csv('/content/drive/MyDrive/datasets/UNSW_NB15_testing-set.csv')
test_data_1 = test_data_read.drop(['id', 'attack_cat', 'label'], axis = 1)

test_le = LabelEncoder()
proto = test_le.fit_transform(test_data_1['proto'])
test_data_1['proto'] = proto

state = test_le.fit_transform(test_data_1['state'])
test_data_1['state'] = state

service = test_le.fit_transform(test_data_1['service'])
test_data_1['service'] = service

test_feat = np.asarray(test_data_1.values[:, :])

test_label = np.asarray(test_data_read.values[:, -1]).astype('int')

test_scaler = preprocessing.StandardScaler().fit(test_feat)
test_scaled_feat = test_scaler.transform(test_feat)
```

## Meta-Heuristic Model Selection and Feature Selection

Four Meta-Heuristic models are available: 
- Particle Swarm Optimization (PSO)
- Sine Cosine Algorithm (SCA)
- Flower Pollination Algorithm (FPA)
- Differential Evolution (DE)

The first model is selected, and feature selection is performed.

```python
import importlib
model_selected = 'sca' 
var = importlib.import_module(model_selected)

xtrain, xtest, ytrain, ytest = train_test_split(train_minmax_feat, train_label, test_size=0.2, stratify=train_label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

opts = {'k':5, 'fold':fold, 'N':10, 'T':5, 'alpha':2}
fmdl = var.jfs(train_minmax_feat, train_label, opts)

sel_feat = fmdl['sf']
num_feat = fmdl['nf']

curve = fmdl['c'].reshape(np.size(fmdl['c'], 1))
x = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Cost as Fitness')
ax.set_title(model_selected)
ax.grid()
plt.show()
```

## Second Model Selection and Feature Selection

After selecting features using the first model, a second Meta-Heuristic model is selected for further feature selection.

```python
model2_selected = 'pso'
var2 = importlib.import_module(model2_selected)

x_train = xtrain[:, sel_feat]
y_train = ytrain

fxtrain, fxtest, fytrain, fytest = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
fold = {'xt':fxtrain, 'yt':fytrain, 'xv':fxtest, 'yv':fytest}

smdl_opts = {'k':5, 'fold':fold, 'N':10, 'T':5, 'c1':1.5, 'c2':2, 'w':0.9}
smdl = var2.jfs(x_train, y_train, smdl_opts)

smdl_sel_feat = smdl['sf']
smdl_num_feat = smdl['nf']

smdl_curve = smdl['c'].reshape(np.size(smdl['c'], 1))
smdl_x = np.arange(0, smdl_opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(smdl_x, smdl_curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Cost as Fitness')
ax.set_title(model2_selected)
ax.grid()
plt.show()
```

## Classification Results on Test Data

The selected features from the second model are used to classify the test data using three classifiers: J48, Random Forest, and SVM.

```python
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import time

x_valid = test_scaled_feat[:, smdl_sel_feat]
y_valid = test_label

clf1 = DecisionTreeClassifier(criterion = "entropy", random_state = 42, max_depth=3, min_samples_leaf=5)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(kernel='linear', probability=True)

scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

for clf, label in zip([clf1, clf2], ['J48', 'Random Forest', 'SVM']):
  start_time = time.time()
  scores = cross_validate(clf, x_valid, y_valid, scoring=scoring, cv=5)
  print("Accuracy: %0.4f Precision: %0.4f Recall: %0.4f F-score: %0.4f [%s]" % (scores['test_accuracy'].mean(), scores['test_precision'].mean(), scores['test_recall'].mean(), scores['test_f1_score'].mean(), label))
  end_time = time.time()
  exec_time = end_time - start_time
  print("Time:", exec_time)
```

### **Purpose and Use of the Model**

#### **Purpose**

The primary purpose of this model is to **optimize feature selection** for a classification task using the UNSW-NB15 dataset, which is commonly used for network intrusion detection research. The goal is to identify the most relevant features (i.e., variables or columns) from the dataset that contribute most significantly to the accuracy and performance of the classification models. By selecting only the most important features, the model can achieve better performance, reduce computational complexity, and improve interpretability.

#### **Use**

1. **Feature Selection with Meta-Heuristic Algorithms**:
   - **Meta-heuristic algorithms** like Particle Swarm Optimization (PSO), Sine Cosine Algorithm (SCA), Flower Pollination Algorithm (FPA), and Differential Evolution (DE) are used to search through the feature space and select an optimal subset of features. These algorithms are inspired by natural processes and are effective in solving complex optimization problems.

2. **Improving Model Performance**:
   - By selecting only the most relevant features, the model can reduce overfitting, improve accuracy, and speed up the training and prediction processes. It ensures that the machine learning models are trained on data that carries the most predictive power.

3. **Handling High-Dimensional Data**:
   - High-dimensional datasets, like the UNSW-NB15, often contain irrelevant or redundant features that do not contribute to the model’s performance. This model helps in reducing dimensionality by selecting a smaller subset of features that still provides good predictive performance.

4. **Ensemble Feature Selection**:
   - The code uses a two-step feature selection process, where the first model reduces the feature set, and the second model refines it further. This ensemble approach leverages the strengths of multiple algorithms to achieve better feature selection results.

5. **Classification in Network Security**:
   - The final reduced feature set is used to train and evaluate classifiers like Decision Tree (J48), Random Forest, and Support Vector Machine (SVM). These classifiers are then used to detect network intrusions by classifying network traffic data as either benign or malicious.

6. **Research and Experimentation**:
   - The model is suitable for research purposes, where different meta-heuristic algorithms can be compared and tested on the same dataset. It can help researchers and practitioners determine which algorithms are more effective in selecting features for a specific type of data or classification task.

### **Conclusion**

The model is an essential tool for tasks where feature selection is critical, particularly in high-dimensional datasets like those found in network intrusion detection. By optimizing feature selection using meta-heuristic algorithms, the model enhances the accuracy and efficiency of machine learning classifiers, making it valuable for both academic research and practical applications in cybersecurity.
