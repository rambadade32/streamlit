import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.title('Play with it')

st.write("""
## Explore different datasets and models
1. choose the  dsataset 
2. choose classifier model 
3. `Play with it` and do some hypertunning.

""")

dataset = st.sidebar.selectbox("Select your Dataset", ('Breast Cancer', 'Iris', 'Wine','Digits','Diabetes'))

st.write(f"## {dataset} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)


def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    elif name == 'Digits':
        data = datasets.load_digits()
    elif name == 'Diabetes':
        data = datasets.load_diabetes()
    else:
        data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    # for i in range(5):
    #     st.write(f'{data.data[i]}-{data.target[i]}')

    return X, y


# X, y = datasets.load_breast_cancer().data, datasets.load_breast_cancer().target
X, y = get_dataset(dataset)
st.write("Shape of Dataset ", X.shape)
st.write("Number of Classes", len(np.unique(y)))

df = pd.DataFrame(X)
df1  = pd.DataFrame(y)
st.write(df,df1)

def add_parameter(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('Max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


params = add_parameter(classifier_name)


def choose_model(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=42)
    return clf

clf = choose_model(classifier_name,params)

# classification

# clf = SVC()

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
preds = clf.predict(X_test)

accuracy = accuracy_score(y_test, preds)

st.write(f'classifier = {classifier_name}')
st.write(f'Accuracy = {accuracy}')

# PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)
