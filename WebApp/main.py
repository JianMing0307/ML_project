import streamlit as st
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt

st.title("Machine Learning-sklearn")
#側邊攔
data_name = st.sidebar.selectbox(
    '請選擇資料集:',
    ['iris','wine','cancer']
)
classifier = st.sidebar.selectbox(
    '請選擇分類器',
    ['KNN','SVM','RandomForest']
)


#下載資料集
def loadData(name):
    data = None
    if name =='iris':
        data = datasets.load_iris()
    elif name == 'wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()


    X = data.data
    y = data.target
    y_name = data.target_names
    return X, y, y_name
        
#顯示資料集資訊
X,y, y_name = loadData(data_name)
st.write("#### 資料集結構 :",X.shape)
st.write("#### 資料集分類 :",len(np.unique(y)))
st.write("#### 資料集分類 :")
for i in y_name:
    st.write("#### ",i)

df = pd.DataFrame(X)
st.write("#### 資料集前10筆:")
st.write(X[:10])

#定義模型參數
def parameter(clf):
    p={}
    if clf == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)
        p['C'] = C
    elif clf == 'KNN':
        K = st.sidebar.slider("K", 1, 20)
        p['K'] = K
    else:
        maz_depth = st.sidebar.slider("max_depth", 2, 15)
        p['dep'] = maz_depth
        trees = st.sidebar.slider("n_estimators", 1, 100)
        p['trees'] = trees
    return p

#取的參數
params = parameter(classifier)

#建立分類器模型
def getClassifier(clf, p) :
    now_clf =None
    if clf == 'SVM':
        now_clf = SVC(C=params['C'])
    elif clf == 'KNN':
        now_clf = KNeighborsClassifier(n_neighbors =params['K'])
    else :
        now_clf = RandomForestClassifier(n_estimator =params['trees'],
                                         max_depth=params['dep'],
                                         random_state=123)
    return now_clf

#取得模型物件
clf = getClassifier(classifier,params)
    
#分割資料集
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

#訓練資料(計算)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#評估準確率
acc = accuracy_score(y_test,y_pred)
#顯示結果
st.write("#### 準確率:", acc)
#PCA降維
pca = PCA(2)
new_X = pca.fit_transform(X)

x1 = new_X[:,0]
x2 = new_X[:,1]

#plt.show()
fig = plt.figure()
plt.scatter(x1, x2, c=y,alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

st.pyplot(fig)




