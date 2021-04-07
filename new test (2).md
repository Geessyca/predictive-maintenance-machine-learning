```python
import pandas as pd

df = pd.read_csv('PdM_telemetry_.csv')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>machineID</th>
      <th>volt</th>
      <th>rotate</th>
      <th>pressure</th>
      <th>vibration</th>
      <th>failure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/01/2015 06:00</td>
      <td>1</td>
      <td>176.217853</td>
      <td>418.504078</td>
      <td>113.077935</td>
      <td>45.087686</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01/01/2015 07:00</td>
      <td>1</td>
      <td>162.879223</td>
      <td>402.747490</td>
      <td>95.460525</td>
      <td>43.413973</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01/01/2015 08:00</td>
      <td>1</td>
      <td>170.989902</td>
      <td>527.349825</td>
      <td>75.237905</td>
      <td>34.178847</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01/01/2015 09:00</td>
      <td>1</td>
      <td>162.462833</td>
      <td>346.149335</td>
      <td>109.248561</td>
      <td>41.122144</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01/01/2015 10:00</td>
      <td>1</td>
      <td>157.610021</td>
      <td>435.376873</td>
      <td>111.886648</td>
      <td>25.990511</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_dados = df[["volt","rotate","pressure","vibration"]]
y_dados = df["failure"]
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X_dados, y_dados, test_size=0.33, random_state=42)

print( 'Dados treino X: {}'.format(len(X_train)) )
print( 'Dados teste X: {}'.format(len(X_test)) )
print( 'Dados treino y: {}'.format(len(y_train)) )
print( 'Dados teste y: {}'.format(len(y_test)) )
```

    Dados treino X: 43596
    Dados teste X: 21474
    Dados treino y: 43596
    Dados teste y: 21474
    


```python
from sklearn.metrics import accuracy_score
```


```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit( X_train, y_train )
preds = gnb.predict( X_test )
# Avaliar a precisão
print(accuracy_score(y_test, preds))
```

    0.9993480488032038
    


```python
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,26)
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)
# Avaliar a precisão
print(accuracy_score(y_test, preds))
```

    0.9994411846884604
    


```python
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
preds = dtc.predict(X_test)
# Avaliar a precisão
print(accuracy_score(y_test, preds))
```

    0.9986029617211511
    


```python
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train,y_train)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
preds = model.predict(X_test)
print(accuracy_score(y_test, preds))
```

    0.9994411846884604
    

    C:\Users\Edsom\anaconda3\lib\site-packages\sklearn\svm\_base.py:976: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn("Liblinear failed to converge, increase "
    


```python

```
