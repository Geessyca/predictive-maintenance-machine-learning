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
      <th>Unnamed: 0</th>
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
      <td>0</td>
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
      <td>1</td>
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
      <td>2</td>
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
      <td>3</td>
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
      <td>4</td>
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

    Dados treino X: 145323
    Dados teste X: 71577
    Dados treino y: 145323
    Dados teste y: 71577
    


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

    0.9991198289953477
    


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

    0.9991617419003311
    


```python
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
preds = dtc.predict(X_test)
# Avaliar a precisão
print(accuracy_score(y_test, preds))

```

    0.9983234838006623
    


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

    0.9991617419003311

    
<h3>ESCOLHA DA MELHOR PRECISÃO DE CLASSIFICAÇÃO<\h3>

```python

import numpy as np
X_test = np.array([[179.30315305501,499.777962383718,111.833027646993,52.3830967199205]])
print("Linear SVC: ", model.predict( X_test ), "\nBanco de dados: comp4\n")
print("Arvore de decisão: ",dtc.predict( X_test ), "\nBanco de dados: comp4\n")
print("Vizinho + proximos: ", knn.predict( X_test ), "\nBanco de dados: comp4\n")
print("Gaussian: ", gnb.predict( X_test ), "\nBanco de dados: comp4\n")
print("\n")
X_test = np.array([[198.257974581675,456.862342339469,89.3339950053984,38.6718999063224]])
print("Linear SVC: ", model.predict( X_test ), "\nBanco de dados: comp1\n")
print("Arvore de decisão: ",dtc.predict( X_test ), "\nBanco de dados: comp1\n")
print("Vizinho + proximos: ", knn.predict( X_test ), "\nBanco de dados: comp1\n")
print("Gaussian: ", gnb.predict( X_test ), "\nBanco de dados: comp1\n")
print("\n")
X_test = np.array([[180.050800888506,346.362480252293,105.661163515464,39.2180547849819]])
print("Linear SVC: ", model.predict( X_test ), "\nBanco de dados: comp2\n")
print("Arvore de decisão: ",dtc.predict( X_test ), "\nBanco de dados: comp2\n")
print("Vizinho + proximos: ", knn.predict( X_test ), "\nBanco de dados: comp2\n")
print("Gaussian: ", gnb.predict( X_test ), "\nBanco de dados: comp2\n")
print("\n")
X_test = np.array([[187.673963498493,493.005160103346,105.334392333578,53.9639607034682]])
print("Linear SVC: ", model.predict( X_test ), "\nBanco de dados: comp4\n")
print("Arvore de decisão: ",dtc.predict( X_test ), "\nBanco de dados: comp4\n")
print("Vizinho + proximos: ", knn.predict( X_test ), "\nBanco de dados: comp4\n")
print("Gaussian: ", gnb.predict( X_test ), "\nBanco de dados: comp4\n")
print("\n")
X_test = np.array([[205.275427905855,362.153175675405,108.623106487313,47.7113737781357]])
print("Linear SVC: ", model.predict( X_test ), "\nBanco de dados: comp1\n")
print("Arvore de decisão: ",dtc.predict( X_test ), "\nBanco de dados: comp1\n")
print("Vizinho + proximos: ", knn.predict( X_test ), "\nBanco de dados: comp1\n")
print("Gaussian: ", gnb.predict( X_test ), "\nBanco de dados: comp1\n")
print("\n")
X_test = np.array([[179.277874469449,322.388169957166,118.15393419358,47.4158846916956]])
print("Linear SVC: ", model.predict( X_test ), "\nBanco de dados: comp2\n")
print("Arvore de decisão: ",dtc.predict( X_test ), "\nBanco de dados: comp2\n")
print("Vizinho + proximos: ", knn.predict( X_test ), "\nBanco de dados: comp2\n")
print("Gaussian: ", gnb.predict( X_test ), "\nBanco de dados: comp2\n")
print("\n")
```    
    
    Linear SVC:  ['0'] 
    Banco de dados: comp4
    
    Arvore de decisão:  ['0'] 
    Banco de dados: comp4
    
    Vizinho + proximos:  ['0'] 
    Banco de dados: comp4
    
    Gaussian:  ['0'] 
    Banco de dados: comp4
    
    
    
    Linear SVC:  ['0'] 
    Banco de dados: comp1
    
    Arvore de decisão:  ['comp1'] 
    Banco de dados: comp1
    
    Vizinho + proximos:  ['0'] 
    Banco de dados: comp1
    
    Gaussian:  ['0'] 
    Banco de dados: comp1
    
    
    
    Linear SVC:  ['0'] 
    Banco de dados: comp2
    
    Arvore de decisão:  ['0'] 
    Banco de dados: comp2
    
    Vizinho + proximos:  ['0'] 
    Banco de dados: comp2
    
    Gaussian:  ['0'] 
    Banco de dados: comp2
    
    
    
    Linear SVC:  ['0'] 
    Banco de dados: comp4
    
    Arvore de decisão:  ['comp4'] 
    Banco de dados: comp4
    
    Vizinho + proximos:  ['0'] 
    Banco de dados: comp4
    
    Gaussian:  ['0'] 
    Banco de dados: comp4
    
    
    
    Linear SVC:  ['0'] 
    Banco de dados: comp1
    
    Arvore de decisão:  ['comp1'] 
    Banco de dados: comp1
    
    Vizinho + proximos:  ['0'] 
    Banco de dados: comp1
    
    Gaussian:  ['0'] 
    Banco de dados: comp1
    
    
    
    Linear SVC:  ['0'] 
    Banco de dados: comp2
    
    Arvore de decisão:  ['0'] 
    Banco de dados: comp2
    
    Vizinho + proximos:  ['0'] 
    Banco de dados: comp2
    
    Gaussian:  ['0'] 
    Banco de dados: comp2
    
    
    
    


```python

```
