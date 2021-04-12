<h3>Treinamento e Validação das Classificações</h3>
    
    
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


<h3>Usando GaussianNB</h3>



```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit( X_train, y_train )
preds = gnb.predict( X_test )
# Avaliar a precisão
print(accuracy_score(y_test, preds))

```

    0.9991198289953477
    
<h3>Usando K-ésimo Vizinho mais Próximo</h3>

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
    
<h3>Usando Arvore de Decisão</h3>

```python
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
preds = dtc.predict(X_test)
# Avaliar a precisão
print(accuracy_score(y_test, preds))

```

    0.9983234838006623
    
<h3>Usando Classificação do vetor de suporte linear</h3>

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

    
<h3>ESCOLHA DA MELHOR PRECISÃO DE CLASSIFICAÇÃO</h3>

  
    
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
    
    
    
<h4>A Arvore de decisão apresentou melhor eficácia nos resudados</h4>
