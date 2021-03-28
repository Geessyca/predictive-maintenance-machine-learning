# Coleta dos Dados


```python
import pandas as pd
```


```python
df = pd.read_csv('database2021.csv', sep=';', encoding='ansi')
```


```python
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
      <th>vibracao</th>
      <th>temperatura</th>
      <th>falha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201.000000</td>
      <td>24.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>165.491509</td>
      <td>26.162342</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>256.880637</td>
      <td>22.672392</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>218.475777</td>
      <td>34.534921</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>324.587769</td>
      <td>30.074960</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.vibracao.describe()
```




    count    200.000000
    mean     208.654996
    std      118.533573
    min        2.525349
    25%      115.510337
    50%      207.536735
    75%      305.510671
    max      417.765658
    Name: vibracao, dtype: float64




```python
df.temperatura.describe()
```




    count    200.000000
    mean      24.379336
    std       14.428990
    min        0.580957
    25%       12.559474
    50%       24.167485
    75%       36.119931
    max       49.813418
    Name: temperatura, dtype: float64



# TESTES


```python
import numpy as np
```


```python
# Variáveis de entrada
X_v = df[['vibracao','temperatura']]
```


```python
# Saída esperada
y = df.falha
```


```python
from sklearn.linear_model import LinearRegression
model_v = LinearRegression()
model_v.fit(X_v, y)
```




    LinearRegression()




```python
predict = model_v.predict(np.array([[201,24]]))
predict
```




    array([80.57497812])




```python
print(df.vibracao[0],"\t",df.temperatura[0],"\t",df.falha[0],"\t",predict)

```

    201.0 	 24.0 	 0 	 [80.57497812]
    

# Teste 02


```python
import pandas as pd
```


```python
df_2 = pd.read_csv('database2021-2.csv', sep=';', encoding='ansi')
```


```python
df_2
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
      <th>area</th>
      <th>bedrooms</th>
      <th>age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>3</td>
      <td>20</td>
      <td>550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>4</td>
      <td>15</td>
      <td>565000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>0</td>
      <td>18</td>
      <td>610000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>3</td>
      <td>30</td>
      <td>595000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4000</td>
      <td>5</td>
      <td>8</td>
      <td>760000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4100</td>
      <td>6</td>
      <td>8</td>
      <td>810000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
```


```python
# Variáveis de entrada
X_v = df_2[['area','bedrooms','age']]
```


```python
# Saída esperada
y = df_2.price
```


```python
from sklearn.linear_model import LinearRegression
model_v = LinearRegression()
model_v.fit(X_v, y)
```




    LinearRegression()




```python
model_v.predict(np.array([[2600,3,20]]))
```




    array([521891.50854318])


