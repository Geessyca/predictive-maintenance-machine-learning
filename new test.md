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
X_train, X_test, y_train, y_test = train_test_split(X_dados, y_dados, test_size=0.33, random_state=42)
```


```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit( X_train, y_train )
```


```python
preds = gnb.predict( X_test )
```


```python
from sklearn.metrics import accuracy_score

# Avaliar a precisão
print(accuracy_score(y_test, preds))
```

    0.9994761655316919
    
