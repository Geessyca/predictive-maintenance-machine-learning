# Analise e pratica de machine learning

- Link para o banco de dados:

https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv

- GitHub fonte: https://github.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy

<h2>Descrição dos dados:</h2>

<b>Telemetry:</b> Dados da série temporal de telemetria, que consistem em medições de tensão, rotação, pressão e vibração coletadas de 100 máquinas em tempo real, calculadas em média a cada hora coletada.
```
telemetry = pd.read_csv('PdM_telemetry.csv')
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")
telemetry.head()
```
![dadostelemetry](https://user-images.githubusercontent.com/72661229/104634253-21a18180-567f-11eb-9054-91c46802e698.png)

<b>Errors:</b> Erros ininterruptos lançados enquanto a máquina ainda está operacional e não constituem falhas. A data e as horas do erro são arredondadas para a hora mais próxima, pois os dados de telemetria são coletados por hora.
```
errors = pd.read_csv('PdM_errors.csv')
errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
errors.head()
```
![dadoserrors](https://user-images.githubusercontent.com/72661229/104633919-a50ea300-567e-11eb-9248-74e4291b8de6.png)

<b>Maint:</b> Registros de manutenção programada e não programada que correspondem tanto à inspeção regular dos componentes quanto às falhas. Um registro é gerado se um componente for substituído durante a inspeção programada ou substituído devido a uma avaria. Os registros criados devido a falhas serão chamados de falhas, o que é explicado nas seções posteriores.
```
maint = pd.read_csv('PdM_maint.csv')
maint['datetime'] = pd.to_datetime(maint['datetime'], format="%Y-%m-%d %H:%M:%S")
maint.head()
```
![dadosmaint](https://user-images.githubusercontent.com/72661229/104634242-1ea69100-567f-11eb-8bf7-79ac3b9e3730.png)

<b>Machines:</b> Informações sobre as máquinas como de modelo e idade (anos de serviço).
```
machines = pd.read_csv('PdM_machines.csv')
machines.head()
```
![dadosmachine](https://user-images.githubusercontent.com/72661229/104635427-a345df00-5680-11eb-9b86-889a40811725.png)

<b>Failures:</b> Substituições de componentes devido a falhas. Cada registro possui uma data e hora, ID de máquina e tipo de componente com falha.
```
failures = pd.read_csv('PdM_failures.csv')
failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
failures.head()
```
![dadosfailures](https://user-images.githubusercontent.com/72661229/104634236-1e0dfa80-567f-11eb-953c-d65b8af2ca01.png)

<h2>Verificação de dados ausentes nos dados</h2>
```
telemetry.isnull().sum()
```

![isnulltelemetry](https://user-images.githubusercontent.com/72661229/104637826-100ea880-5684-11eb-8b48-b84e0fd88f8c.png)

```
errors.isnull().sum()
```

![isnullerrors](https://user-images.githubusercontent.com/72661229/104637830-10a73f00-5684-11eb-9566-76b8f785c43b.png)

```
maint.isnull().sum()
```

![isnullmaint](https://user-images.githubusercontent.com/72661229/104637825-100ea880-5684-11eb-8fda-e54d2a0d6439.png)

```
failures.isnull().sum()
```

![isnullfailure](https://user-images.githubusercontent.com/72661229/104637831-113fd580-5684-11eb-9197-2738151b28ef.png)

```
machines.isnull().sum()
```

![isnullmachine](https://user-images.githubusercontent.com/72661229/104637823-0f761200-5684-11eb-9575-cd24ddf6f4ff.png)

Em nenhum dos bancos de dados possui dados ausentes.

<h2>Exibição gráfica dos dados:</h2>

<h4>Telemetry - Pressure</h4>

``` ruby
plot_df = telemetry.loc[(telemetry['machineID'] == 1) &
                        (telemetry['datetime'] > pd.to_datetime('2015-01-01')) &
                        (telemetry['datetime'] < pd.to_datetime('2016-01-01')), ['datetime', 'pressure']]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.title('Data x Pressão - 2015')
plt.plot(plot_df['datetime'], plot_df['pressure'])
plt.ylabel('Pressão')


adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d'
plt.xlabel('Data')
```
![telemetryP](https://user-images.githubusercontent.com/72661229/104624145-fe240a00-5671-11eb-8eb8-425370f70270.png)

<h4>Telemetry - Rotate</h4>

```
plot_df = telemetry.loc[(telemetry['machineID'] == 1) &
                        (telemetry['datetime'] > pd.to_datetime('2015-01-01')) &
                        (telemetry['datetime'] < pd.to_datetime('2016-01-01')), ['datetime', 'rotate']]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.title('Data x Rotação - 2015')
plt.plot(plot_df['datetime'], plot_df['rotate'])
plt.ylabel('Rotação')


adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d'
plt.xlabel('Data')
```

![telemetryR](https://user-images.githubusercontent.com/72661229/104624147-febca080-5671-11eb-934b-f095286cfb76.png)

<h4>Telemetry - Volt</h4>

```
plot_df = telemetry.loc[(telemetry['machineID'] == 1) &
                        (telemetry['datetime'] > pd.to_datetime('2015-01-01')) &
                        (telemetry['datetime'] < pd.to_datetime('2016-01-01')), ['datetime', 'volt']]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.title('Data x Voltagem - 2015')
plt.plot(plot_df['datetime'], plot_df['volt'])
plt.ylabel('Voltagem')


adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d'
plt.xlabel('Data')
```
![telemetryV](https://user-images.githubusercontent.com/72661229/104624136-fc5a4680-5671-11eb-9051-0bd83de8f3aa.png)

<h4>Telemetry - Vibration</h4>

```
plot_df = telemetry.loc[(telemetry['machineID'] == 1) &
                        (telemetry['datetime'] > pd.to_datetime('2015-01-01')) &
                        (telemetry['datetime'] < pd.to_datetime('2016-01-01')), ['datetime', 'vibration']]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.title('Data x Vibração - 2015')
plt.plot(plot_df['datetime'], plot_df['vibration'])
plt.ylabel('Vibtação')


adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d'
plt.xlabel('Data')
```

![telemetryVb](https://user-images.githubusercontent.com/72661229/104624139-fcf2dd00-5671-11eb-83ac-8ad33144ef32.png)

<h4>Errors</h4>

```
sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
errors['errorID'].value_counts().plot(kind='bar')
plt.title('Quantidade de erros ao logo da coleta de dados')
plt.ylabel('Nº de erros')
```

![error](https://user-images.githubusercontent.com/72661229/104624140-fd8b7380-5671-11eb-8c11-7a667441ffb8.png)

<h4>Machine</h4>

```
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
machinesplot= plt.hist([machines.loc[machines['model'] == 'model1', 'age'],
                       machines.loc[machines['model'] == 'model2', 'age'],
                       machines.loc[machines['model'] == 'model3', 'age'],
                       machines.loc[machines['model'] == 'model4', 'age']],
                       20, stacked=True, label=['model1', 'model2', 'model3', 'model4'])
plt.title('Maquina e seu tempo')
plt.xlabel('Idade')
plt.ylabel('Nº de Máquinas')
```

![machines](https://user-images.githubusercontent.com/72661229/104624142-fd8b7380-5671-11eb-866b-5ae1dea6ddbc.png)

<h4>Failures</h4>

```
sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
failures['failure'].value_counts().plot(kind='bar')
plt.ylabel('Nº de componentes')
```

![Failures](https://user-images.githubusercontent.com/72661229/104624141-fd8b7380-5671-11eb-93f8-5b6bfbbd2da4.png)

<h4>Maint</h4>

```
sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
maint['comp'].value_counts().plot(kind='bar')
plt.ylabel('Nº de componentes')
```
![Maint](https://user-images.githubusercontent.com/72661229/104624144-fe240a00-5671-11eb-9a7e-8129d84c1394.png)

<h2>Recursos de atraso da telemetria</h2>

<h4>Recursos de atraso da telemetria curto com janela de 3hrs</h4>

<h5>Média Móvel</h5>

```
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).rolling(window=3).mean().resample('3H',
                                                                              closed='left',
                                                                              label='right').mean().unstack())
telemetrymean = pd.concat(temp, axis=1)
telemetrymean.columns = [i + 'mean_3h' for i in fields]
telemetrymean.reset_index(inplace=True)
```

![atrasomedia3](https://user-images.githubusercontent.com/72661229/104851343-35452600-58d3-11eb-81cc-5c429aec44e0.png)

<h5>Desvio Padrão</h5>

```
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).rolling(window=3).std().resample('3H',
                                                                             closed='left',
                                                                             label='right').std().unstack())
telemetrysd = pd.concat(temp, axis=1)
telemetrysd.columns = [i + 'sd_3h' for i in fields]
telemetrysd.reset_index(inplace=True)
```

![atrasostd3](https://user-images.githubusercontent.com/72661229/104851340-3413f900-58d3-11eb-910c-71e7c7d9e9db.png)

<h4>Recursos de atraso da telemetria curto longo com janela de 24hrs</h4>
<h5>Média Móvel</h5>

```
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).rolling(window=24).mean().resample('3H',
                                                                              closed='left',
                                                                              label='right').mean().unstack())
telemetrymean_24hrs = pd.concat(temp, axis=1)
telemetrymean_24hrs.columns = [i + 'mean_24h' for i in fields]
telemetrymean_24hrs.reset_index(inplace=True)
telemetrymean_24hrs = telemetrymean_24hrs.loc[-telemetrymean_24hrs['voltmean_24h'].isnull()]
```

![atrasomedia24](https://user-images.githubusercontent.com/72661229/104851344-35ddbc80-58d3-11eb-816c-a87748985e1d.png)

<h5>Desvio Padrão</h5>

```
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).rolling(window=24).std().resample('3H',
                                                                             closed='left',
                                                                             label='right').std().unstack())
telemetrysd_24hrs = pd.concat(temp, axis=1)
telemetrysd_24hrs.columns = [i + 'sd_24h' for i in fields]
telemetrysd_24hrs.reset_index(inplace=True)
telemetrysd_24hrs = telemetrysd_24hrs.loc[-telemetrysd_24hrs['voltsd_24h'].isnull()]

```

![atrasostd24](https://user-images.githubusercontent.com/72661229/104851341-34ac8f80-58d3-11eb-825c-5e21b6d69e80.png)

<h4>Mesclar os dados acima</h4>

```
telemetry_feat  =  pd.concat ([ telemetrymean ,
                             telemetrysd.iloc [:, 2 : 6 ],
                             telemetrymean_24hrs.iloc [:, 2 : 6 ],
                             telemetrysd_24hrs.iloc [:, 2 : 6 ]], axis = 1 ).dropna ()
```

![telemetry_feat](https://user-images.githubusercontent.com/72661229/104851342-35452600-58d3-11eb-9c3e-823fe734cef0.png)

<h4>Recursos de atraso de errors com janela de 24hrs</h4>
```
error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
error_count = error_count.groupby(['machineID','datetime']).sum().reset_index()
error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)
```

![errors_head](https://user-images.githubusercontent.com/72661229/104851466-e946b100-58d3-11eb-83b5-e43ae57ab1a5.png)

<h6> Dando error, reafazer!!
temp = []
fields = ['error%d' % i for i in range(1,6)]
for col in fields:
    temp.append(pd.pivot_table(error_count.rolling(window=24,center=False).sum()).resample('3H',
                                                                                           closed='left',
                                                                                           label='right',
                                                                                           how='first').first())
error_count = pd.concat(temp, axis=1)
error_count.columns = [i + 'count' for i in fields]
error_count.reset_index(inplace=True)
error_count = error_count.dropna()
</h6>

<h2>Dias desde a última substituição da manutenção</h2>

```
# crie uma coluna para cada tipo de erro 

comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

# combina reparos para uma determinada máquina em uma determinada hora 
comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

# adicionar pontos de tempo onde nenhum componente foi substituído 
comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                      on=['datetime', 'machineID'],
                                                      how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])

components = ['comp1', 'comp2', 'comp3', 'comp4']
for comp in components:
     # converter o indicador para a data mais recente de alteração do componente 
    comp_rep.loc[comp_rep[comp] < 1, comp] = None
    comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
    
    # forward-fill a data mais recente da alteração do componente 
    comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

# remove as datas em 2014 (pode ter NaN ou datas de mudança de componente futura)     
comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]

# substitua as datas da mudança de componente mais recente por dias desde a mudança de componente mais recente 
for comp in components:
    comp_rep[comp] = (comp_rep["datetime"] - pd.to_datetime(comp_rep[comp])) / np.timedelta64(1, "D") 

```

![comp_head](https://user-images.githubusercontent.com/72661229/104851469-ea77de00-58d3-11eb-95cc-e734eab99101.png)

<h2>Características da máquina</h2>

```
final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(machines, on=['machineID'], how='left')
```

![final_head](https://user-images.githubusercontent.com/72661229/104851468-e9df4780-58d3-11eb-946e-11f5a3ad8ce8.png)

<h2>Construção de etiqueta</h2>

```
labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
labeled_features = labeled_features.fillna(method='bfill', limit=7) # fill backward up to 24h
labeled_features = labeled_features.fillna('none')
```

![etiqueta](https://user-images.githubusercontent.com/72661229/104851467-e9df4780-58d3-11eb-910b-ef37c77aa983.png)

<h2>Modelagem</h2>

<h4>Treinamento, Validação e Teste</h4>

```
from sklearn.ensemble import GradientBoostingClassifier

# fazer divisões de teste e treinamento 
threshold_dates = [[pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')],
                   [pd.to_datetime('2015-08-31 01:00:00'), pd.to_datetime('2015-09-01 01:00:00')],
                   [pd.to_datetime('2015-09-30 01:00:00'), pd.to_datetime('2015-10-01 01:00:00')]]

test_results = []
models = []
for last_train_date, first_test_date in threshold_dates:
     # divide os dados de treinamento e teste 
    train_y = labeled_features.loc[labeled_features['datetime'] < last_train_date, 'failure']
    train_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['datetime',
                                                                                                        'machineID',
                                                                                                        'failure'], 1))
    test_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] > first_test_date].drop(['datetime',
                                                                                                       'machineID',
                                                                                                       'failure'], 1))
     # treinar e prever usando o modelo, armazenando resultados para posterior 
    my_model = GradientBoostingClassifier(random_state=42)
    my_model.fit(train_X, train_y)
    test_result = pd.DataFrame(labeled_features.loc[labeled_features['datetime'] > first_test_date])
    test_result['predicted_failure'] = my_model.predict(test_X)
    test_results.append(test_result)
    models.append(my_model)
    
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
labels, importances = zip(*sorted(zip(test_X.columns, models[0].feature_importances_), reverse=True, key=lambda x: x[1]))
plt.xticks(range(len(labels)), labels)
_, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.bar(range(len(importances)), importances)
plt.ylabel('Importance')
```

<h2>Avaliação</h2>

```

sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
labeled_features['failure'].value_counts().plot(kind='bar')
plt.xlabel('Component failing')
plt.ylabel('Count')

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

def Evaluate(predicted, actual, labels):
    output_labels = []
    output = []
    
    # Calcular e exibir a matriz de confusão 
    cm = confusion_matrix(actual, predicted, labels=labels)
    print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
    print(cm)
    
    # Calcule precisão, recall e precisão de  pontuação F1
    accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
    precision = precision_score(actual, predicted, average=None, labels=labels)
    recall = recall_score(actual, predicted, average=None, labels=labels)
    f1 = 2 * precision * recall / (precision + recall)
    output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
    output_labels.extend(['accuracy', 'precision', 'recall', 'F1'])
    
    # Calcule as versões macro da saída dessas métricas
    output.extend([[np.mean(precision)] * len(labels),
                   [np.mean(recall)] * len(labels),
                   [np.mean(f1)] * len(labels)])
    output_labels.extend(['macro precision', 'macro recall', 'macro F1'])
    
    # Encontre a matriz de confusão um x todos 
    cm_row_sums = cm.sum(axis = 1)
    cm_col_sums = cm.sum(axis = 0)
    s = np.zeros((2, 2))
    for i in range(len(labels)):
        v = np.array([[cm[i, i],
                       cm_row_sums[i] - cm[i, i]],
                      [cm_col_sums[i] - cm[i, i],
                       np.sum(cm) + cm[i, i] - (cm_row_sums[i] + cm_col_sums[i])]])
        s += v
    s_row_sums = s.sum(axis = 1)
    
    # Adicionar precisão média e precisão / recall / F1 
    avg_accuracy = [np.trace(s) / np.sum(s)] * len(labels)
    micro_prf = [float(s[0,0]) / s_row_sums[0]] * len(labels)
    output.extend([avg_accuracy, micro_prf])
    output_labels.extend(['average accuracy',
                          'micro-averaged precision/recall/F1'])
    
    # Calcule métricas para o classificador majoritário 
    mc_index = np.where(cm_row_sums == np.max(cm_row_sums))[0][0]
    cm_row_dist = cm_row_sums / float(np.sum(cm))
    mc_accuracy = 0 * cm_row_dist; mc_accuracy[mc_index] = cm_row_dist[mc_index]
    mc_recall = 0 * cm_row_dist; mc_recall[mc_index] = 1
    mc_precision = 0 * cm_row_dist
    mc_precision[mc_index] = cm_row_dist[mc_index]
    mc_F1 = 0 * cm_row_dist;
    mc_F1[mc_index] = 2 * mc_precision[mc_index] / (mc_precision[mc_index] + 1)
    output.extend([mc_accuracy.tolist(), mc_recall.tolist(),
                   mc_precision.tolist(), mc_F1.tolist()])
    output_labels.extend(['majority class accuracy', 'majority class recall',
                          'majority class precision', 'majority class F1'])
        
    # Precisão aleatória e kappa 
    cm_col_dist = cm_col_sums / float(np.sum(cm))
    exp_accuracy = np.array([np.sum(cm_row_dist * cm_col_dist)] * len(labels))
    kappa = (accuracy - exp_accuracy) / (1 - exp_accuracy)
    output.extend([exp_accuracy.tolist(), kappa.tolist()])
    output_labels.extend(['expected accuracy', 'kappa'])
    

    # Estimativa  aleatória
    rg_accuracy = np.ones(len(labels)) / float(len(labels))
    rg_precision = cm_row_dist
    rg_recall = np.ones(len(labels)) / float(len(labels))
    rg_F1 = 2 * cm_row_dist / (len(labels) * cm_row_dist + 1)
    output.extend([rg_accuracy.tolist(), rg_precision.tolist(),
                   rg_recall.tolist(), rg_F1.tolist()])
    output_labels.extend(['random guess accuracy', 'random guess precision',
                          'random guess recall', 'random guess F1'])
    
    # Estimativa  ponderada
    rwg_accuracy = np.ones(len(labels)) * sum(cm_row_dist**2)
    rwg_precision = cm_row_dist
    rwg_recall = cm_row_dist
    rwg_F1 = cm_row_dist
    output.extend([rwg_accuracy.tolist(), rwg_precision.tolist(),
                   rwg_recall.tolist(), rwg_F1.tolist()])
    output_labels.extend(['random weighted guess accuracy',
                          'random weighted guess precision',
                          'random weighted guess recall',
                          'random weighted guess F1'])

    output_df = pd.DataFrame(output, columns=labels)
    output_df.index = output_labels
                  
    return output_df

evaluation_results = []
for i, test_result in enumerate(test_results):
    print('\nSplit %d:' % (i+1))
    evaluation_result = Evaluate(actual = test_result['failure'],
                                 predicted = test_result['predicted_failure'],
                                 labels = ['none', 'comp1', 'comp2', 'comp3', 'comp4'])
    evaluation_results.append(evaluation_result)
```
