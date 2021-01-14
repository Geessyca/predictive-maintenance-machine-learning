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

<h4>Mesclar os dados acima</h4>

```
telemetry_feat  =  pd.concat ([ telemetrymean ,
                             telemetrysd.iloc [:, 2 : 6 ],
                             telemetrymean_24hrs.iloc [:, 2 : 6 ],
                             telemetrysd_24hrs.iloc [:, 2 : 6 ]], axis = 1 ).dropna ()
```

<h4>Recursos de atraso de errors com janela de 24hrs</h4>
<h6>
```
error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
error_count = error_count.groupby(['machineID','datetime']).sum().reset_index()
error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)

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

<h2>Características da máquina</h2>

```
final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(machines, on=['machineID'], how='left')
```

<h2>Construção de etiqueta</h2>

```
labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
labeled_features = labeled_features.fillna(method='bfill', limit=7) # fill backward up to 24h
labeled_features = labeled_features.fillna('none')
```



