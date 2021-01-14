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

<b>Errors:</b> Erros ininterruptos lançados enquanto a máquina ainda está operacional e não constituem falhas. A data e as horas do erro são arredondadas para a hora mais próxima, pois os dados de telemetria são coletados por hora.

<b>Maint:</b> Registros de manutenção programada e não programada que correspondem tanto à inspeção regular dos componentes quanto às falhas. Um registro é gerado se um componente for substituído durante a inspeção programada ou substituído devido a uma avaria. Os registros criados devido a falhas serão chamados de falhas, o que é explicado nas seções posteriores.

<b>Machines:</b> Informações sobre as máquinas como de modelo e idade (anos de serviço).

<b>Failures:</b> Substituições de componentes devido a falhas. Cada registro possui uma data e hora, ID de máquina e tipo de componente com falha.

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









