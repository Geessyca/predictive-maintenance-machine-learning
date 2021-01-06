import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


telemetry = pd.read_csv('PdM_telemetry.csv')
errors = pd.read_csv('PdM_errors.csv')
maint = pd.read_csv('PdM_maint.csv')
failures = pd.read_csv('PdM_failures.csv')
machines = pd.read_csv('PdM_machines.csv')


#Alterar dados refrente a data e hora
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")
errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
maint['datetime'] = pd.to_datetime(maint['datetime'], format="%Y-%m-%d %H:%M:%S")
failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")


#linhas nulas
telemetry.isnull().sum()
errors.isnull().sum()
maint.isnull().sum()
failures.isnull().sum()
machines.isnull().sum()

#substituindo pelo valor  - NÃO TEVE NECESSIDADE
    #dadomedio = df['dado'].mean()
    #dadomedio = math.floor (dadomedio)

    #df.update(df['dado'].fillna(dadomedio))
    
#Telemetry gráficos 1º ano

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

#################################################
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

#################################################
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

#################################################
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

#Errors gráfico

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
errors['errorID'].value_counts().plot(kind='bar')
plt.title('Quantidade de erros ao logo da coleta de dados')
plt.ylabel('Nº de erros')

#Machines gráfico

sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
_, bins, _ = plt.hist([machines.loc[machines['model'] == 'model1', 'age'],
                       machines.loc[machines['model'] == 'model2', 'age'],
                       machines.loc[machines['model'] == 'model3', 'age'],
                       machines.loc[machines['model'] == 'model4', 'age']],
                       20, stacked=True, label=['model1', 'model2', 'model3', 'model4'])
plt.title('Maquina e seu tempo')
plt.xlabel('Idade')
plt.ylabel('Nº de Máquinas')

#Failures gráfico

sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
failures['failure'].value_counts().plot(kind='bar')
plt.ylabel('Nº de componentes')

#Maint gráfico

sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
maint['comp'].value_counts().plot(kind='bar')
plt.ylabel('Nº de componentes')

#Recursos de atraso de 3 horas

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




#Recursos de atraso de 24 horas
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
