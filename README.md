# Analise e pratica de machine learning

- Link para o banco de dados:

https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv

- GitHub fonte: https://github.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy

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
