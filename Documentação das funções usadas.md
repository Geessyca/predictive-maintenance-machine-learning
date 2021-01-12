Pandas:

df = pd.read_csv() - > Ler um arquivo de valores separados por vírgula (csv) no DataFrame.

pd.to_datetime(df, format="%Y-%m-%d %H:%M:%S") - > Converte a coluna DataFrame para DataTempo.

df.isnull().sum() - > Detecta valores ausentes.

df['dado'].mean() - > Calcula a média dos valores em uma coluna.

math.floor() - > Arrendonda o valor.

df.update(df['dado'].fillna()) - > Adicona o valor ao valores ausentes.

df.loc - > Acesse um grupo de linhas e colunas por rótulo.

df.append () - > Acrescente linhas de outro ao final do quadro.

pd.pivot_table() - > Crie uma tabela dinâmica no estilo de planilha como um DataFrame.

.rolling(window=3) - > Cria uma janela de 3 horas nos dados.

.mean() - Calcula a media.

.resample() - > Conversão de frequência e reamostragem de séries temporais.

.unstack() - > Dinamize um nível dos rótulos de índice.

pd.concat() - > Junda os dados ao longo das colunas (axis=1), retorna um DataFrame.

.columns = []  - > Nomeia as colunas do dataframe.

.reset_index(inplace=True) - > Retorna DataFrame sem nenhum índice.
                                                                              
.std() - > Calcula o desvio padrão.

.iloc() - > Indexação puramente baseada em localização inteira para seleção por posição.

.dropna() - > Remove os valores ausentes.

pd.get_dummies() - > Converta variáveis categóricas em variáveis  indicadoras.

.set_index() - > Defina o índice DataFrame usando colunas existentes.

.groupby() - > Agrupe DataFrame usando um mapeador ou por uma série de colunas.

.merge() - > Mescle DataFrame ou objetos Series nomeados com uma junção de estilo de banco de dados.

.fillna() - > Preencha os valores NA / NaN usando o método especificado.

.sum() - > Adiciona os itens de um iterável e retorna a soma.

Matplotlib

.sort_values() - > Classifique pelos valores ao longo de qualquer eixo.

sns.set_style() - > Define o estilo estético dos enredos.

plt.figure(figsize=(x, y)) - > Define o tamanho da figura (polegadas).

plt.title() - > Adciona titulo ao gráfico.

plt.plot(x, y) - > Plota o gráfico.

plt.ylabel() - > Nomeia o eixo y.

plt.gca().get_xaxis().get_major_formatter() - > Ajusta o gráfico. 

adf.scaled[] - > Definir uma escala.

plt.xlabel() - > Nomeia o eixo x.

df.value_counts().plot(kind='bar') - > Retorne uma série contendo contagens de valores únicos e cria um grafico em barras.

plt.hist() - > cria um histograma.

Numpy

np.timedelta64 () - > Representa uma duração, a diferença entre datas.

array() - > Cria uma matriz.

.sum() - > Soma dos elementos da matriz em um determinado eixo.

.zeros() - > Retorne uma nova matriz de forma e tipo fornecidos, preenchida com zeros .

.trace() - > Retorne a soma ao longo das diagonais da matriz.

.where() - > Retorna elementos escolhidos de x ou y dependendo da condição .

.max() - > Retorna o máximo de uma matriz ou máximo ao longo de um eixo.

.ones() - > Retorne um novo array de forma e tipo fornecidos, preenchido com uns .

Sklearn

confusion_matrix() - > Calcule a matriz de confusão para avaliar a precisão de uma classificação.

precision_score() - > Calcule a precição 

recall_score() - > Calcula o recall que é a proporção em que está o número de verdadeiros positivos e o número de falsos negativos.

accuracy_score() - > Pontuação de classificação de precisão.

.extend() - > Estende a lista anexando todos os itens do iterável.
