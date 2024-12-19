## SYMPY ##

# pip install sympy
# módulo para operações matemáticas com símbolos
import sympy

# para criar um símbolo
varSimb = sympy.Symbol('nomeSimb')

# para transformar uma string em uma expressão
nomeExpr = sympy.sympify(nomeString)

# para substituir apenas uma variável em uma expressão
Windef.subs(varSimb, valor)

# para substituir as variáveis em uma expressão
Fr = F.evalf(subs={varSimb1: valor1, varSimb2: valor2})

# para calcular uma integral indefinida em função de varSimb
sympy.integrate(nomeExpr, varSimb)

# para calcular uma integral definida (de A até B) em função de varSimb
sympy.integrate(nomeExpr, (varSimb, valorA, valorB))

# para calcular uma derivada em função de varSimb
sympy.diff(nomeExpr, varSimb)









## SCIPY ##

# pip install scipy
# módulo para operações matemáticas
from scipy import integrate
from scipy import signal
from scipy import io
from scipy.io import wavfile

# para realizar uma integral entre os intervalos A até B
def funcIntegrada(t, par1, par2)
	return # expressão em função de t, podendo utilizar os parêmtros passados
nomeIntegral = integrate.quad(funcIntegrada, valorA, valorB, args=(par1, par2))[0]
# obs: par1 e par2 se referem aos valores que serão passados para funcIntegrada
# obs2: os valores referentes a t serão passados pelo próprio scipy

# para convoluir dois vetores
signal.convolve(vetor1, vetor2)

# para convouluir duas matrizes
signal.convolve2d(matriz1, matriz2)

# para aplicar um sinal de entrada a um sistema (digital filter 1-D)
signal.lfilter(arrayX, arrayY, arrayEntrada)
# arrayX: são os coeficientes de x no sistema
# arrayY: são os coeficientes de y no sistema
# arrayEntrada: é o sinal de entrada a ser passado pelo filtro

# para abrir um arquivo .mat (arquivo de dados do MATLAB)
io.loadmat("endereçoDir/nomeArquivo.mat")
# obs: o objeto será do tipo dicionário

# para ler um arquivo de áudio .wav
wavfile.read("endereçoDir/nomeArquivo.wav")

# para escrever um vetor em um arquivo de áudio .wav
wavfile.write("endereçoDir/nomeArquivo.wav", freqAmostragem, nomeArray)
# obs: caso ocorrer algum problema, observar o formato do array original (int16, float32, etc)









## SKLEARN ##
from sklearn.linear_model import LinearRegression

# para realizar uma regressão linear
nomeReg = LinearRegression().fit(arrayX, arrayY)
# obs: x deve estar organizado em coluna e y em linha
# dica: fazer LinearRegression().fit(arrayX.reshape(-1,1), arrayY) se os dois arrays estiverem organizados em linhas

# para retornar os coeficientes da regressão (pesos)
nomeReg.coef_

# para retornar a interseção da regressão (bias)
nomeReg.intercept_

# para predizer o y a partir do valor de x
y = nomeReg.coef_ * arrayX + nomeReg.intercept_ # MODO1
y = nomeReg.predict(arrayX) # MODO2 (x deve estar organizado em coluna)









## NUMPY ##

# pip install numpy
# módulo para análise de dados
import numpy


# para que todo o array seja mostrado no prompt
numpy.set_printoptions(threshold=sys.maxsize)

# para se referir a um elemento do objeto numpy (array ou matriz)
nomeObjeto[valorLinha, valorColuna]
# obs: o primeiro elemento tem índice 0
# obs2: o último elemento tem índice -1
# obs3: não é possível armazenar um valor em uma posição inexistente

# para se referir a determinadas linhas e colunas (slice) de um objeto numpy (array ou matriz)
nomeObjeto[linhaInicial:linhaFinal+1:passo, colunaInicial:colunaFinal+1:passo]
# obs: nomeObjeto[valorLinha] (retorna linha inteira)
#obs2: nomeObjeto[:, valorColuna] (retorna coluna inteira)

# para transformar um objeto em um array
numpy.asarray(nomeObjeto)

# para transformar um objeto em uma matriz
numpy.asmatrix(nomeObjeto)



# para criar um array
nomeArray = numpy.array(valorLista, dtype='tipoDado') # array unidimensional
nomeArray = numpy.array([lista1, lista2, lista3], dtype='tipoDado') # array bidimensional
# obs: um array pode armazenar qualquer tipo de dado (int, float, string, etc)
# obs2: no array bidimensional, cada lista representará uma linha da matriz, e a posição do elemento na lista representa a coluna
# obs3: algumas operações entre arrays seguem regras diferentes das operações entre matrizes

# para criar um array com valores sequenciais (passo MENOR que 1)
nomeArray = numpy.arange(valorInicial, valorFinal+passo, passo)

# para criar um array com valores sequenciais (passo MAIOR que 1)
nomeArray = numpy.arange(valorInicial, valorFinal+1, passo)

# para inverter a ordem dos números de um array
numpy.flip(nomeArray)

# para gerar um array de valores aleatórios normalmente distribuídos
nomeArray = numpy.random.normal(loc=valorMedia, scale=valorDesvioPad, size=tamanhoArray)
# obs: os valores mais próximos da média terão maior probabilidade de ocorrência
# obs2: quanto menor o desvio padrão, mais alta e estreita será a curva (maior probabilidade de ocorrência de valores próximos à média)
# tamanhoArray: tamanho do array de números aleatórios gerado

# para gerar um array de valores aleatórios uniformemente distribuídos
nomeArray = numpy.random.uniform(low=limiteInferior, high=limiteSuperio, size=tamanhoArray)
# obs: todos os valores entre o mínimo e o máximo terão a mesma probabilidade de ocorrência
# tamanhoArray: tamanho do array de números aleatórios gerado

# para adicionar elementos no final de um array
numpy.append(nomeArray, [elemento1, elemento2])

# para concatenar arrays 
numpy.hstack((nomeArray1, nomeArray2, nomeArray3)) # um ao lado do outro
numpy.vstack((nomeArray1, nomeArray2, nomeArray3)) # um embaixo do outro

# para dividir um array em N pedaços
numpy.hsplit(nomeArray, valorN) # divide verticalmente
numpy.vsplit(nomeArray, valorN) # divide horizontalmente

# para que seja possível passar um array como parâmetro para uma função, devemos vetorizar a função
funcVetorizada = numpy.vectorize(nomeFunc)
# obs: podemos também utilizar list comprehension ou map para passar um array como parâmetro para uma função

# para salvar um array em um arquivo .npy
numpy.save('nomeArquivo', nomeArray)

# para carregar um array de um arquivo .npy
numpy.load('nomeArquivo.npy')



# para converter um array para uint8
novoArray = numpy.uint8(nomeArray)
# obs: transforma os elementos do array em um inteiro de 8 bits, ou seja, de 0 a 255
# obs2: todo elemento do tipo float é transformado em inteiro
# obs3: após essa transformação, todo elemento maior que 255, é SUBTRAÍDO 256 (N vezes), até ficar entre 0 e 255
# obs4: após essa transformação, todo elemento menor que 0, é ADICIONADO 256 (N vezes), até ficar entre 0 e 255

# para converter um array para float64
novoArray = numpy.float64(nomeArray)



# para criar uma matriz
nomeMatriz = numpy.matrix([lista1, lista2, lista3])
# obs: aconselhável utilizar matriz apenas para dados numéricos
# obs2: cada lista representará uma linha da matriz, e a posição do elemento na lista representa a coluna
# obs3: note que as listas devem possuir o mesmo número de elementos (mesmo length)
# obs4: as operações entre matrizes seguirão as regras matemáticas

# para criar matrizes com elementos iguais a 0
nomeArray = numpy.zeros([nMatrizes, nLinhas, nColunas])

# para criar matrizes com elementos iguais a 1
nomeArray = numpy.ones([nMatrizes, nLinhas, nColunas])

# para criar uma matriz diagonal com elementos iguais a 1
nomeArray = numpy.eye(nLinhas)

# para criar uma matriz diagonal definindo os elementos
nomeArray = numpy.diag(valorL1, valorL2, valorL3)

# para adicionar uma nova linha no final de uma matriz
numpy.append(nomeArray, [[valorC1, valorC2]], axis=0)

# para adicionar uma nova coluna no final de uma matriz
numpy.append(nomeArray, [[valorL1], [valorL2]], axis=1)

# para repetir uma matriz N vezes no sentido das linhas ou das colunas
numpy.tile(nomeArray, (NxLinhas, NxColunas))

# para gerar a matriz transposta
nomeMatriz.T

# para calcular o determinante de uma matriz
numpy.linalg.det(nomeMatriz)

# para gerar a inversa de uma matriz
numpy.linalg.inv(nomeMatriz)

# para transformar uma matriz em um vetor
nomeMatriz.ravel()



# para verificar o formato (linhas, colunas) de um objeto numpy (array ou matriz)
nomeObjeto.shape

# para para redimensionar um objeto numpy (array ou matriz)
nomeVetor.reshape(nLinhas, nColunas)

# para verificar o número de elementos de um objeto numpy (array ou matriz)
nomeObjeto.size

# para verificar o tipo de dado de um objeto numpy (array ou matriz)
nomeObjeto.dtype

# para converter os elementos de um objeto numpy para outro tipo de dado (array ou matriz)
nomeObjeto.astype(tipoDado)

# para ordenar os elementos de um objeto numpy por linha (array ou matriz)
nomeObjeto.sort(axis=1)

# para ordenar os elementos de um objeto numpy por coluna (array ou matriz)
nomeObjeto.sort(axis=0)


# para retornar o índice do elemento de maior valor em um objeto numpy
numpy.argmax(nomeObjeto)

# para retornar o índice do elemento de maior valor, em cada linha, em um objeto numpy
numpy.argmax(nomeObjeto, axis=1)

# para retornar o elemento de menor valor em um objeto numpy (array ou matriz)
nomeObjeto.max()

# para retornar o elemento de menor valor em um objeto numpy (array ou matriz)
nomeObjeto.min()

# para calcular a média dos elementos de um objeto numpy (array ou matriz)
nomeObjeto.mean()

# para calcular a variância dos elementos de um objeto numpy (array ou matriz)
nomeObjeto.var()

# para calcular o desvio padrão dos elementos de um objeto numpy (array ou matriz)
nomeObjeto.std()

# para verificar a soma de todos os elementos de um objeto numpy (array ou matriz)
nomeObjeto.sum()

# para contar quantas vezes determinado valor aparece em um objeto numpy (array ou matriz)
numpy.count_nonzero(nomeObjeto==valor)

# para verificar se todos os elementos de um objeto numpy (array ou matriz) possuem determinada condição
numpy.all(nomeObjeto operadorRelacional valor)

# para verificar o indice dos elementos que possuem determinada condição
numpy.where(nomeArray operadorRelacional valor)
# obs: colocar as condições entre parênteses quando utilizar operadores relacionais 
#exemplo: numpy.where((array > valor) & (array < valor))
# operadoresLogicos: | (or), & (and), ~ (not)

# para criar um array a partir do resultado de uma comparação
numpy.where(nomeArray operadorRelacional valor, valor1, valor2)
# obs: se a comparação com o elemento do array for True o novo array recebe valor1, caso contrário recebe valor2

# para verificar se determinados elementos (arrayElementos) estão contidos em um array (nomeArray)
numpy.isin(nomeArray, arrayElementos)



# operações e constantes com numpy
# obs: se o valor for um array, a operação acontecerá com CADA elemento do array
numpy.pi #constante pi
numpy.e #constante de Euler
numpy.exp(valor) #e^(valor)
numpy.log(valor) #ln(valor) (log na base e)
numpy.log10(valor) #log10(valor) (log na base 10)
numpy.sin(valor) #sin(valor) (em radianos)
numpy.cos(valor) #cos(valor) (em radianos)
numpy.tan(valor) #tan(valor) (em radianos)
numpy.arctan(valor) #arctan(valor) (retorna em radianos)

# para fazer operações com elemento por elemento de um objeto numpy (array ou matriz) por um valor específico
nomeObjeto + valor #soma
nomeObjeto - valor #subtração
nomeObjeto * valor #multiplicação
nomeObjeto / valor #divisão
nomeArray ** valor #potenciação: APENAS para arrays

# para fazer operações com elmentos de mesma posição entre dois arrays
nomeObjeto1 + nomeObjeto2 #soma
nomeObjeto1 - nomeObjeto2 #subtração
nomeObjeto1 / nomeObjeto2 #divisão
nomeArray1 * nomeArray2 #multiplicação: APENAS para arrays

# para fazer multiplicação de matrizes (linha vezes coluna)
# obs: apenas na multiplicação o resultado é diferente em relação a operação entre arrays
nomeMatriz1 * nomeMatriz2

# para fazer o somatório da multiplicação dos elementos de mesma posição de dois arrays
numpy.dot(array1, array2)

# para comparar elemento por elemento entre dois objetos numpy (array ou matriz), retornando True ou False
nomeObjeto1 > nomeObjeto2 #maior
nomeObjeto1 < nomeObjeto2 #menor
nomeObjeto1 >= nomeObjeto2 #maior igual
nomeObjeto1 <= nomeObjeto2 #menor igual
nomeObjeto1 == nomeObjeto2 #igual
nomeObjeto1 != nomeObjeto2 #diferente

# para fazer operações lógicas com elemento por elemento de dois objetos numpy (array ou matriz)
nomeObjeto1 & nomeObjeto2 #and
nomeObjeto1 | nomeObjeto2 #or
numpy.logical_not(nomeObjeto) # not



# para retornar a transformada de fourier de um array 1D
fourierArray = numpy.fft.fft(nomeArray)

# para retornar a transformada de fourier de um array 2D
fourierArray = numpy.fft.fft2(nomeArray)

# para retornar a transformada de fourier inversa de um array 1D
inversaArray = numpy.fft.ifft(nomeArray)

# para retornar a transformada de fourier inversa de um array 2D
inversaArray = numpy.fft.ifft2(nomeArray)

# para retornar os ângulos de fase de um array de números complexos
arrayAngulo = numpy.angle(nomeArray)

# para retornar os módulos de um array de números complexos
arrayModulo = numpy.abs(nomeArray)

# para retornar apenas a parte real de um array de números complexos
arrayReal = numpy.real(nomeArray)

# para retornar apenas a parte imaginária de um array de números complexos
arrayImag = numpy.imag(nomeArray)









## PANDAS ##

# pip install pandas
# módulo para análise de dados
import pandas

# para importar um dataframe a partir de um csv
nomeDF = pandas.read_csv('nomeArquivo.csv', sep='caracterDelimitador', index_col=nColuna)
# obs: se o parâmetro index_col for utilizado, o índice da coluna passado para o parâmetro, será utilizada como índice das linhas,
#ou seja, se quiser utilizar a primeira coluna como o índice das linhas, basta fazer index_col=0

# para importar um dataframe a partir de um arquivo excel (.xlsx)
nomeDF = pandas.read_excel('nomeArquivo.xlsx', index_col=nColuna)


# para salvar um dataframe em um csv
nomeDF.to_csv('nomeArquivo.csv', sep='caracterDelimitador')

# para criar um dataframe
nomeDF = pandas.DataFrame({'nomeC1':[valorL1C1, valorL2C1], 'nomeC2':[valorL1C2, valorL2C2]}, index=['nomeL1', 'nomeL2'])
nomeDF = pandas.DataFrame([conjuntoL1, conjuntoL2], index=['nomeL1', 'nomeL2'], columns=['nomeC1', 'nomeC2'])

# para visualizar as N primeiras linhas de um dataframe
nomeDF.head(valorN)
# obs: utilizar este comando quando o dataframe aparecer com uma parte ocultada

# para visualizar as N últimas linhas de um dataframe
nomeDF.tail(valorN)

# para se referir a determinado elemento do dataframe
nomeDF['nomeColuna'][indiceLinha]

# para retornar colunas específicas de um dataframe
nomeDF[['nomeColuna1', 'nomeColuna2']]

# para retornar uma linhas e colunas específicas de um dataframe pelo NOME 
nomeDF.loc[nomeLinhaInicial:nomeLinhaFinal+1:passo, ['nomeColuna1', 'nomeColuna2']] # MODO1
nomeDF.loc[[nomeLinha1, nomeLinha2], ['nomeColuna1', 'nomeColuna2']] # MODO2
# obs: o nome da linha é aquela que se encontra no dataframe (index)

# para retornar uma linhas e colunas específicas de um dataframe pela POSIÇÃO
nomeDF.iloc[posLinhaInicial:posLinhaFinal+1:passo, posColunaInicial:posColunaFinal+1:passo]
nomeDF.iloc[[posLinha1, posLinha2], [posColuna1, posColuna2]]
# obs: a posição da linha deve ser a posição real, e NÃO aquela que se encontra no dataframe após linhas serem excluídas

# para retornar apenas os elementos que obedecem a uma condição em determinada coluna
nomeDF[nomeDF['nomeColuna'] operadorRelacional valor] 
# obs: ou seja, nomeDF precisará de uma lista de valores booleanos (True ou False), e com base nisso selecionará as linhas do dataframe
# obs2: colocar as condições entre parênteses quando utilizar operadores relacionais 
#exemplo: df[(df['nomeColuna1'] > valor) & (df['nomeColuna2'] < valor)]
# operadoresLogicos: | (or), & (and), ~ (not)

# para retornar apenas os elementos que existem valores iguais em uma determinada lista
nomeDF[nomeDF['nomeColuna'].isin(listaValores)] 
# obs: essa condição pode ser mesclada com a condição acima (envolvendo operadorRelacional), basta utlizar operadores lógicos

# para aplicar uma determinada função em uma coluna do dataframe
nomeDF['nomeColuna'].apply(nomeFunc)

# para retornar o dataframe ordenado de acordo com determinada coluna
nomeDF.sort_values('nomeColuna', ascending=valorBool)
# valorBool: True (ordem ascendente, crescente), False (ordem descendente, decrescente)

# para retornar uma amostra aleatória do dataframe
nomeDF.sample(tamanhoAmostra)

# para renomear os nomes de algumas colunas
nomeDF = nomeDF.rename(columns={'Nome Coluna Antiga 1': 'Nome Coluna Nova 1', 'Nome Coluna Antiga 2': 'Nome Coluna Nova 2'})

# para substituir os valores de determinada coluna por outros
nomeDF['nomeColuna'] = nomeDF['nomeColuna'].replace([valorAntigo1, valorAntigo2], [valorNovo1, valorNovo2])

# para inverter as colunas pelas linhas (transposta do dataframe)
nomeDF.T

# para incluir uma coluna em um dataframe
nomeDF.insert(valorPosicao, 'nomeColuna', arrayColuna)

# para adicionar uma linha ao dataframe
 nomeDF.append({'nomeC1':valorC1, 'nomeC2':valorC2, 'nomeC3':valorC3}, ignore_index=True)

# para remover determinada coluna do dataframe
del nomeDF['nomeColuna']

# para remover determinadas linhas do dataframe
nomeDF = nomeDF.drop(listaNomesLinhas)

# para verificar a quantidade de dados faltantes em um dataframe
nomeDF.isnull().sum()


# para criar um array com uma sequência de datas
pandas.date_range(dataInicial, dataFinal, freq="valorFreq")
# valorFreq: d (dia), m (mês), y (ano)

# para transformar a data de uma coluna do tipo string para o tipo datetime
nomeDF['nomeColuna'] = pandas.to_datetime(nomeDF['nomeColuna'], format='%d/%m/%Y')
# obs: colocar %d, %m e %Y, na mesma ordem que estiver na string e com o mesmo caracter separador (alterar a barra pelo caracter)

# para transformar o tipo de dado de uma coluna em dados categóricos
nomeDF['nomeColuna'] = nomeDF['nomeColuna'].astype('category')

# para verificar as categorias de uma coluna de dados categóricos
nomeDF['nomeColuna'].cat.categories
# obs: o código de cada categoria está relacionado com a ordem em que a categoria aparece
# ex: a 1ª categoria mostrada com o comando acima, terá o código igual a 0

# para reordenar as categorias
nomeDF['nomeColuna'] = nomeDF['nomeColuna'].cat.reorder_categories(['categoria1', 'categoria2', 'categoria3'])
# obs: útil para as categorias ficarem organizadas em um gráfico

# para transformar colunas do tipo string em numéricas
nomeDF['nomeColuna'] = nomeDF['nomeColuna'].cat.codes



# para se referir apenas aos valores de um dataframe
nomeDF.values

# para se referir apenas aos índices das linhas de um dataframe
nomeDF.index

# para se referir apenas aos nomes das colunas de um dataframe
nomeDF.columns

# para contar o número de registro em cada coluna do dataframe
nomeDF.count()



# para verificar a correlação entre as colunas de um dataframe
nomeDF.corr('tipoCorrelacao')
# obs: apenas as colunas númericas serão correlacionadas (transformar colunas do tipo string em numéricas)
# tipoCorrelacao: pearson (NÃO pode ser utilizado para correlações não lineares)
#                 spearman (pode ser utilizado para correlações não lineares)

# para verificar o elemento de maior valor em um dataframe
nomeDF['nomeColuna'].max()

# para verificar o elemento de maior valor em um dataframe
nomeDF['nomeColuna'].min()

# para calcular a média dos elementos de um dataframe
nomeDF['nomeColuna'].mean()

# para calcular a mediana dos elementos de um dataframe
nomeDF['nomeColuna'].median()

# para calcular a moda dos elementos de um dataframe
nomeDF['nomeColuna'].mode()

# para calcular a variância dos elementos de um dataframe
nomeDF['nomeColuna'].var()

# para calcular o desvio padrão dos elementos de um dataframe
nomeDF['nomeColuna'].std()

# para retornar determinado quartil
nomeDF['nomeColuna'].quantile(valorQuartil)
# valorQuartil: 1º quartil (0.25), 2º quartil (0.5), 3º quartil (0.75)

# para verificar a assimetria dos dados
# obs: assimetria do histograma referente a uma curva normal
nomeDF['nomeColuna'].skew()
# 0: distribuição normal
# positiva: assimétrica à direita
# negativa: assimétrica à esquerda

# para verificar a curtose dos dados
# obs: curtose do histograma (se a curva é afunilada ou achatada) em relação a uma curva normal
nomeDF['nomeColuna'].kurtosis()

# para verificar a soma de todos os elementos de um dataframe
nomeDF['nomeColuna'].sum()

# para retornar apenas os valores únicos de um dataframe
nomeDF['nomeColuna'].unique()

# para mostrar um resumo dos dados estatísticos de um dataframe
nomeDF['nomeColuna'].describe()




# para agrupar linhas que possuem dados iguais em determinada coluna
nomeDF.value_counts('nomeColuna')
# obs: os grupos são ordenados pelo número de ocorrências de determinado valor

# para visualizar a quantidade de grupos formados
nomeDF.value_counts('Final de Semana').size

# para agrupar linhas que possuem dados iguais em determinada coluna
nomeDF.groupby('nomeColuna').size()
# obs: os grupos são ordenados pelo valor da coluna

# para tirar a média dos agrupamentos em determinada coluna
nomeDF.groupby('ColunaAgrupamento')['ColunaMedia'].mean()

# para ordenar um objeto groupby
objGroupby.sort_values(ascending=True) # ordem crescente
objGroupby.sort_values(ascending=False) # ordem decrescente

# para agrupar linhas que possuem dados iguais em várias colunas, selecionando o menor valor e o maior valor em algumas colunas
# obs: dessa maneira, as linhas que possuirem valores iguais (o valor de uma linha for igual ao valor da outra linha),
#em todas as colunas ColunaGroup, farão parte de um mesmo grupo, sendo informado o menor valor e o maior valor em cada ColunaLimiar
# obs2: os dados serão ordenados de acordo com o valor das colunas: ColunaGroup e nova_colunaLimiar
groups = nomeDF.groupby(["ColunaGroup1", "ColunaGroup2", "ColunaGroup3"]).agg(
nova_colunaLimiar1_min=("ColunaLimiar1", "min"),
nova_colunaLimiar1_max=("ColunaLimiar1", "max"),
nova_colunaLimiar2_min=("ColunaLimiar2", "min"),
nova_colunaLimiar2_max=("ColunaLimiar2", "max"),
).reset_index().sort_values(["ColunaGroup", "nova_colunaLimiar"])

# para unir dois dataframes através de determinada coluna
pandas.merge(nomeDF1, nomeDF2, on="nomeColuna")

# para unir dois dataframes pelo index
pandas.merge(nomeDF1, nomeDF2, left_index=True, right_index=True)


# para plotar um gráfico relacionado a colunas do dataframe
nomeDF.plot(kind='tipoPlot', x='nomeColunaX', y='nomeColunaY', c='nomeColunaC', cmap=listaCores)
# obs: utilizar pyplot.show() após o comando acima
# tiposPlot: scatter=pontos
#            line=linha 
#            pie=pizza
#            area=área colorida
#            density=densidade
#            hist=histograma
#            bar=barras verticais
#            barh=barras horizontais
# nomeColunaC: coluna do dataframe para classificar o par ordenado (x,y), sendo que cada classe terá uma cor

# para plotar um boxplot relacionado a colunas do dataframe
nomeDF.boxplot(column='nomeColunaPlotada', by='nomeColunaAgrupamento')
# nomeColunaPlotada: nome da coluna que será plotada nos boxplots
# nomeColunaAgrupamento: serão plotado um boxplot para cada agrupamento desta coluna









## PLOT ##

# pip install matplotlib
# módulo para plotagem de gráficos
from matplotlib import pyplot

# para plotar um gráfico de linha
pyplot.plot(conjuntoX, conjuntoY, linestyle='tipoLinha', linewidth=espessuraLinha, color='cor', zorder=nPlanoZ)
# obs: a espessura padrão da linha é 1.5
# cor: pelo nome (green, blue, red, black, etc) ou hexadecimal (#Red(00-ff)Green(00-ff)Blue(00-ff))
# tipoLinha: -:linha contínua
#       	 --: linha tracejada

# para personalisar um cmap (color map)
from matplotlib.colors import ListedColormap
listaCores = ListedColormap(['#hexaCor1', '#hexaCor2', '#hexaCor3'])

# para plotar um gráfico de pontos
nomeScatter = pyplot.scatter(conjuntoX, conjuntoY, marker='tipoPonto', s=tamanhoPonto, color='cor', c=conjuntoClassific, cmap=listaCores, zorder=nPlanoZ)
# obs: se o parâmetro color for usado, os parâmetros c e cmap não poderão ser usados
# obs2: o parâmetro c fornecerá o array para classificar o par ordenado (x,y), sendo que cada classe terá uma cor, de acordo com cmap
# colorMap: bwr=(azul e vermelho); winter=(verde e azul); Set1=(conjunto de cores)
# tipoPonto: *: asterisco
#            o: ponto maior
#            .: ponto menor
#            s: quadrado
#            ^: triângulo
#			 d: losango

# para plotar um gráfico de uma função discreta
pyplot.stem(conjuntoX, conjuntoY)

# para plotar um histograma
pyplot.hist(conjunto, bins=nBarras)
# obs: utilizado para variáveis quatitativas
# nBarras: número de barras que irá aparecer no gráfico (número de divisões)

# para plotar um gráfico de barras
pyplot.bar(conjunto, range(len(conjunto)), labels=listaLabels, width=espessuraBarra, color='cor')
# obs: utilizado para variáveis categóricas

# para plotar um boxplot
pyplot.boxplot([conjunto1, conjunto2, conjunto3], labels=['nomeConj1', 'nomeConj2', 'nomeConj3' ])
# 1º quartil: início da caixa
# mediana: risquinho dentro da caixa
# 3º quartil: final da caixa
# IQR: distância entre 3º quartil e 1º quartil
# outliers: pontos posteriores aos limites superior e inferior

# para exibir uma legenda referente a vários gráficos de linha
pyplot.legend(['nomeLinha1', 'nomeLinha2'], title="Título Legenda", loc="tipoLugar")

# para exibir uma legenda referente a um scatter plot
pyplot.legend(*nomeScatter.legend_elements(), title="Título Legenda", loc="tipoLugar")
# tipoLugar: center right; upper right

# para inserir um título no gráfico
pyplot.title("Título Gráfico")

# para exibir labels específicos no gráfico plotado
pyplot.xlabel("Título Eixo X")
pyplot.ylabel("Título Eixo Y")

# para definir os valores exibidos no eixo x
pyplot.xticks(listaValores, labels=listaStrings)
# listaValores: lista com os valores que serão marcados no eixo x
# listaStrings: lista com as strings que serão exibidas nos pontos marcados

# para definir os valores exibidos no eixo y
pyplot.yticks(listaValores, labels=listaStrings)
# listaValores: lista com os valores que serão marcados no eixo y
# listaStrings: lista com as strings que serão exibidas nos pontos marcados

# para grades (várias linhas horizontais e verticais) no plot
pyplot.grid(linewidth=0.2, color='cor')

# para traçar uma linha horizontal
pyplot.axhline(y=valorY, linestyle='tipoLinha', linewidth=espessuraLinha, color='cor')

# para traçar uma linha vertical
pyplot.axvline(x=valorX, linestyle='tipoLinha', linewidth=espessuraLinha, color='cor')

# para exibir uma escala específica
pyplot.axis([valorInicialX, valorFinalX, valorInicialY, valorFinalY])

# para fazer um subplot (exibir dois plots em uma mesma janela)
pyplot.subplot(nLinhas, nColunas, posPlot)
# obs: os códigos referentes a cada plot deve ficar abaixo da função subplot
# nLinhas: nº de linhas do subplot
# nColunas: nº de colunas do subplot
# posPlot: posição do plot atual (1, 2, ..., nLinhas*nColunas)

# para plotar uma imagem
pyplot.imshow(arrayImagem)

# para salvar uma imagem
pyplot.imsave('endereçoDir', arrayImagem)

# para plotar vários gráficos de uma só vez, utilizar o comando abaixo ANTES de cada plot
pyplot.figure()
# obs: mesmo utilizando o figure antes, é necessário utilizar o pyplot.show() posteriormente

# para visualizar os gráficos plotados
pyplot.show()

# para gerar um plot animado
# obs: utilizando pyplot.pause não tem necessidade de usar pyplot.show
while cond: # ou laço for
    pyplot.clf() # limpa o plot anterior
    # códigos de plotagem (plot, scatter, etc)
    pyplot.pause(tempoDalay) # delay em segundos entre um plot e outro




## PLOTLY ##

# pip install plotly
# módulo para plotagem de gráficos
import plotly.graph_objects as pgo
import plotly.io as pio

# para plotar um gráfico de candles
nomeGrafico = pgo.Figure(data=pgo.Candlestick(x=conjuntoTempo, open=conjuntoAbertura, high=conjuntoMaximas, low=conjuntoMinimas, close=conjuntoFechamentos))

# para plotar um gráfico de linha
nomeGrafico = pgo.Figure(data=pgo.Scatter(x=conjuntoX, y=conjuntoY))

# para mostrar o gráfico plotado
nomeGrafico.show()

# para plotar e mostrar um gráfico de maneira mais específica (modelo ReactJS)
pio.show({
'data': { 'x': conjuntoX,
          'y': conjuntoY,
           'type': 'tipoPlot'}, # tiposPlot: scatter, candlestick, bar
'layout': {'dragmode': 'pan', # iniciar o gráfico com o ponteiro pan (para arrastar)
         	'width': valorLargura, 
            'height': valorAltura,
            'title': 'Título Gráfico',
            'yaxis': {'showticklabels': valorBool, 'dtick': valorIntervalo}, # showticklabels (mostra os valores das marcações), dtick (intervalo entre as marcações)
            'xaxis': {'showticklabels': valorBool, 'dtick': valorIntervalo},  # showticklabels (mostra os valores das marcações), dtick (intervalo entre as marcações)
            'margin': {'l':valorMargemLeft, 'r':valorMargemRight, 'b':valorMargemBottom, 't':valorMargemTop}}},
config={'displayModeBar': valorBool, 'scrollZoom': valorBool}) # displayModeBar (mostra caixa de ferramenta no gráfico), scrollZoom (permite zoom ao rolar o mouse)




## MLXTEND

# pip install mlxtend
# módulo para plotagem de gráficos
from mlxtend.plotting import plot_decision_regions

# para plotar pontos de acordo com um algorítmo de classificação (rede neural)
plot_decision_regions(arrayPlot, arrayClassfic, clf=objetoClassific)
# arrayPlot: array em NUMPY com o valor dos pares ordenados a serem plotados, normalmente o conjunto de dados de treino
#referente a duas das entradas do classificador [(x1,x2), (x1,x2), (x1,x2)]
# arrayClassific: array em NUMPY com o conjunto de dados de treino referente a saída do classificador [y, y, y, y]
# objetoClassific: objeto instanciado a partir da classe do classificador (rede neural)
# obs: de acordo com os dados de treino de saída, será mostrado diferentes estilos de pontos para os pares ordenados de entrada e,
#de acordo com o método predict do o objeto classificador, será mostrada uma região para cada classe, ou seja, essa região será
#definida de acordo com a equação de separação dos dados encontrada pelo classificador
# obs2: note que o classificador deve ter um método chamado predict, que irá prever a saída com base nos dados de entrada
# obs3: lembre que é necessário treinar o classificador antes de passá-lo para esta função de plotagem




## MLXTEND

# pip install seaborn
import seaborn

# para plotar uma matriz de confusão das correlações
seaborn.heatmap(nomeDF.corr('spearman'), annot=True)