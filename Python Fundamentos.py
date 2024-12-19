## PIP E AMBIENTE VIRTUAL ##

# instalar o pip para ser possível instalar módulos python (prompt)
# python get-pip.py

# para instalar um módulo (prompt): pip install nomeModulo
# para verificar a versão de um módulo (prompt): pip show nomeModulo
# para instalar uma versão específica de um módulo: pip install nomeModulo==nVersao
# para atualizar um módulo (prompt): pip install nomeModulo -U
# para desinstalar um módulo (prompt): pip uninstall nomeModulo
# para listar todas as versões de um módulo, basta tentar instalar uma versão que não existe (prompt): 
#pip install nomeModulo==nVersaoInexistente

# instalar o virtualen para criar ambientes virtuais (prompt)
# pip install virtualenv

# para criar um ambiente virtual (prompt)
# cd pastaDestino
# python -m venv ambientevirtual

# para ativar o ambiente virtual (prompt)
# cd pastaDestino/Scripts
# activate

# para atualizar o pip (prompt)
# python -m pip install --upgrade pip









## TAMANHO DE DADOS ##
# float = 24 bytes
# int = 28 bytes
# bool = 28 bytes
# complex = 32 bytes
# string = (49 + nCaracteres) bytes
# lista = (56 + 8*nElementos) bytes









## GERAL ##

# int: valor
# float: valor
# complex: valorR + valorIj
# bin: '0bValor'
# hex: '0xValor'

# string: "Texto"
# bytes: b'["Texto"]'

# False: 0
# True: qualquer valor diferente de 0

# para visualizar as instruções sobre alguma função
help(nomeFunc)

# para imprimir uma mensagem no console
print(valor1, valor2, valor3, sep="caracterSep")
# obs: valor1, valor2 e valor3 serão concatenados e separados por caracterSep

# para criar uma string limitando o número de casas decimais de um valor
'{:.Nf}'.format(valor) # float com N casas decimais
'{:.Ne}'.format(valor) # notação científica com N casas decimais

# para declarar uma variável
nomeVar = valor
# obs: quando uma variável aponta para um objeto, qualquer variável que recebê-lá também
#apontará para este objeto, logo qualquer alteração no OBJETO afetará ambas as variáveis

# para armazenar a string inserida no console em uma variável
nomeVar=input("Texto EXIBIDO na tela") # python 3
nomeVar=raw_input("Texto EXIBIDO na tela") # python 2

# para que uma variável armazene o restante dos elementos de um conjunto usamos asterisco (*) 
nomeVar1, nomeVar2, *nomeVarResto, nomeVarN = valorConjunto
# obs: valorConjunto pode ser um range, lista, tupla

# para deletar um objeto (ou variável)
del nomeObj
# obs: tudo em python é um objeto, inclusive as variáveis

# para verificar o tipo do objeto
type(valor)

# para transformar o tipo de dado
str(valor) #para string
int(valor) #para int (inteiros)
float(valor) #para float (reais)
complex(valorReal, valorImg) #para complexos
bool(valor) #para bool
list(valor) #para lista
tuple (valor) #para tupla
range(valor) #para range
bytes(valorNumerico) #valor numérico para bytes
bytes(valorString, 'utf-8') #string para bytes
valorString.encode() #string para bytes
nomeByte.decode() #bytes para string
bin(valor) # binário em string
hex(valor) #hexadecimal em string



# para saber o código ASCII de determinado caracter
ord("caracter")

# para retornar o caracter referente ao código ASCII informado
chr(valorASCII)

# para CONTINUAR o restante da string na linha de baixo usamos uma barra invertida (\)
"texto 1 \
texto2"
# obs: a barra invertida não será exibida no console
# obs: isso não resultara em uma quebra de linha

# para gerar uma quebra de linha em uma string usamos barra-n (\n)
"textoLinha1\ntextoLinha2"
# obs: a quebra de linha será visível apenas no comando print

# para expressar um caracter reservado em uma string basta antecedê-lo uma barra invertida (\)
"\caracterReservado"
# obs: exemplos de caracteres reservados (\  "  \n)

# para expressar aspas em uma string basta usar aspas simples e aspas duplas
"''" #neste caso as aspas simples serão exibidas
'""' #neste caso as aspas duplas serão exibidas
# obs: dessa forma não há necessidade de usar barra invertida para exibí-la

# quando usamos aspas triplas as quebras de linha são exibidas na string
("""textoLinha1
textoLinha2
textoLinha3""")

# para concatenar string a um tipo numérico
f'Texto da string {varNum} resto do texto' #MODO1
'Texto da string '+str(varNum)+' resto do texto' #MODO2



# operadores aritméticos
valorDividendo//valorDivisor #divisão retornando int
valorDividendo%valorDivisor #resto da divisão
valorBase**valorExpoente #potenciação
valor1+=valor2 #valor1=valor1+valor2
valor1-=valor2 #valor1=valor1-valor2
valor1*=valor2 #valor1=valor1*valor2
valor1/=valor2 #valor1=valor1/valor2

# operadores lógicos
valor1 and valor2 #and
valor1 or valor2 #or
not valor #not
valor1 is valor2 #verifica se os dois valores são EXATAMENTE os mesmos
valor1 in [valor2,valo3,valo4] #verifica se um valor está contido em um conjunto

# para retornar o valor absoluto de um valor
abs(valor)

# para arredondar um número
round(valor, nCasasDecimais)



# módulo para auxiliar em operações matemáticas
import math

# para retornar a constante pi
math.pi

# para retornar a constante de Euler
math.e



# módulo para manipular datas
import datetime
import time

# para retornar a data e hora atual
datetime.datetime.now()
# obs: (ano, mês, dia, hora, minuto, segundo, milisegundo)

# para criar uma data específica
objDatetime = datetime.datetime(valorAno, valorMes, valorDia, valorHora, valorMinuto, valorSegundo, valorMS)

# para retornar apenas a hora
objDatetime.time()

# para retornar apenas a data
objDatetime.date()

# para retornar o dia da semana
objDatetime.weekday()
# obs: 0 (seg), 1 (ter), 2 (qua), 3 (qui), 4 (sex), 5 (sab), 6 (dom)

# para retornar um atributo específico do objeto datetime
objDatetime.year # ano
objDatetime.month # mês
objDatetime.day # dia
objDatetime.hour # hora
objDatetime.minute # minuto
objDatetime.second # segundo
objDatetime.microsecond # microsegundo

# para somar um delta a um datetime 
objDatetime + datetime.timedelta(nomeAtributo=valorSomado)
# nomeAtributo: hours, days, months, year

# para alterar o objeto datetime
objDatetime = objDatetime.replace(nomeAtributo=novoValor)

# para passar o horário de um objeto datetime para string
objDatetime.time().strftime('%H:%M:%S')

# para passar a data de um objeto datetime para string
objDatetime.time().strftime('%d/%m/%Y')

# para passar uma string para um objeto datetime
datetime.datetime.strptime(valorString,'%d/%m/%Y')
# obs: colocar %d, %m e %Y, na mesma ordem que estiver na string e com o mesmo caracter separador (alterar a barra pelo caracter)

# para passar um objeto datetime para timestamp (formato PostgreSQL)
datetime.datetime.timestamp(nomeData)

# para capturar o segundo atual
time.time()

# para gerar um delay
time.sleep(valorSegundo)









## CONJUNTOS ##

# para criar uma lista
nomeLista = [valor1, valor2, valor3]
# obs: em uma mesma lista pode ser armazenados dados de tipos diferentes

# para se referir a um elemento da lista
nomeLista[valorIndice]
# obs: o primeiro elemento tem índice 0
# obs2: o último elemento tem índice -1
# obs3: não é possível armazenar um valor em uma posição inexistente

# para se referir a um elemento dentro de uma lista filha
nomeListaMae[indiceMae][indiceFilha]

# para se referir aos elementos de uma lista de maneira sequencial
lista[indiceInicial:indiceFinal+1:passo]
# obs: lista[N:] (sequência do elemento de índice N até o último elemento)
# obs2: lista[:N] (sequência do primeiro elemento até o elemento de índice N)
# obs3: lista[:] (todos os elementos da lista)

# para criar uma cópia compartilhada da lista (cria um apelido para uma variável)
apelidoLista = nomeLista

# para passar uma lista para uma NOVA VARIÁVEL (cria uma nova variável)
novaLista = nomeLista[:]

# para concatenar listas
nomeLista1+nomeLista2

# para repetir N vezes uma lista
valorN*nomeLista

# para adicionar um elemento no final da lista (modificando o objeto)
nomeLista.append(valorAdicionado)

# para adicionar um elemento em determinada posição de uma lista (modificando o objeto)
nomeLista.insert(indice, valorAdicionado)
# obs: os elementos à direita deslocarão o índice em uma unidade

# para remover um item da lista
nomeLista.pop(indice)

# para adicionar um elemento no final da lista sem modificar o objeto
nomeLista+[valorAdicionado]

# para inverter a ordem dos elementos de uma lista
nomeLista.reverse()

# para ordenar os elementos de uma lista
nomeLista.sort(reverse=False) #ordem crescente
nomeLista.sort(reverse=True) #ordem decrescente
# obs: esta lista deve possuir um único tipo de dado



# para criar uma tupla (lista imutável)
nomeTupla = (valor1, valor2, valor3)
# obs: as tuplas são manipulada da mesma maneira que manipulamos listas
# obs2: os MÉTODOS de listas não podem ser aplicados em tuplas
# obs3: o uso do parênteses não é obrigatório



# obs: uma string é uma lista imutável de caracter
# obs2: as strings são manipulada da mesma maneira que manipulamos listas
# obs3: os MÉTODOS de listas não podem ser aplicados em strings

# para converter uma string para letras maiúsculas
nomeString.upper()

# para converter uma string para letras minúsculas
nomeString.lower()

# para transformar apenas o primeiro caracter em letra maiúscula
nomeString.capitalize()

# para retornar o índice (da primeira ocorrência) de determinados caracteres
nomeString.find("caracteres")

# para verificar se todos os caracteres são minúsculos
nomeString.islower()

# para verificar se todos os caracteres são maiúsculos
nomeString.isupper()

# para verificar se todos os caracteres são espaço
nomeString.isspace()

# para verificar se uma string termina com determinados caracteres
nomeString.endswith("caracteres")

# para substituir parte de uma string
nomeString.replace("parteExcluída","parteIncluída")

# para dividir a string sempre que encontrar determinado caracter
nomeString.split("caracter")
# obs: o caracter encontrado será excluído

# para unir uma lista de strings em uma única string com um caracter separador
"caracterSep".join(listaStrings)



# para criar um dicionário
nomeDic = {valorIndice1:valor1, valorIndice2:valor2, valorIndice3:valor3}
# obs: os índices podem ser qualquer tipo de dado (string, float, tuple, etc)
# obs2: não pode haver dois índices de mesmo valor
# obs3: os dicionários são manipulados da mesma maneira que manipulamos listas
# obs4: os MÉTODOS de listas não podem ser aplicados nos dicionários
# obs5: diferentemente das listas, é possível inserir um valor para uma chave inexistente

# para retornar uma sequência de tuplas com o par (chave, valor)
nomeDic.items()

# para retornar apenas os índices (chaves) de um dicionário
nomeDic.keys()

# para retornar apenas os elementos de um dicionário
nomeDic.values()

# para unir dois dicionários
nomeDic1.update(nomeDic2)
# obs: apenas o dicionário 1 será alterado



# para criar um range (sequência de elementos) (vInicial, vFinal+1, passo)
nomeRange = range(valorInicial, valorFinal+1, passo)
# obs: apenas valores int podem ser inseridos na função range, para float use numpy.arange
# obs2: para visualizar o range basta transformá-lo em uma lista



# para criar um set
nomeSet = set([valor1, valor2, valor3])
# obs: os elementos NÃO são ordenados
# obs2: os elementos NÃO possuem chaves nem índices para ser acessados
# obs3: NÃO é possível inserir dois elementos de mesmo valor (caso o valor já exista NENHUM erro é gerado)
# obs4: é possível inserir elementos de diferentes tipos
# obs5: set é uma estrutura de dados mutável

# para adicionar elemento em um set
nomeSet.add(valor)

# para adicionar vários elementos a um set
nomeSet.update([valor1, valor2, valor3])

# para remover elemento em um set
nomeSet.remove(valor)

# para remover um elemento SEM retornar ERRO caso este não exista
nomeSet.discard(valor)

# para criar um set IMUTÁVEL
nomeFSet = frozenset([valor1, valor2, valor3])

# para retornar a união de dois sets
# obs: retorna os elementos de nomeSet1 e de nomeSet2
nomeSet1.union(nomeSet2) #1ª maneira
nomeSet1|nomeSet2 #2ª maneira

# para retornar a interseção entre dois sets
# obs: retorna apenas os alementos que existem em nomeSet1 e em nomeSet2
nomeSet1.intersection(nomeSet2) #1ª maneira
nomeSet1&nomeSet2 #2ª maneira

# para retornar a diferença entre dois sets
# obs: retorna os elementos de nomeSet1 que não existem em nomeSet2
nomeSet1.difference(nomeSet2) #1ª maneira
nomeSet1-nomeSet2 #2ª maneira

# para retornar a diferença simétrica entre dois sets
# obs: retorna os elementos de CADA set que não existem no outro
nomeSet1.symmetric_difference(nomeSet2) #1ª maneira
nomeSet1^nomeSet2 #2ª maneira


# para verificar se existe determinado elemento em um conjunto
# obs: quando é utilizado em um dicionário, irá verificar a existência de uma CHAVE
valorElemento in nomeConjunto
# obs: retornará True ou False

# para retornar o tamanho de um conjunto
len(nomeConjunto)

# para somar todos os elementos de um conjunto
sum(nomeConjunto)

# para contar quantas vezes determinado valor aparece em um conjunto
nomeConjunto.count(valor)

# para retornar o elemento de maior valor em um conjunto
max(nomeConjunto)
# obs: este conjunto deve possuir um único tipo de dado

# para retornar o elemento de menor valor em um conjunto
min(nomeConjunto)
# obs: este conjunto deve possuir um único tipo de dado

# para retornar o índice da primeira ocorrência de determinado valor
nomeConjunto.index(valor)

# para retornar cada elemento do conjunto e seu respectivo índice
nomeEnumerate = enumerate(nomeConjunto)
# obs: transformar o objeto enumerate e uma lista para ser possível visualizá-lo



# para criar N tuplas com um elemento de cada conjunto
# obs: N é o número de elementos do menor conjunto
# obs2: transformar o objeto zip em uma lista para ser possível visualizá-lo
nomeZip = zip(conjunto1, conjunto2, conjunto3)

# executa uma função N vezes, passando cada elemento do conjunto como parâmetro
nomeMap = map(nomeFunc, nomeConjunto)
# obs: transformar o objeto map em uma lista para ser possível visualizá-lo
# obs2: nomeMap possui cada valor retornado pela função

# executa uma função N vezes, passando cada elemento do conjunto como parâmetro
nomeFilter = filter(nomeFunc, nomeConjunto)
# obs: transformar o objeto filter em uma lista para ser possível visualizá-lo
# obs2: nomeFilter possui apenas os elementos que fizeram a função retornar True

# para importar a função reduce
from functools import reduce

# executa uma função N vezes, passando o retorno anterior e um elemento como parâmetros
reduce(nomeFunc, nomeConjunto)
# obs: par1=retorno da execução anterior de nomeFunc; par2=próximo elemento do conjunto
# obs2: na primeira execução (par1=nomeConjunto[0]; par2=nomeConjunto[1])
# obs3: reduce retorna apenas o resultado da última execução da função









## BLOCOS CONDICIONAIS E FUNÇÕES ##

# obs: a identação que será responsável por indicar determinada instrução
#está dentro de um bloco

# if executa o bloco de instrução SE valorBool for True
# obs: se valorBool for False, else é executado
# obs2: função elif é o mesmo que colocarmos outro if dentro do else
if (valorBool1):
	#instrução
elif (valorBool2):
	#instrução
else:
	#instrução

# while executa o bloco de instrução ENQUANTO valorBool for True,
# obs: se valorBool for False, else é executado e o laço é encerrado
while (valorBool):
	#instrução
else:
    #instrução

# for executará o bloco de instrução com nomeVar possuindo o valor de cada elemento do conjunto
# obs: o conjunto pode ser uma lista, string, range, tupla, dicionário, etc
for nomeVar in nomeConjunto:
	#instrução

# obs: o comando break encerra imediatamente o laço de repetição superior mais próximo
# obs2: o comando continue encerra imediatamente a etapa atual do laço de repetição superior mais próximo
# obs3: os comandos break ou continue não estão relacionados com os laços if, apenas laços de REPETIÇÃO

# podemos utilizar o comando if em uma única linha
nomeVar=valor1 if valorBool else valor2
# obs: se valorBool for True comandoTrue será executado
# obs2: se valorBool for False comandoFalse será executado

# podemos utilizar o comando for em uma única linha
nomeVar=[comandoFor for varFor in nomeConjunto]
# obs: comando será executado com varFor possuindo o valor de cada elemento do conjunto
# obs2: mais RÁPIDO que MAP, e mais rápido que for normal para operações SIMPLES

# podemos utilizar os dois comandos acima em uma única linha
nomeVar=[(valor1 if valorBool else valor2) for varFor in nomeConjunto]


# try TENTA executar uma instrução, caso ocorra algum erro, except é executado
# obs: apenas um bloco except é executado
# NameError: variável não definida
# TypeError: operação entre tipos de dados diferentes
try:
    #instrução (executada caso não ocorra nenhum erro)
except erroTipo1:
	#instrução (executada caso ocorra erroTipo1 em try)
except erroTipo2:
	#instrução (executada caso ocorra erroTipo2 em try)
except:
    #instrução (executada caso ocorra OUTRO tipo de erro em try)
else:
    #instrução (executada se try for executada)
finally:
    #instrução (executada independente de qualquer coisa)

# para levantar determinado tipo de erro para determinada condição
if condErro:
    raise tipoError("Mensagem de Erro")
# obs: quando raise é executado, o código para instantaneamente

# para resultar em erro quando determinada condição NÃO é satisfeita
assert condicao, "Mensagem de Erro"
# obs: quando a condição não é satisfeita, o código para instantaneamente

# para abrir arquivos e fechá-los autmaticamente caso ocorra algum erro
# obs: neste caso não é necessário fazer nomeOpen.close()
with open("endereço") as nomeOpen:
    # código


# para criar uma função
# obs: é possível utilizar dentro da função, uma variável criada fora dela
# obs2: não é possível alterar, dentro da função, uma variável criada fora dela
# obs3: qualquer variável criada dentro da função, existirá apenas dentro dela 
# obs4: podemos atribuir um valor padrão para os últimos parâmetros da função
def nomeFunc(par1, par2, par3=valorPadrao1, par4=valorPadrao2):
	#instrução
    return valorRetornado

# para criar uma variável global (que existirá fora da função)
# obs: quando a variável é criada fora da função, SEM USAR global, ela pode ser ACESSADA dentro da função,
#mas qualquer alteração dessa variável dentro da função, NÃO permanecerá fora da função
# obs2: quando a variável é criada dentro da função usando global, esta variável poderá ser ACESSADA e ALTERADA fora da função
def nomeFunc(par1, par2, par3):
	global nomeVar
	nomeVar=valor
	return valorRetornado	

# args (*) torna parTupla uma tupla que armazenará todos os parâmetros passados
# obs: dentro da função nos referimos a parTupla sem o asterisco
def nomeFunc(*parTupla):
	#instrução
	return valorRetornado

# para executar uma função
nomeFunc(par1, par2, par3)

# kwargs (**) torna parDic um dicionário que armazenará todos os parâmetros passados
# obs: dentro da função nos referimos a parDic sem os asteriscos
def nomeFunc(**parDic):
	#instrução
	return valorRetornado

# para executar uma função de parâmetro kwargs
nomeFunc(nomeVar1=par1, nomeVar2=par2, nomeVar3=par3)
# obs: o nome das variáveis serão os índices do dicionário
# obs2: os parâmetros passados serão os elementos do dicionário

# podemos utilizar args e kwargs em uma mesma função
# obs: args deve ser colocado antes de kwargs
def nomeFunc(*parTupla, **parDic):
	#instrução
	return valorRetornado

# para executar uma função de parâmetros mistos (args e kwargs)
# obs: os parâmetros args devem ser passados antes dos parâmetros kwargs
nomeFunc(par1, par2, nomeVar1=par3, nomeVar2=par4)

# para criar uma função em tempo de execução
lambda par: comando
# obs: utilizar em map, filter ou reduce

# o código abaixo é inserido no mesmo arquivo em que determinada função foi implementada,
#ele executa determinada função deste arquivo, caso este arquivo seja executado,
#mas essa função não é executada quando este arquivo é importado em um script
if __name__ == "__main__":
	nomeFunc(par1, par2, par3)

# para modificar funções, criamos decorators (funções modificadores de funções)
def funcDecorator(func):
    def funcInterna(*args, **kwargs):
        varFunc = func(*args, **kwargs) # valor a ser retornado pela função que será modificada
        # fazer as operações necessárias com varFunc
        return valorRetornado
    return funcInterna

# para modificar uma função utilizando um decorator
@funcDecorator
def nomeFunc(par1, par2, par3):
    #instrução
    return valorRetornado









## ARQUIVOS E DIRETÓRIOS ##

# módulo para gerenciamento de diretórios
import os

# módulo para gerenciamento de diretórios
import shutil

# para se referir ao diretório atual
'./'

# para se referir ao diretório anterior
'../'

# para se referir a 2 diretórios atrás (diretório anterior do diretório anterior)
'../../'

# para se referir a um diretório seguinte
'./nomeDir'

# para listar os arquivos contidos em um diretório
os.listdir(path='endereçoDir')

# para criar um diretório
os.mkdir("endereçoDir")

# para remover um arquivo
os.remove("endereçoArquivo")

# para renomear um arquivo
os.rename("endereçoDir/nomeAntigo", "endereçoDir/nomeNovo")

# para remover uma árvore de diretórios 
shutil.rmtree("endereçoDir")

# para copiar um arquivo
shutil.copyfile("endereçoOrigem", "endereçoDestino")

# para copiar uma árvore de diretórios
shutil.copytree("endereçoOrigem", "endereçoDestino")



# módulo para gerenciar variáveis utilizadas pelo interpretador
import sys

# para incluir um diretório nos caminhos de busca do interpretador
sys.path.insert(0, 'endereçoDir')
# obs: 0 representa a posição em que o endereço será inserido na lista de diretórios
# obs2: útil para o interpretador buscar algum módulo neste diretório

# para listar os diretórios de busca do interpretador
sys.path



# módulo: arquivo .py que possui um conjunto de funções
# pacote: diretório que possui um conjunto de módulos .py

# para se referir ao endereço de um módulo: nomePasta.nomeArquivo
# obs: a primeira pasta citada deve ter sido incluída nos caminhos de busca do interpretador
# obs2: os caracteres ".py" não devem ser colocados após nomeArquivo

# para importar módulos
import endereçoArquivo1 as apelidoModulo1, endereçoArquivo2 as apelidoModulo2

# para importar apenas uma função de determinado módulo
from endereçoArquivo import nomeFunc

# para visualizar os pacotes, módulos e funções importadas
dir()

# para verificar as funções disponíveis em determinado módulo
dir(apelidoModulo)

# para executar uma função
apelidoModulo.nomeFunc(par1, par2, par3)
# obs: se a função tiver sido diretamente importada, não é necessário mencionar o módulo



# para abrir um arquivo para leitura
nomeOpen = open("endereço", encoding="utf-8")

# para abrir um arquivo para escrita, limpando todo o arquivo (w)
nomeOpen = open("endereço", "w")

# para abrir um arquivo para escrita, sem apagar o que há nele (a)
nomeOpen = open("endereço", "a")

# para ler uma quantidade valorN de caracteres de um arquivo aberto para leitura
nomeOpen.read(valorN)
# obs: se nenhum parâmetro for informado todo arquivo será lido

# para ler o arquivo aberto e retornando as linhas em uma lista
nomeOpen.readlines() 

# para escrever em um arquivo aberto
nomeOpen.write("Texto escrito no arquivo")

# para retornar a posição do cursor
nomeOpen.tell()

# para definir uma nova posição para o cursor (valorPos)
nomeOpen.seek(valorPos)
# obs: a posição inicial é igual a 0

# para fechar e assim salvar o arquivo modificado
nomeOpen.close()


# módulo paramanipular arquivos CSV
import csv

# para ler um arquivo CSV aberto com a função open
csv.reader(nomeOpen, delimiter='caracter')
# obs: caracter se refere ao caracter delimitador de colunas (linhas são separadas automaticamente)
# obs2: transformar o objeto csv em uma lista para ser possível visualizá-lo


# módulo para abrir um arquivo na internet
import urllib.request

# para abrir um arquivo da internet
nomeURLOpen = urllib.request.urlopen("endereçoURL")


# módulo para manipular arquivos JSON
import json

# para transformar um dicionário em uma string de formato JSON
json.dumps(nomeDic)

# para transformar uma string de formato JSON em um dicionário
json.loads(nomeStringJSON)









## CLASSES E OBJETOS ##

# para criar uma classe
# obs: nomeClasse irá herdar os atributos e métodos de nomeClasseMae
# obs2: caso não for utilizada uma classe mãe, NÃO é necessário utilizar o parênteses.
class nomeClasse(nomeClasseMae):
    # para criar variáveis de classe
    # obs: se a variável for alterada em qualquer instância, todas as instâncias terão o valor alterado
	# obs2: se a variável de classe criada tiver mesmo nome que a herdada, a herdada é substituída
	varClasse1 = valor1
	varClasse2 = valor2
         
    # para criar um método
    # obs: o primeiro parâmetro (parSelf) se refere ao próprio objeto (self)
    # obs2: para se referir a um atributo dentro de uma função, devemos citar o objeto (parSelf.atributo)
    # obs3: se o método criado tiver mesmo nome que um método herdado, o herdado é substituído
    def nomeMetodo (parSelf, par2, par3):
    	#instrução
    	return valorRetornado


    # método construtuor (__init__) será executado sempre que um objeto desta classe for criado
    # obs: o primeiro parâmetro (par1) se refere ao próprio objeto (self)
    # obs2: os demais parâmetros se referem aos valores passados na criação do objeto
    def __init__(parSelf, par2, par3):
        # para criar variáveis de instância
        # obs: se o valor dessa variável for alterado em uma instância, não será alterado nas demais
        parSelf.varInstancia1 = valor3
        parSelf.varInstancia2 = valor4
        
        # para executar o método construtor da classe mãe
        super(nomeClasse, parSelf).__init__(valor1, valor2)
        # obs: valor1 e valor2 são valores passados como parâmetros para o construtor da classe mãe



# para criar um objeto através de determinada classe
nomeObjeto = nomeClasse(par2, par3)

# para se referir a um atributo de determinado objeto
nomeObjeto.nomeAtributo 

# para executar um método de determinado objeto
nomeObjeto.nomeMetodo(par2, par3)

# para retornar um atributo de determinado objeto
getattr(nomeObjeto, "nomeAtributo")

# para verificar se um objeto possui determinado atributo
hasattr(nomeObjeto, "nomeAtributo")

# para alterar um atributo de determinado objeto
setattr(nomeObjeto, "nomeAtributo", novoValor)

# para remover um atributo de determinado objeto
delattr(nomeObjeto, "nomeAtributo")

# para listar os atributos e métodos de determinado objeto
dir(nomeObjeto)



  





## MYSQL - BANCO DE DADOS SQL ##

# pip install mysql-connector-python
# módulo para gerenciamento de banco de dados MySQL
import mysql.connector

# para criar uma conexão com um banco de dados MySQL
nomeConex = mysql.connector.connect(user='usuarioBanco', password='senhaBanco', host='endereçoServidor', port='3306', database='nomeBanco')

# para criar um cursor para manipular os dados em um banco MySQL
nomeCursor = nomeConex.cursor()

# para executar um comando no banco MySQL
nomeCursor.execute("comando executado")

# para se referir a última coisa que foi retornada pelo banco MySQL
nomeCursor.fetchall()

# para salvar as alterações feitas no banco MySQL
nomeConex.commit()

# para encerrar o cursor criado no banco MySQL
nomeCursor.close()

# para encerrar a conexão criada no banco MySQL
nomeConex.close()









## POSTGRESQL - BANCO DE DADOS SQL ##

# pip install psycopg2
# módulo para gerenciamento de banco de dados PostgreSQL
import psycopg2

# para criar uma conexão com um banco de dados PostgreSQL
nomeConex = psycopg2.connect(user = 'usuarioBanco', password = 'senhaBanco', host = 'endereçoServidor', port = '5432', database = 'nomeBanco')

# para criar um cursor para manipular os dados em um banco PostgreSQL
nomeCursor = nomeConex.cursor()

# para executar um comando no banco PostgreSQL
nomeCursor.execute("comando executado")

# para se referir a última coisa que foi retornada pelo banco PostgreSQL
nomeCursor.fetchall()

# para salvar as alterações feitas no banco PostgreSQL
nomeConex.commit()

# para encerrar o cursor criado no banco PostgreSQL
nomeCursor.close()

# para encerrar a conexão criada no banco PostgreSQL
nomeConex.close()









## MONGO DB - BANCO DE DADOS NOSQL ##

# pip install pymongo
# funções para manipular banco de dados MongoDB
from pymongo import MongoClient

# para iniciar uma conexão com o MongoDB
objMongo = MongoClient("localhost",27017)

# para visualizar os banco de dados existentes
objMongo.database_names()

# para criar um objeto referente a um banco de dados
objDB = objMongo.nomebanco

# para verificar o nome do banco de dados de um objDB
objDB.name

# para visualizar as coleções disponíveis
objDB.collection_names()

# para criar um objeto referente a uma collection (tabela) em um banco de dados
objCollection = objDB.nomeCollection

# para retornar um único documento contido em uma coleção
objCollection.find_one()

# para visualizar os documentos armazenados em uma coleção
list(objCollection.find())

# para contar o número de documentos contidos em uma coleção
objCollection.count()

# para inserir um único documento em uma collection
objDocumento = objColecao.insert_one(nomeDic)
# obs: nomeDic é um dicionário no python

# para inserir vários documentos em uma collection
objDocumento = objColecao.insert_one([nomeDic1, nomeDic2])
# obs: nomeDic1 e nomeDic2 são dicionários no python

# para visualizar a id do documento inserido na coleção
objDocumento.inserted_id









## REQUISIÇÕES WEB ##

# pip install requests
# módulo para requisições web
import requests

# para fazer uma requisição
nomeReq = requests.tipoMetodo("endereçoURL", headers=dicHeaders, cookies=dicCookies, data=dicData)
# obs: criar um dicionário para enviar as informações do headers
# tipoMetodos: get, post, put, patch, delete

# exemplo de um dicionário para form-data
dicData = {'usuario': 'ramon', 'senha': 'water123', 'Entrar': 'Entrar'}

# exemplo de um dicionário para cookie
dicCookies = {'cookie': list(nomeReq.cookies)[0].name+'='+list(nomeReq.cookies)[0].value}

# exemplo de requisição post
reqLogin = requests.post("https://app.tagplus.com.br/ufv/home/login", data=dicData)

# exemplo de requisição get
reqGet = requests.get('https://app.tagplus.com.br/ufv/', cookies=dicCookies)









## TERMINAL ##

# iniciar o script python com
# coding: utf-8

# módulo para interagir com o interpretador
import sys

# para rodar um script python pelo terminal
# python nomeArquivo.py

# para rodar um script python pelo terminal passando parâmetros
# python nomeArquivo.py par1 par2 par3

# para se referir aos parâmetros passados no terminal
sys.argv[1] # par1
sys.argv[2] # par2
sys.argv[3] # par3
# obs: os parâmetros são recebidos como uma string









## REGEX - EXPRESSÕES REGULARES DE STRINGS ##

# módulo para encontrar expressões regulares
import re

# para encontrar a PRIMEIRA ocorrência de uma expressão regular em uma string
re.search(r'expressão', nomeString, par)

# para encontrar TODAS as ocorrências de uma expressão regular em uma string
re.findall(r'expressão', nomeString, par)

# expressão: expressão a ser buscada 
#  		     \w=qualquer caracter alfa-numérico
#   		 \d=um número qualquer				
#            \s=espaço em branco
#		     \t=tabulação (espaço com tab)
#            \n=quebra de linha
#            .=qualquer caracter exceto quebra de linha
#            \W=qualquer caracter não alfa-numérico
#            \D=qualquer caracter exceto dígitos
#            \S=qualquer caracter exceto espaço em branco
#            *=repete a condiação anterior de 0 a infinitas vezes (ex: \w*)
#            \caracterEspecial=é necessário usar barra antes de algum caracter reservado
#            |=expressão OR
#            [char1char2char3]=char1 OU char2 OU char3 
#  			 [(string1)(string2)(string3)]=string1 OU string2 OU string3
#  			 [charInicial-charFinal]=charInicial até charFinal
#  			 [^char1char2char3]=qualquer caracter menos char1, char2 ou char3
# 			 ^char=caso char estiver no início da expressão como um todo
# 			 char$=caso char estiver no final da expressão como um todo
# 			 expreção{nMin,nMax}=no mínimo nMin e no máximo nMax trechos com a expressão desejada 
# 			                    SEGUIDAS (sem espaçamento ou outro caracter entre as repetições)
#
# par: re.I=ignora o case dos caracters (maiúsculo/minúsculo) 
#      re.M=divide a expressão como um todo em várias expressões para cada quebra de linha









## METATRADER5 - INTEGRAÇÃO COM PLATAFORMA TRADER ##
# Python 3.8.1/ MetaTrader 5.00 build 2361

# pip install MetaTrader5==5.0.19
# módulo para integração ao MetaTrader5
import MetaTrader5

# para iniciar a conexão com o MetaTrader (abrir o MetaTrader antes)
MetaTrader5.initialize()

# para que o MetaTrader5 estabeleça uma conexão com o servidor
MetaTrader5.wait()

# para visualizar o status da conexão
# obs: função não disponível para algumas versões
MetaTrader5.terminal_info()

# para visualizar as informações da conta conectada
MetaTrader5.account_info()

# para pegar os dados de um ativo (tick atual)
MetaTrader5.symbol_info_tick("nomeAtivo")

# para pegar os dados de um ativo (range de ticks)
MetaTrader5.copy_ticks_range("nomeAtivo", msInicial, msFinal, True)

# para enviar uma ordem de compra
# obs: o Auto Trading deve estar ativo
MetaTrader5.Buy("nomeAtivo", nVolume)

# para enviar uma ordem de venda
# obs: o Auto Trading deve estar ativo
MetaTrader5.Sell("nomeAtivo", nVolume)

# para fechar todas as posições de determinado ativo
MetaTrader5.Close("nomeAtivo")

# para enviar uma ordem de compra
# obs: o Auto Trading deve estar ativo
# obs2: WIN (valorMargem=múltiplo de 5); WDO (valorMargem=múltiplo de 0.5)
MetaTrader5.order_send({"action": 1,
"type": 0, #0 indica compra
"symbol": "nomeAtivo",
"volume": nVolume, #quantidade do ativo
"price": m.symbol_info_tick("nomeAtivo").ask, #preço para comprar
"sl": m.symbol_info_tick("nomeAtivo").bid-valorMargem, #valor mínimo (stop loss)
"tp": m.symbol_info_tick(nomeAtivo).bid+valorMargem, #valor máximo (stop gain)
"type_filling": valorTipo})  # 0=Fill or Kill; 1=Immediate or Cancel; 2=Return

# para enviar uma ordem de venda
# obs: o Auto Trading deve estar ativo
# obs2: WIN (valorMargem=múltiplo de 5); WDO (valorMargem=múltiplo de 0.5)
MetaTrader5.order_send({"action": 1,
"type": 1, #1 indica venda
"symbol": "nomeAtivo",
"volume": nVolume, #quantidade do ativo
"price": m.symbol_info_tick("nomeAtivo").bid, #preço para vender
"sl": m.symbol_info_tick(nomeAtivo).ask+valorMargem, #valor máximo (stop loss)
"tp": m.symbol_info_tick(nomeAtivo).ask-valorMargem, #valor mínimo (stop gain)
"type_filling": valorTipo}) # 0=Fill or Kill; 1=Immediate or Cancel; 2=Return

# para encerrar uma posição de compra
MetaTrader5.order_send({"action": 1,
"type": 1,
"symbol": "nomeAtivo",
"volume": nVolume,
"price": m.symbol_info_tick("nomeAtivo").bid,
"type_filling": valorTipo,
"position": ticketOrdem})

# para encerrar uma posição de venda
MetaTrader5.order_send({"action": 1,
"type": 0,
"symbol": "nomeAtivo",
"volume": nVolume,
"price": m.symbol_info_tick("nomeAtivo").ask,
"type_filling": valorTipo,
"position": ticketOrdem})

# para verificar sua posição
MetaTrader5.positions_get()
# type: 0=comprado; 1=vendido;









## PERFORMANCE ##

# módulo para gerenciar quaisquer opções do sistema
import sys

# para exibir o tamanho de um objeto em BYTES
sys.getsizeof(nomeObjeto)









## STATISTICS - BIBLIOTECA PARA ESTATÍSTICA ##

# módulo para cálculos estatísticos
import statistics
# obs: NÃO utilizar este módulo para arrays

# para calcular a média de um conjunto
statistics.mean(nomeConjunto)

# para calcular a mediana de um conjunto
statistics.median(nomeConjunto)

# para calcular a moda de um conjunto
statistics.mode(nomeConjunto)

# para calcular a variância de um conjunto
statistics.stdev(nomeConjunto)

# para calcular o desvio padrão de um conjunto
statistics.variance(nomeConjunto)









## TKINTER - INTERFACE GRÁFICA ## 

# pip install tk
# pip install functools
# módulo para criação de interface gráfica
import tkinter
import functools


# para criar uma janela
objJanela = tkinter.Tk()

# para que o processo se mantenha em execução (a janela não feche)
objJanela.mainloop()

# para alterar o título da janela
objJanela.title("Título Janela")

# para alterar a cor de fundo da janela
# obs: o código hexadecimal possui 2 dígitos representando cada cor (vermelho, verde e azul, respectivamente)
objJanela["bg"] = "#valorHexadecimal"

# para alterar o tamanho e margem inicial da janela
# obs: LARGURA, ALTURA, ESQUERDA e TOPO são valores numéricos (em pixels)
objJanela.geometry("LARGURAxALTURA+ESQUERDA+TOPO")

# para criar um objeto label
objLabel = tkinter.Label(objJanela, text="Texto do Label")

# para exibir o objeto label usaremos um gerenciador de layout
objLabel.place(x=ESQUERDA, y=TOPO)

# para alterar o texto de um objeto label
objLabel["text"] = "Novo Texto"

# para criar um botão
# obs: nomeFunc é a função que será executada ao clicar no botão
objBotao = tkinter.Button(objJanela, width="LARGURA", text="Texto do Botão", command=nomeFunc)

# para passar parâmetros para a função do botão
objBotao["command"] = functools.partial(nomeFunc, par1, par2, par3)

# para exibir o objeto botão usaremos um gerenciador de layout
objBotao.place(x=ESQUERDA, y=TOPO)

# para criar uma caixa para entrada de texto
objCaixa = tkinter.Entry(objJanela)

# para exibir o objeto caixa usaremos um gerenciador de layout
objCaixa.place(x=ESQUERDA, y=TOPO)

# para se referir ao texto digitado no objeto caixa
objCaixa.get()









## PIL - IMAGEM PARA PDF ##

# pip install Pillow
# módulo para carregar imagens
from PIL import Image

# para carregar uma imagem
nomeImg = Image.open("endereço")

# para mostrar a imagem carregada
nomeImg.show()

# para alterar o tipo do objeto (para ser possível salvar em PDF)
novaImg = nomeImg.convert("RGB")

# para alterar o tamanho da imagem
novaImg = nomeImg.resize((valorLargura, valorAltura), resample=0)

# para salvar as imagens como pdf
nomeImg.save("endereço.pdf", save_all = True, append_images = [nomeImg2,nomeImg3])
# obs: as imagens nomeImg2 e nomeImg3, respectivamente, serão salvas no pdf









## PYMUPDF - PDF PARA IMAGEM ##

# pip install pymupdf
import fitz

objOpen = fitz.open("endereço.pdf")
matriz = fitz.Matrix(4,4) #quanto maior o tamanho da matriz melhor a qualidade
page = objOpen.load_page(nPagina) #nPagina é o número da página que deseja carregar (0 = 1ª página)
page.get_pixmap(matrix=matriz).save("endereço.png")









## SELENIUM - MACRO NO NAVEGADOR ##

# baixar o ChromeDriver para Google Chrome e GeckoDriver para Mozilla Firefox
# ChromeDriver: https://sites.google.com/a/chromium.org/chromedriver/downloads
# GeckoDriver: https://github.com/mozilla/geckodriver/releases

# pip install selenium
# módulo para automação de testes
from selenium import webdriver

# para executar o Chrome a partir do selenium
nomeDriver = webdriver.Chrome(executable_path="endereçoDir/chromedriver.exe")

# para executar o Firefox a partir do selenium
nomeDriver = webdriver.Firefox(executable_path="endereçoDir/geckodriver.exe")

# para acessar um enderço URL a partir do driver criado acima
nomeDriver.get("endereçoURL")

# para encontrar uma tag HTML na página
nomeTag = nomeDriver.find_element_by_xpath("//tipoTag[@parTag='valorPar']")
# obs: tipoTag (a, div, span, input, etc)
# obs2: parTag (href, name, id, etc)

# para encontrar várias tags de determinado tipo
nomeTag = nomeDriver.find_element_by_tag_name('tipoTag')

# para navegar dentro de um frame
nomeDriver.switch_to.frame(nomeTagFrame)
# obs: nomeTagFrame=nomeDriver.find_element_by_xpath("//frame[@parTag='valorPar']")

# para sair do frame
nomeDriver.switch_to.default_content()

# para executar um script JS
nomeScript = nomeDriver.execute_script("return scriptJS")

# para clicar em um botão ou em algum link
nomeTag.click()

# para abrir um link em uma nova aba
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
ActionChains(nomeDriver) \
.key_down(Keys.CONTROL) \
.click(nomeTagLink) \
.key_up(Keys.CONTROL) \
.perform()

# para abrir um link em uma nova aba
nomeDriver.execute_script("window.open('endereçoURL', 'new_window')")

# para limpar um campo de input (caixa para inserção de texto)
nomeTag.clear()

# para preencher um campo de input (caixa para inserção de texto) com uma string
nomeTag.send_keys(valorString)

# para trocar de janelas
nomeJanelas = nomeDriver.window_handles
nomeDriver.switch_to.window(nomeJanelas[valorIndice])









## PYSERIAL - TRANSMISSÃO SERIAL PARA EMBARCADOS ##

# pip install pyserial
# módulo para transmitir dados do computador por comunicação serial para um microcontrolador
import serial

# função para listar as portas conectadas
import serial.tools.list_ports

# para armazenar a porta conectada em uma variável
nomePorta = str(serial.tools.list_ports.comports()[idPorta][0])
# obs: se estiver apenas uma porta conectada idPorta será 0

# para criar uma comunicação serial com determinada porta
objSerial = serial.Serial(nomePorta, valorBPS)
# valorBPS: mesmo velocidade de transmissão definido em Serial.begin() no código do arduino

# para enviar uma string por comunicação serial
objSerial.write(varString.encode())
# obs: lembrar sempre de colocar \n no final da string

# para ler uma string por comunicação serial
objSerial.readline().decode().split('\r')[0]

# para encerrar a comunicação serial
objSerial.close()









## MULT THREADING ##

# módulo para executar mais de uma thread
import threading

# para criar uma função em uma thread
nomeThread = threading.Thread(target=nomeFunc, args=(par1, par2,))
# obs: cada thread só pode ser executada uma única vez
# obs2: par1 e par2 são os parâmetros a serem passados para a função
# obs3: deve ser passado uma tupla como parâmetro, logo, se houver APENAS UM argumento, deixar uma vírgula no final,
#para que o interpretador entenda que é um tupla e não o tipo de dado de par

# para executar a thread criada
nomeThread.start()

# para verificar se a thread está ativa
nomeThread.is_alive()

# para criar um objeto lock
nomeLock = threading.Lock()

# para que duas ou mais threads não executem ao mesmo tempo determinado trecho de código
nomeLock.acquire() #bloqueia
# código
nomeLock.release() #desbloqueia









## HBMQTT ##

# python 3.6
# pip install websockets==4.0.1
# pip install hbmqtt==0.9.2
