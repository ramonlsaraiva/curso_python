## MULTILAYER PERCEPTRON ##

# pip install scikit-learn
# função para normalizar os dados
from sklearn.preprocessing import StandardScaler

# pip install keras==2.2
# pip install tensorflow==2.2
# função para criar a estruturar de uma rede neural
from tensorflow.keras.models import Sequential

# função para criar dense layers (camadas totalmente conectadas)
from tensorflow.keras.layers import Dense

# função para carregar um modelo salvo
from tensorflow.keras.models import load_model

# função para salvar uma escala
from joblib import dump

# função para carregar uma escala
from joblib import load



# para normalizar os dados (ou seja, os dados irão variar de 0 a 1)
# obs: note que os dados de teste NÃO devem ser utilizados para a normalização dos dados na função fit, pois,
#devemos fazer todo o processo como se não tivéssemos os dados de teste, de forma que eles não influenciem no modelo
# obs2: a normalização ocorre apenas nos dados de entrada

# escala para as entradas
objScale = StandardScaler().fit(xTrain)
xTrainMod = objScale.transform(xTrain)
xTestMod = objScale.transform(xTest)

# salvando as escalas
dump(xObjScaler, './endereçoDir/xObjScaler.bin', compress=True)



# para criar uma rede neural
nomeModelo = Sequential()

# para adicionar uma primeira camada de neurônios à rede neural criada
nomeModelo.add(Dense(units=nNeuronios, input_dim=nEntradas, activation=tipoFA))
# tipoFA: "relu" (tipo da função de ativação)

# para adicionar uma camadas intermediárias à rede neural criada
nomeModelo.add(Dense(units=nNeuronios, activation=tipoFA))
# tipoFA: "relu" (tipo da função de ativação)

# para adicionar uma camada de neurônios no final da rede neural criada
nomeModelo.add(Dense(units=nNeuronios))

# para compilar o modelo
nomeModelo.compile(optimizer=nomeOtimizador, loss=lossFunc, metrics=lossMetrics)
# nomeOtimizador: "adam" (otimizador)
# lossFunc: "mean_squared_error" (função de custo)
# lossMetrics: ["mae"] (mean absolute error)

# para visualizar as características do modelo
nomeModelo.summary()

# para treinar o modelo
objTreino = nomeModelo.fit(xTrainNorm, yTrain, batch_size=tamanhoLote, validation_split=dadosVal, epochs=nEpocas)
# obs: caso essa função seja executada NOVAMENTE, o modelo não será treinado do zero, mas sim de onde parou no último treino
# obs2: observar o erro em relação aos dados de treino (mae) e aos dados de validação (val_mae)
# tamanhoLote: tamanho do lote de dados que será passado para calcular o gradiente (lotes pequenos tem a capacidade de
#encontrar bons modelos com poucas épocas, no entanto, lotes grandes tem mais facilidade de encontrar o mínimo global,
#mas irá demorar mais tempo para treinar o modelo)
# dadosVal: 0.1 (proporção de dados que será separado para validação)
# nEpocas: 250

# para salvar o modelo
nomeModelo.save('endereçoDir')

# para acessar o log das métricas
objTreino.history


# para carregar um modelo
nomeModelo = load_model('endereçoDir')

# para carregar as escalas
objScale = load('./endereçoDir/xObjScaler.bin')

# para utilizar o modelo
saida = nomeModelo.predict(objScale.transform(entrada))
# obs: a entrada deve ser um array com shape (1, nEntradas)









## TREINANDO ARQUITETURAS DE CNN PRÉ-TREINADAS ##

# obs: instalar o KERAS ANTES do tensorflow
# instalar CUDA Toolkit 10.1
# baixar cuDNN 7.6.5 para CUDA 10.1
# pip install keras==2.2
# pip install tensorflow==2.2

# funções para utilizar a arquitetura de CNN ResNet50
from tensorflow.keras.applications.resnet import ResNet50
from keras.applications.resnet50 import preprocess_input

# funções para utilizar a arquitetura de CNN ResNet101
from tensorflow.keras.applications.resnet import ResNet101
from keras.applications.resnet import preprocess_input

# funções para utilizar a arquitetura de CNN ResNet152
from tensorflow.keras.applications.resnet import ResNet152
from keras.applications.resnet50 import preprocess_input

# funções para utilizar a arquitetura de CNN VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# funções para utilizar a arquitetura de CNN Inception V3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# funções para utilizar a arquitetura de CNN Xception
from tensorflow.keras.applications.xception import Xception
from keras.applications.xception import preprocess_input



# função para criar a estruturar de uma rede neural
from tensorflow.keras.models import Sequential

# função para criar dense layers (camadas totalmente conectadas)
from tensorflow.keras.layers import Dense

# módulo para importar as métricas do Keras
from tensorflow.keras import metrics

# função para criar os lotes de imagens
from keras.preprocessing.image import ImageDataGenerator

# função para inserir um otimizador no modelo
from tensorflow.keras import optimizers

# função para salvar o modelo durante o treinamento
from tensorflow.keras.callbacks import ModelCheckpoint

# classe mãe para criar uma classe de callback
from keras.callbacks import Callback

# utilizar o pyplot para plotar as métricas de treinamento e validação
from matplotlib import pyplot



# criando um objeto que representa uma rede neural
nomeModelo = Sequential()


# obs: apenas uma das arquiteturas abaixo deverá ser inserida no modelo e, em seguida, deverá ser inserido uma camada densa
# obs2: inserindo imagenet em weights, irá fazer o download automaticamento dos pesos do ResNet50
# obs3: como o número de classificações será diferente do número de classificações padrão da ResNet50, include_top receberá False
#e as dense layers não serão incluídas no modelo
# tipoPool: "max" (Max Pooling), "avg" (Average Pooling)

# para adicionar a arquitetura ResNet50 (sem dense layers) ao modelo criado
nomeModelo.add(ResNet50(include_top=False, pooling=tipoPool, weights='imagenet'))

# para adicionar a arquitetura ResNet101 (sem dense layers) ao modelo criado
nomeModelo.add(ResNet101(include_top=False, pooling=tipoPool, weights='imagenet'))

# para adicionar a arquitetura ResNet152 (sem dense layers) ao modelo criado
nomeModelo.add(ResNet152(include_top=False, pooling=tipoPool, weights='imagenet'))

# para adicionar a arquitetura VGG16 (sem dense layers) ao modelo criado
nomeModelo.add(VGG16(include_top=False, pooling=tipoPool, weights='imagenet'))

# para adicionar a arquitetura InceptionV3 (sem dense layers) ao modelo criado
nomeModelo.add(InceptionV3(include_top=False, pooling=tipoPool, weights='imagenet'))

# para adicionar a arquitetura Xception (sem dense layers) ao modelo criado
nomeModelo.add(Xception(include_top=False, pooling=tipoPool, weights='imagenet'))



# para adicionar as camadas totalmente conectadas ao modelo criado
nomeModelo.add(Dense(nClasses, activation=tipoFA))
# nClasses: número de classes de saída que o modelo irá classificar
# tipoFA: "softmax" (tipo da função de ativação de saída)

# para congelar as camadas de convolução
nomeModelo.layers[0].trainable = False

# para que as camadas de convolução não sejam congeladas
nomeModelo.layers[0].trainable = True

# para criar um otimizador do tipo SGD (Stochastic Gradient Descent)
objOtimizador = optimizers.SGD(learning_rate=valorTA, weight_decay=valorWD, momentum=valorMomentum, nesterov=True)
# valorTA: valor da taxa de aprendizado (sugestão: 0.01)
# valorWD: valor do decaimento dos pesos (sugestão: 0.0005)
# valorMomentum: valor do momentum (sugestão: 0.9)

# para criar um otimizador do tipo Adagrad
objOtimizador = optimizers.Adagrad(learning_rate=valorTA, initial_accumulator_value=valorAcumulador, epsilon=valorEp)
# valorTA: valor da taxa de aprendizado (sugestão: 0.01)
# valorAcumulador: valor do acumulador inicial (sugestão: 0.1)
# valorEp: valor do parâmetro epilson (sugestão: 1e-07)

# para compilar o modelo
nomeModelo.compile(optimizer=objOtimizador, loss="categorical_crossentropy", metrics=["accuracy", metrics.Precision(), metrics.Recall()])

# para criar um objeto para gerar as imagens de treino para CNN
trainDG = ImageDataGenerator(rescale=valorMult, preprocessing_function=preprocess_input, width_shift_range=valorRange1,
height_shift_range=valorRange2, horizontal_flip=valorBool, vertical_flip=valorBool, rotation_range=valorRange3)
# obs: valorMult é o valor pelo qual os pixels serão multiplicados para alteração de escala (não é necessário fazer nos dados de validação)
# obs2: preprocess_input prepara as imagens, ou seja, estabiliza as entradas para funções de ativação não lineares
# obs3: os demais parâmetros são referentes ao data augmentation
# obs4: caso for utilizado o data augmentation, criar um outro objDG sem data augmentation para o loteVal
# obs5: valorRange é o valor máximo dos possíveis valores aleatórios gerados pelo Keras

# para criar os batches para o treino da CNN
loteTreino = trainDG.flow_from_directory('endereçoDirTreino', batch_size=tamanhoLote, class_mode='categorical')
# obs: as imagens de cada classe deverão ser inseridas em uma SUBPASTA dentro do diretório informado
# tamanhoLote: referente ao tamanho do lote de imagens, ou seja, quantidade de dados de entrada que serão serão passados
#EM CADA ÉPOCA no treinamento, definir este parâmetro a partir da quantidade de memória disponível na GPU, caso a memória
#estourar, será necessário diminuir o tamanho do lote

# para criar um objeto para gerar as imagens de treino para CNN
valDG = ImageDataGenerator(preprocessing_function=preprocess_input)
# obs2: preprocess_input prepara as imagens, ou seja, estabiliza as entradas para funções de ativação não lineares

# para criar os batches para a validação da CNN
loteVal = valDG.flow_from_directory('endereçoDirVal', batch_size=tamanhoLote, class_mode='categorical')
# obs: as imagens de cada classe deverão ser inseridas em uma SUBPASTA dentro do diretório informado
# tamanhoLote: referente ao tamanho do lote de imagens que serão passados na validação do modelo

# para visualizar as características do modelo
nomeModelo.summary()

# para criar uma função de callback para salvar o modelo durante o treinamento
objSalva = ModelCheckpoint(filepath='endereçoDir/nomeArquivo_{epoch}.hdf5', monitor='val_accuracy', save_best_only=True, mode='auto')
# obs: se "save_best_only" estiver habilitado, será salvo apenas o melhor modelo de acordo com a métrica contida em "monitor",
#caso contrário, será salvo cada modelo ao final de cada época

# para treinar o modelo
objTreino = nomeModelo.fit_generator(loteTreino, epochs=nEpocas, validation_data=loteVal, callbacks=[objSalva])

# plotando gráficos de acurácia e loss
pyplot.figure()
pyplot.subplots_adjust(hspace=0.8)
pyplot.subplot(2,1,1)
pyplot.plot(objTreino.history['accuracy'])
pyplot.plot(objTreino.history['val_accuracy'])
pyplot.title('Accuracy X Epochs')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['Treino', 'Validação'])
pyplot.subplot(2,1,2)
pyplot.plot(objTreino.history['loss'])
pyplot.plot(objTreino.history['val_loss'])
pyplot.title('Loss X Epochs')
pyplot.ylabel('Loss')
pyplot.xlabel('Epochs')
pyplot.legend(['Treino', 'Validação'])
pyplot.suptitle(nomeModelo.layer[0].name)
pyplot.savefig(nomeModelo.layer[0].name+'.png')









## TESTANDO O MODELO DE CNN TREINADO ##

# função para carregar o modelo
from tensorflow.python.keras.models import load_model

# carregar o preprocess_input de acordo com a arquitetura treinada
from keras.applications.NOMEARQUITETURA import preprocess_input

# função para gerar a matriz de confusão
from sklearn.metrics import confusion_matrix

# módulo para plotar a matriz de confusão
import seaborn



# carregando o modelo treinado
nomeModelo = load_model('endereçoModelo')

# criando um data generator para os dados de teste
testDG = ImageDataGenerator(preprocessing_function=preprocess_input)
loteTest = testDG.flow_from_directory('endereçoDirTest', batch_size=tamanhoLote, class_mode='categorical', shuffle=False)

# verificando as métricas obtidas no teste
nomeModelo.evaluate(loteTest)

# verificando o número de acertos por classe
arrayPred = nomeModelo.predict(loteTest)
acertosPred = numpy.argmax(arrayPred, axis=1) == loteTest.labels
for classe in loteTest.class_indices:
  idxClasses = numpy.where(loteTest.labels == loteTest.class_indices[classe])
  print(classe+':', numpy.count_nonzero(acertosPred[idxClasses]==True)*100/idxClasses[0].size, "%")

# plotando a matriz de confusão
matrizConf = confusion_matrix(loteTest.labels, numpy.argmax(pred, axis=1))
seaborn.set(font_scale=1)
seaborn.heatmap(matrizConf, annot=True, xticklabels=loteTest.class_indices.keys(),
yticklabels=loteTest.class_indices.keys(), linewidths=0.7, cmap='Blues')









## UTILIZANDO O MODELO DE CNN TREINADO ##

# função para carregar o modelo
from tensorflow.python.keras.models import load_model

# funções para carregar e ajustar a imagem
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# carregar o preprocess_input de acordo com a arquitetura treinada
from keras.applications.NOMEARQUITETURA import preprocess_input



# carregando o modelo treinado
nomeModelo = load_model('endereçoModelo')

# carregando a imagem
nomeImg = load_img('endereçoImg', target_size=(alturaImg, larguraImg))

# ajustando a imagem para inserí-la no modelo
inputImg = img_to_array(nomeImg)
inputImg = numpy.expand_dims(inputImg, axis=0)
inputImg = preprocess_input(inputImg)

# criando um data generator para os dados de teste (para identificar o id das classes)
testDG = ImageDataGenerator(preprocessing_function=preprocess_input)
loteTest = testDG.flow_from_directory('endereçoDirTest', batch_size=tamanhoLote, class_mode='categorical', shuffle=False)

# predizendo a imagem
arrayPred = nomeModelo.predict(inputImg)
idxClasse = numpy.where(arrayPred[0] == arrayPred.max())[0][0]
classe = list(loteTest.class_indices.keys())[idxClasse]