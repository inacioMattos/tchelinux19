# Importando o TensorFlow com o apelido 'tf'
import tensorflow as tf


# Pegando o MNIST dos datasets que já vem por padrão com o TensorFlow
# O MNIST é um conjunto de milhares de imagens 28x28px de digítos escritos a mão (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
mnist = tf.keras.datasets.mnist


# Pegando as imagens e rotulos delas do MNIST
# E separando as imagens em '_treino' e '_teste'
# As '_treino' são para o treino do modelo (o aprendizado em si)
# Enquanto as '_teste' são para o teste pós-treinamento do modelo
# img_treino e img_teste são vetores com as imagens 28x28px neles
# Vale destacar que as imagens em si, são matrizes de 28 por 28 com valores de pixel de 0 a 255
# Enquanto label_treino e label_teste são vetores com os rótulos de cada imagem em ordem
# Ou seja, o rótulo da label_treino[204] é o da img_treino[204]; bem como o label_teste[1234] é o da img_teste[1234]
(img_treino, label_treino), (img_teste, label_teste) = mnist.load_data()

# Caso você queira visualizar alguma das imagens do MNIST, basta descomentar as próximas 3 linhas de código:
# import matplotlib.pyplot as plt
# plt.imshow(img_treino[10])
# plt.show()


# Como já citado acima, os valores dos pixels das imagens vão de 0 a 255, entretanto, é boa prática sempre ter seus valores entre 1 e 0
# OBS: Caso você queira entender o porquê, pesquise a respeito de funções de ativações (Sigmoid e Relu)
# Então, aqui está sendo 'normalizados' esses valores para serem de 1 a 0
# O que essa função está fazendo é, simplesmente, dividando cada valor individualmente por 255
img_treino = tf.keras.utils.normalize(img_treino)
img_teste = tf.keras.utils.normalize(img_teste)


# Criando o modelo e dizendo que é do tipo sequencial.
# O tipo 'Sequential' quer dizer, somente, que é uma rede neural padrão, onde as camadas vão da esquerda para a direita. Ou seja, de maneira sequencial.
modelo = tf.keras.models.Sequential()


# Adicionado a primeira camada ao nosso modelo
# 'Flatten' pois nossas imagens são na verdade matrizes 28x28, então, usamos o 'Flatten' para planificar essas matrizes, transformando elas em vetores de tamanho 784
modelo.add(tf.keras.layers.Flatten(input_shape=(28, 28)))


# Adicionando as duas camadas escondidas do nosso modelo, cada uma com 16 neurônios
# E especificando o tipo de algoritmo de ativação a ser usado (Relu, Sigmoid, Softmax, etc)
# Vale a pena pesquisar sobre, pois isso influencia bastante
modelo.add(tf.keras.layers.Dense(16, activation=tf.nn.sigmoid))
modelo.add(tf.keras.layers.Dense(16, activation=tf.nn.sigmoid))


# Adicionando a última camada ao nosso modelo
# Uma camada padrão (dense) com 10 neurônios, pois há somente 10 possíveis resultados, já que há somente 10 digitos: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
modelo.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))


# Configurando as opções de treinamento do nosso modelo
# optimizer -> É o algoritmo a ser usado para encontrar os valores dos pesos das conexões dos neurônios
# Vale a pena pesquisar para conhecer outros algoritmos, entretanto, em 99,99% dos casos atuais, o 'adam' é o mais eficiente
# loss -> É como a loss será calculada
# Vale destacar que é a partir da loss que o modelo se ajusta, então vale muuuuito a pena pesquisar a respeito disso para obter melhor performance
# metrics -> Vetor com as métricas que serão mostradas enquanto estamos treinando o modelo
# Muito útil para degubar o modelo e entender ele melhor
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Iniciando, de fato, o treinamento do modelo
# modelo.fit(imagens_de_treino, labels_dela, epochs)
# epochs -> Número de vezes que o modelo vai percorer completamente os dados de treinamento
modelo.fit(img_treino, label_treino, epochs=3)


# Fazendo o cálculo da accuracy (precisão) e da loss (perda) com os dados de testes
# Essa é a melhor maneira de se obter essas métricas, pois são com imagens que o modelo nunca viu antes
# O que evita memorização e coisas similares
val_loss, val_acc = modelo.evaluate(img_teste, label_teste)
print ('\nloss: {}  -   accuracy: {}'.format(val_loss, val_acc))
