# Desafio-GustavoSilva

Código desenvolvido por: Gustavo Bandeira da Silva

O código foi dividido entre cinco arquivos: model, train e teste

Ferramentas utilizadas:

-Keras (https://github.com/keras-team/keras)

-Augmentor (https://github.com/mdbloice/Augmentor)

-Numpy (https://github.com/numpy/numpy)

-Cv2 (https://github.com/opencv/opencv/tree/master/samples/python)

#Obs: o programa só funcionara com imagens bmp;
#Obs: o arquivo model-3000.h5 contém os pesos da rede neural treinada com 3000 imagens

Para treinar utilize train.py:
Usage: python train.py <com_anel_directory> <sem_anel_directory> [model_save_name.h5]

Para testar utilize test.py:
Usage: python test.py <weights_filename> <test_image_directory>


O arquivo model.py contém a modelagem da rede neural criada com a ferramenta Keras para a resolução do problema. O modelo usado foi o sequencial (pilha linear de camadas). Em seguida são utilizados camadas  neurais convolucionais e a tecnica de Maxpooling para fornecer uma forma abstraída da representação, reduzinho o custo computacional. A função de perda utilizada é a binary_crossentropy, já que a modelagem trata de um problema de 0 a 1, sendo considerada uma imagem com anel uma predição acima de 0.5 (50%).


O arquivo train.py contém a parte de treinamento do modelo. Para realizar o treinamento, o programa segue o fluxo de:

- Data Augmentation: Uma boa forma de combater o Overfitting é angariando dados. Atráves do Data Augmentation, podemos criar mais dados apartir dos dados já existentes;

- Reduzir escala dos dados: A escala das imagens é reduzida para obter um desempenho maior na hora do treinamento. A redução foi feita até um ponto que ainda seja possível identificar se possui anel ou não com o olho humano;

- Modelagem e validação das arquiteturas: (Construção, Treinamento...)

- Deploy;

O arquivo teste.py carrega os pesos treinados da rede neural. Em seguida, procura os arquivos de teste no diretório escolhido e utiliza da predição carregada do arquivo para processar se uma imagem possui um anel ou não.





