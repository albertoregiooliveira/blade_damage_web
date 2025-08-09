########################################################################################################################
# IAModel.py
# Desenvolvido por: Alberto Régio A. de Oliveira
#
# Descrição:
#
########################################################################################################################
import os.path
import tensorflow as tf

from collections import Counter

from BladeDamageConfiguration import BladeDamageConfiguration
from module_boundingbox.ImageBoundingBox import ImageBoundingBox
from module_interface.model_constants import *


#=======================================================================================================================
# Classe responsável carregar o modelo de IA
class IAModel:

    # Metodo inicial
    def __init__(self, model_filename=None, usa_paralelismo=False, image_dim=320, backbone_function=None, preprocess_function=None):

        # TEMP
        # tf.config.set_visible_devices([], 'GPU')

        # Resolucao das imagens de entrada.
        self.img_dim = image_dim
        # self.img_dim = 120

        # Determina arquivo do modelo
        if model_filename is None:
            model_filename = './model/25_rand_noagr_augum_efficientnetb2_3.weights.h5'

        # Armazena caminho do modelo
        self.model_filename = model_filename

        # Armazena função de pré-processamento do backbone
        if backbone_function is None:
            self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        else:
            self.preprocess_input = preprocess_function

        # Armazena função de criação do backbone
        if backbone_function is None:
            self.backbone = tf.keras.applications.EfficientNetB2
        else:
            self.backbone = backbone_function

        # Cria modelo
        self.model = self.create_model(usa_paralelismo=usa_paralelismo)


    # Função para carregar o modelo
    def load_model(self):
        self.model.load_weights(self.model_filename)


    # Função para salvar o modelo
    def save_model(self):
        self.model.save_weights(self.model_filename)


    # Função para criar um novo modelo
    def create_model(self, usa_paralelismo=False):

        # Define a estratégia de processamento distribuido
        if tf.config.list_physical_devices('GPU') and usa_paralelismo:
            strategy = tf.distribute.MirroredStrategy()
            # strategy = tf.distribute.experimental.CentralStorageStrategy()
            # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
        else:  # Use the Default Strategy
            strategy = tf.distribute.get_strategy()

        with strategy.scope():

            # #  Carrega o backbone pré-treinado na base ImageNet
            # backbone = tf.keras.applications.EfficientNetB2(input_shape=(self.img_dim, self.img_dim, 3), include_top=False, weights='imagenet')
            # #  Congela o backbone
            # backbone.trainable = False

            #  Cria o modelo usando a API funcional do keras

            # Camada de entrada
            input = tf.keras.Input(shape=(None, None, 3))

            # Camadas ocultas
            layers = input
            # layers = tf.cast(input, tf.float32)
            layers = tf.keras.layers.Resizing(self.img_dim, self.img_dim)(layers) # Redimensionamento TEMP
            # layers = tf.keras.applications.efficientnet.preprocess_input(layers)
            layers = self.preprocess_input(layers)

            # layers = backbone(layers, training=False)  # passa o backbone
            # layers = tf.keras.layers.GlobalAveragePooling2D()(layers)  # GAP

            #  Carrega o backbone pré-treinado na base ImageNet
            # backbone = tf.keras.applications.EfficientNetB2(input_shape=(self.img_dim, self.img_dim, 3), input_tensor=layers, include_top=False, weights='imagenet')
            backbone = self.backbone(input_shape=(self.img_dim, self.img_dim, 3), input_tensor=layers, include_top=False, weights='imagenet')
            backbone.trainable = False
            layers = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)  # GAP

            # Camada de saída
            output = tf.keras.layers.Dense(5, activation='softmax')(layers)  # saida

            # Modelo
            model = tf.keras.models.Model(inputs=input, outputs=output, name="minha_rede")  # cria o modelo

        return model


    #
    def predicao(self, imagem, configuration=None):

        if configuration is None:
            configuration = BladeDamageConfiguration(None)

        # Gera recortes da imagem principal
        def gera_recortes(image):

            # Instância um objeto da classe ImageBoundingBox
            # TODO: criar classe especifica
            ibb = ImageBoundingBox(image, configuration)
            # ibb.overlap = 32

            # Gera recortes
            return ibb.get_images_sliced()

        # Preve classes dos recortes
        def preve_recortes(crops):

            # Lista para armazenar resultados
            result = []

            # Percorre crops para realizar a predição
            for i in range(len(crops)):

                # Lista com resultado dos crops
                # crop_result = []
                crop_result = {}

                # Realiza a predição do modelo
                im_crop = np.array([crops[i][0]])
                # if im_crop.shape[1] != self.img_dim or im_crop.shape[2] != self.img_dim:
                #     im_crop = cv2.resize(im_crop, (self.img_dim, self.img_dim))

                pred = self.model.predict(im_crop, verbose=0)

                # Armazena percentuais
                for j in range(pred.size):
                    crop_result[CLASSES_LABELS[j]] = pred[0][j]

                # Armazena dados do recorte
                result.append({'recorte': i, 'classe': CLASSES_LABELS[np.argmax(pred, axis=-1)[0]], 'probabilidades': crop_result})

            # Retorna resultado
            return result

        # Realiza a contagem da predição dos recortes
        def realiza_votacao(prediction):

            # Agrega o somatório das previsões por classe
            pred_by_crop = [(crop['classe'], crop['probabilidades'][crop['classe']]) for crop in prediction]
            pred_by_classes_dict = {}
            for key, value in pred_by_crop:
                if key in pred_by_classes_dict:
                    pred_by_classes_dict[key] += value
                else:
                    pred_by_classes_dict[key] = value

            # Calcula a quantidade de previsões dos recortes
            c = Counter([crop['classe'] for crop in prediction])

            # Remove classe Normal, caso existam recortes preditos como dano
            if len(c.most_common()) > 1:
                if 'Normal' in c.elements():
                    c.pop('Normal')

            # Recupera a contagem de recortes vencedora
            max_count = c.most_common()[0][1]
            max_class = (None, 0)

            # Pode haver empate entre as classes vencedoras, então é necessário verificar o desempate
            # Verifica qual classe que tem a maior soma de probabilidades
            for key, value in c.most_common():
                if value == max_count:
                    if pred_by_classes_dict[key] > max_class[1]:
                        max_class = (key, pred_by_classes_dict[key])
                else:
                    break

            # Retorna a classe vencedora
            return max_class[0]

            # # Calcula o resultado da votação
            # if c.most_common()[0][0] == 'Normal' and len(c.most_common()) > 1:
            #     return c.most_common()[1][0]
            # else:
            #     return c.most_common()[0][0]

        # # Retorna lista com probabilidades previstas para montar o mapa de calor
        # def calcula_mapa_calor(label, prediction):
        #     return [crop['probabilidades'][label] if crop['classe'] == label else 0 for crop in prediction]

        # Retorna lista com probabilidades previstas para montar o mapa de calor
        def calcula_mapa_calor(label, prediction):
            # return [[f"N:{crop['probabilidades']['Normal'] * 100:.2f}%",
            #          f"D:{crop['probabilidades']['Descarga'] * 100:.2f}%",
            #          f"E:{crop['probabilidades']['Erosao'] * 100:.2f}%",
            #          f"P:{crop['probabilidades']['Pintura'] * 100:.2f}%",
            #          f"T:{crop['probabilidades']['Trinca'] * 100:.2f}%"]
            #         for crop in prediction]
            return [[ ('N',crop['probabilidades']['Normal']),
                      ('D',crop['probabilidades']['Descarga']),
                      ('E',crop['probabilidades']['Erosao']),
                      ('P',crop['probabilidades']['Pintura']),
                      ('T',crop['probabilidades']['Trinca'])    ]
                    for crop in prediction]

        # ===================================================================
        # Executa processo de predicao
        # imagem = carrega_imagem(filename)
        recortes = gera_recortes(imagem)
        previsao = preve_recortes(recortes)
        votacao = realiza_votacao(previsao)
        mapa_calor = calcula_mapa_calor(votacao, previsao)

        # Retorna valores
        return votacao, mapa_calor



# ======================================================================================================================
# METODO PRINCIPAL - Exemplo de uso
if __name__ == "__main__":
    model_filename = os.path.join(get_PATH_MODEL(), 'balanced_model_efficientnetb2', 'balanced_model_efficientnetb2_weights_0.h5')
    iamodel = IAModel(model_filename)
    iamodel.load_model()