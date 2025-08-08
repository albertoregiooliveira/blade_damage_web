########################################################################################################################
# ImagePredict.py
# Desenvolvido por: Alberto Régio A. de Oliveira
#
# Descrição:
#
########################################################################################################################

import cv2
import numpy as np
import matplotlib as mpl
from module_interface.ImageInterface import ImageInterface
from module_predict.IAModelSingleton import IAModelSingleton

import time
from datetime import timedelta

#=======================================================================================================================
# Classe responsável por controlar a exibicao da imagem
class ImagePredict(ImageInterface):

    # Metodo inicial
    def __init__(self, image, configuration=None):

        super().__init__(image, configuration)
        self.model = IAModelSingleton.instance()
        self.predict_flag = False
        self.prediction = None
        self.text_predict_flag = False
        self.text_percentual = False
        self.text_layout = 0
        self.heatmap_flag = False
        # self.alpha = 0.4
        self.alpha = 1.0

        self.predict_flag = True
        self.text_predict_flag = True
        self.heatmap_flag = True

        # Cores (Normal, Descarga, Erosao, Pintura, Trinca, Outro)
        self.colors_background = {'N':(255, 255, 200), 'D':(200, 255, 200), 'E':(255, 200, 200), 'P':(200, 255, 255),
                                  'T':(120, 120, 255), 'O':(255, 200, 255)}

        # Define fonte
        self.FONT = cv2.FONT_HERSHEY_PLAIN

    # Gera uma imagem com as previsoes
    def get_image(self):

        # Recupera imagem original
        image = super().get_image()

        # Mostra predicao
        if self.predict_flag or self.text_predict_flag or self.heatmap_flag:

            # Calcula predicao se necessario
            if self.prediction is None:
                start = time.time()
                self.prediction = self.model.predicao(image, self.configuration)
                sec = time.time() - start
                self.status_text = 'Tempo gasto: %s segundos' % str(timedelta(seconds=sec))

                # self.predict_flag = True
                # self.text_predict_flag = True
                # self.heatmap_flag = True

            if self.heatmap_flag:
                image, _ = self._show_heatmap(image)

            if self.predict_flag:
                image, _ = self._show_grid_prediction(image)

            if self.text_predict_flag:
                image, _ = self._show_text_prediction(image)

            if self.text_predict_flag:
                image = self._show_voting_prediction(image)


        # Aplica o zoom
        image = self._apply_zoom(image)

        # Retorna a imagem
        return image


    # Mostra a grid do recorte com a borda na cor da previsao
    def _show_grid_prediction(self, image, image_crop_list=None):

        # TEMP
        if image_crop_list is None:
            image_crop_list = []

        # Recupera a grid
        grid = self.grid.reshape((self.grid.shape[0] * self.grid.shape[1], 2, 2))

        # Recupera a predicao da imagem
        crops_title_list = self.prediction[1]

        # Monta grid na imagem
        for i, (p1, p2) in enumerate(grid):

            # Executa transformacao na grid
            image_crop = image[p1[1]:p2[1], p1[0]:p2[0], :]

            # TEMP
            image_crop_individual = None
            if i < len(image_crop_list):
                image_crop_individual = image_crop_list[i]

            if i < len(crops_title_list) and crops_title_list[i] is not None:
                perc = 0
                color_grid = self.COLOR_GRAY
                exibe = True

                # Seleciona a previsao da classe vencedora
                for j, text in enumerate(reversed(crops_title_list[i])):
                    percentual = round(text[1],4)
                    if percentual > perc:
                        perc = percentual
                        color_grid = self.colors_background[text[0]]
                        if text[0] == 'N':
                            exibe = False

                # Destaca borda da classe com a cor da classe vencedora
                if exibe:
                    p11 = (2, 2)
                    p21 = (image_crop.shape[0] - 2, image_crop.shape[1] - 2)
                    cv2.rectangle(image_crop, p11, p21, color_grid, 2)

                    # TEMP
                    if image_crop_individual is not None:
                        cv2.rectangle(image_crop_individual, p11, p21, color_grid, 2)

            # Aplica imagem do recorte na imagem principal
            image[p1[1]:p2[1], p1[0]:p2[0], :] = image_crop

            # TEMP
            if image_crop_individual is not None:
                image_crop_list[i] = image_crop_individual

        return image, image_crop_list


    # Mostra rotulos dos recortes nas grids na imagem
    def _show_text_prediction(self, image, image_crop_list=None):

        # TEMP
        if image_crop_list is None:
            image_crop_list = []

        # Recupera a grid
        grid = self.grid.reshape((self.grid.shape[0] * self.grid.shape[1], 2, 2))

        # Recupera a predicao da imagem
        crops_title_list = self.prediction[1]

        # Monta grid na imagem
        for i, (p1, p2) in enumerate(grid):

            # Executa transformacao na grid
            image_crop = image[p1[1]:p2[1], p1[0]:p2[0], :]

            # TEMP
            image_crop_individual = None
            if i < len(image_crop_list):
                image_crop_individual = image_crop_list[i]

            if i < len(crops_title_list) and crops_title_list[i] is not None:

                if self.text_layout == 0:
                    self._show_text_layout_01(image_crop, crops_title_list[i], image_crop_individual)
                else:
                    self._show_text_layout_02(image_crop, crops_title_list[i], image_crop_individual)

            # Aplica imagem do recorte na imagem principal
            image[p1[1]:p2[1], p1[0]:p2[0], :] = image_crop

            # TEMP
            if image_crop_individual is not None:
                image_crop_list[i] = image_crop_individual

        return image, image_crop_list



    # Exibe texto no layout 01
    def _show_text_layout_01(self, image_crop, prediction_list, image_crop_individual=None):

        # Tamanho da fonte
        font_scale = 1

        # Percorre os textos
        max_index = np.argmax(([x[1] for x in reversed(prediction_list)]))
        text = list(reversed(prediction_list))[max_index]

        # Nao mostra classe normal
        if text[0] != 'N':

            # Define formato de exibicao (percentual ou numerico)
            if self.text_percentual:
                text = f'{text[0]}:{text[1] * 100:.2f}%'  # Percentual
            else:
                text = f'{text[0]}:{round(text[1],4):.4f}' # Decimal

            # Recupera cor do texto
            color = self.colors_background[text[0]]

            # Calcula posicao do texto
            textsize = cv2.getTextSize(text, self.FONT, font_scale, 2)[0]

            p11x = 3
            p11y = textsize[1] + 4
            p11 = (p11x, p11y)
            p21 = (p11x + textsize[0], p11y - textsize[1])

            cv2.rectangle(image_crop, p11, p21, color, -1)
            cv2.putText(image_crop, text, p11, self.FONT, font_scale, (0, 0, 0), 1)

            # TEMP
            if image_crop_individual is not None:
                cv2.rectangle(image_crop_individual, p11, p21, color, -1)
                cv2.putText(image_crop_individual, text, p11, self.FONT, font_scale, (0, 0, 0))

        return image_crop


    # Exibe texto no layout 02
    def _show_text_layout_02(self, image_crop, prediction_list, image_crop_individual=None):

        # Percorre os textos
        lj = len(prediction_list)
        # for j, text in enumerate(reversed(prediction_list)):
        for j, text in enumerate(prediction_list):

            # Define formato de exibicao (percentual ou numerico)
            if self.text_percentual:
                text = f'{text[0]}:{text[1] * 100:.2f}%'  # Percentual
            else:
                text = f'{text[0]}:{round(text[1],4):.4f}' # Decimal

            # Recupera cor do texto
            color = self.colors_background[text[0]]

            # Calcula posicao do texto
            textsize = cv2.getTextSize(text, self.FONT, 1, 2)[0]

            p11x = (image_crop.shape[0] // 2) - (textsize[0] // 2)
            # p11y = (image_crop.shape[1] // 2) + (textsize[1] // 2) - (15 * j)

            all_text_sizes = lj * textsize[1]
            all_text_spaces = (lj - 1) * 5
            all_text_area = all_text_sizes + all_text_spaces
            top_offset = (image_crop.shape[1] - all_text_area) // 2

            prior_text_sizes = (lj - j) * textsize[1]
            prior_text_spaces = (lj - j - 1) * 5
            prior_text_area = prior_text_sizes + prior_text_spaces
            text_offset = all_text_area - prior_text_area

            p11y = top_offset + text_offset

            # offset = (image_crop.shape[1] - all_textsizes - all_textspaces) // 2
            # p11y = offset + all_textsizes + all_textspaces
            # p11y = ((lj - j - 1) * textsize[1]) + ((lj - j - 1) * 5) + (((lj * textsize[1]) + ((lj - 1) * 5)) // 2)

            p11 = (p11x, p11y)
            p21 = (p11x + textsize[0], p11y - textsize[1])

            cv2.rectangle(image_crop, p11, p21, color, -1)
            cv2.putText(image_crop, text, p11, self.FONT, 1, (0, 0, 0))

            # TEMP
            if image_crop_individual is not None:
                cv2.rectangle(image_crop_individual, p11, p21, color, -1)
                cv2.putText(image_crop_individual, text, p11, self.FONT, 1, (0, 0, 0))


        return image_crop


    # Mostra previsao geral da imagem
    def _show_voting_prediction(self, image):

        pred_title = self.prediction[0]

        # Texto da predição
        if pred_title is not None:
            textsize = cv2.getTextSize(pred_title, self.FONT, 1.5, 2)[0]
            cv2.rectangle(image, (10,30+textsize[1]), (10+textsize[0],20), self.COLOR_GRAY, -1)
            cv2.putText(image, pred_title, (10,textsize[1]+25), self.FONT, 1.5, (255, 255, 255))

            # cv2.rectangle(image, (10, 40 + textsize[1]), (10 + textsize[0], 30), self.COLOR_GRAY, -1)
            # cv2.putText(image, pred_title, (10,textsize[1]+35), self.FONT, 1.5, (255, 255, 255))

        return image


    # Recupera o heatmap das previsoes
    def get_heatmap(self, heatmap, image):

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]
        # jet = mpl.colormaps["jet_r"]

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = cv2.resize(jet_heatmap, (image.shape[1], image.shape[0]))
        jet_heatmap = np.uint8(255 * jet_heatmap)

        return jet_heatmap


    # Aplica o heatmap a imagem
    def _show_heatmap(self, image, image_crop_list=None):

        # TEMP
        if image_crop_list is None:
            image_crop_list = []

        class_pred = self.prediction[0][0]
        crop_winner_pred = []

        for crop_pred in self.prediction[1]:
            winner = max(crop_pred, key=lambda x: x[1])
            if winner[0] == class_pred and winner[1] > 0.5:
                crop_winner_pred.append(winner[1])
            else:
                crop_winner_pred.append(0)

        # for crop_pred in self.prediction[1]:
        #     for crop_class_pred in crop_pred:
        #         if crop_class_pred[0] == class_pred:
        #             crop_winner_pred.append(crop_class_pred[1])





        crop_winner_pred = np.array(crop_winner_pred).reshape((self.grid.shape[0], self.grid.shape[1]))

        # Superimpose the heatmap on original image
        heatmap = self.get_heatmap(crop_winner_pred, image)
        superimposed_img = heatmap * self.alpha + image
        superimposed_img *= (255.0 / superimposed_img.max())
        superimposed_img = superimposed_img.astype(np.uint8)

        # TEMP
        for i, crop in enumerate(image_crop_list):
            heatmap = self.get_heatmap(crop_winner_pred.flatten()[i].reshape((1,1)), crop)
            superimposed_crop = heatmap * self.alpha + crop
            superimposed_crop *= (255.0 / superimposed_crop.max())
            superimposed_crop = superimposed_crop.astype(np.uint8)
            image_crop_list[i] = superimposed_crop

        return superimposed_img, image_crop_list


    # Retorna um vetor com os recortes
    def get_images_sliced(self, x_size = None, y_size = None):

        sliced_list = super().get_images_sliced(None, None)

        # Recupera imagem original
        image = self.image_original.copy()
        cropped_list = self._get_crops(image)

        if self.heatmap_flag:
            _, cropped_list = self._show_heatmap(image, cropped_list)

        if self.predict_flag:
            _, cropped_list = self._show_grid_prediction(image, cropped_list)

        if self.text_predict_flag:
            _, cropped_list = self._show_text_prediction(image, cropped_list)

        # Armazena imagem do recorte
        for i, cropped in enumerate(cropped_list):
            name = str(i+1).zfill(3)
            sliced_list.append((cropped, name))

        return sliced_list


    # Recupera recortes da imagem original
    def _get_crops(self, image):

        # Lista com recortes separados
        image_crop_list = []

        # Recupera a grid
        grid = self.grid.reshape((self.grid.shape[0] * self.grid.shape[1], 2, 2))

        # Monta grid na imagem
        for i, (p1, p2) in enumerate(grid):

            # Executa transformacao na grid
            image_crop = image[p1[1]:p2[1], p1[0]:p2[0], :]
            image_crop_list.append(image_crop.copy())

        return image_crop_list