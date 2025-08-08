########################################################################################################################
# ImageBoundingBox.py
# Desenvolvido por: Alberto Régio A. de Oliveira
#
# Descrição: Aplica em uma única imagem principal o zoom e/ou camadas de exibição das caixas delimitadoras.
#
########################################################################################################################

import math
import numpy as np
import cv2

from module_interface.ImageInterface import ImageInterface

# Classe responsavel por processar uma única imagem
class ImageBoundingBox(ImageInterface):

    # Metodo inicial
    def __init__(self, image, configuration=None):

        super().__init__(image, configuration)

        # Lista dos bounding boxes
        self.bb_list = []

        # Cria uma mascara
        self.mask = np.zeros(self.image.shape[:2], np.uint8)

        # Tamanho da área do mouse para "pintar" a segmentação
        self.segment_size = 20
        self.segment_mode = False

        # Esconde bounding boxes
        self.hide_bb = False
        self.hide_segment = False
        self.hide_grid = True

        # Define cor
        green = (0, 255, 0)
        white = (255, 255, 255)
        self.ADD_SEGMENT_COLOR = 255
        self.REMOVE_SEGMENT_COLOR = 0

        # Atribui cores
        self.descriptions = ["Descarga Atmosferica", "Pintura", "Erosao", "Trincas", "Normal", "Outro"]
        self.colors = {'N':(255, 255, 0), 'D':(0, 255, 0), 'E':(255, 0, 0), 'P':(0, 255, 255),
                       'T':(0, 0, 255), 'O':(255, 0, 255), 'X':(120, 0, 120)}

        # Define o texto a ser colocado no topo da imagem
        self.title = ""
        self.title_suffix = ""

        # Define bounding box de preview
        self.bb_preview = None

        # Lista rotulos dos recortes
        self.crops_title_list = []

        # Define como gerar os recortes
        self.SLICED_BY_GRID = 0
        self.SLICED_BY_BB = 1
        self.sliced_mode = self.SLICED_BY_GRID


    # Calcula as coordenadas dos pontos x1,y1 e x2,y2 e recalcula o centro caso seja necessario
    def _calculate_bb_coordinates(self, bb):

        # Recupera dados da bb
        cx, cy, sx, sy, text = bb

        # Calcula a distancia de x1 e x2 em relacao ao centro
        sx1 = sx // 2
        sx2 = sx - sx1

        # Calcula a distancia de y1 e y2 em relacao ao centro
        sy1 = sy // 2
        sy2 = sy - sy1

        # Calcula e ajusta a posicao de x1 para nao ficar fora da tela
        x1 = cx - sx1
        if x1 < 0:
            cx -= x1
            x1 = 0

        # Calcula e ajusta a posicao de y1 para nao ficar fora da tela
        y1 = cy - sy1
        if y1 < 0:
            cy -= y1
            y1 = 0

        # Calcula e ajusta a posicao de x2 para nao ficar fora da tela
        x2 = cx + sx2
        if x2 > self.image_size[0]:
            dif = x2 - self.image_size[0]+1
            cx -= dif
            x2 -= dif
            x1 -= dif

        # Calcula e ajusta a posicao de y2 para nao ficar fora da tela
        y2 = cy + sy2
        if y2 > self.image_size[1]:
            dif = y2 - self.image_size[1]+1
            cy -= dif
            y2 -= dif
            y1 -= dif

        # Define as coordenadas de inicio e fim
        p1 = (x1, y1)
        p2 = (x2, y2)
        center = (cx, cy)
        size = (sx, sy)

        # Retorna as coordenadas
        return p1, p2, center, size


    # Desenha um bounding box na imagem
    def _draw_bb(self, image, bb, color=(0, 255, 0)):

        # Cor branca
        white = (255, 255, 255)

        # Recupera dados da bb
        cx, cy, sx, sy, text = bb

        # Recupera pontos da imagem original
        p1, p2, center, size = self._calculate_bb_coordinates(bb)

        # Ajusta todos dos pontos de acordo com o zoom
        p1 = self._calculate_zoom_point_position(p1)
        p2 = self._calculate_zoom_point_position(p2)
        cx, cy = self._calculate_zoom_point_position((cx, cy))

        # Recupera valores individuais dos pontos
        x1, y1 = p1
        x2, y2 = p2

        # Desenha retangulo
        cv2.rectangle(image, p1, p2, color)

        # Desenha centro
        const = 2
        cv2.line(image, (cx, cy - const), (cx, cy + const), color)
        cv2.line(image, (cx - const, cy), (cx + const, cy), color)

        # Desenha o tamanho do recorte
        size_text = f"{size[0]} x {size[1]}"
        fonte = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, size_text, (x1+2, y2-3), cv2.FONT_HERSHEY_PLAIN, 0.6, white)

        # Desenha o tipo de problema
        if text != "":
            cv2.putText(image, text, (x1+2, y1+13), fonte, 1, white)


    # Aplica os bounding boxes na imagem
    def _apply_bb(self, image):

        # Desenha os bounding boxes
        for bb in self.bb_list:

            # Recupera dados da bb
            cx, cy, sx, sy, text = bb

            # Recupera o indice da cor
            try:
                index = self.descriptions.index(text)
            except ValueError:
                index = len(self.descriptions) -1

            class_type = self.descriptions[index][0]
            self._draw_bb(image, bb, color=self.colors[class_type])

        return image


    # Aplica os bounding boxes na imagem
    def _apply_mask(self, image):

        alpha = 0.80
        beta = (1.0 - alpha)

        color_background = np.full((self.image.shape[0], self.image.shape[1], 3), (0,255,0), np.uint8)
        color_image = cv2.addWeighted(image, alpha, color_background, beta, 0.0)
        color_image_mask = cv2.bitwise_or(color_image, color_image, mask=self.mask)
        image = cv2.add(image, color_image_mask)

        return image


    # Verifica se um retângulo possui interseção com alguma bounding box
    def _get_overlap_bb_text(self, p1, p2):
        text_list = []
        for bb in self.bb_list:

            # Recupera informações do bounding box
            _, _, _, _, text = bb
            bb_p1, bb_p2, _, _ = self._calculate_bb_coordinates(bb)

            # Calcula área de interseção
            dx = min(p2[0], bb_p2[0]) - max(p1[0], bb_p1[0])
            dy = min(p2[1], bb_p2[1]) - max(p1[1], bb_p1[1])

            # Verifica se houve interseção
            if (dx >= 0) and (dy >= 0):
                text_list.append(text)

        return text_list


    # Verifica se um retângulo possui alguma marcacao de segmentacao
    def _get_overlap_segment_text(self, p1, p2):
        text_list = []
        for bb in self.bb_list:

            # Recupera informações do bounding box
            _, _, _, _, text = bb
            bb_p1, bb_p2, _, _ = self._calculate_bb_coordinates(bb)

            # Calcula área de interseção
            dx = min(p2[0], bb_p2[0]) - max(p1[0], bb_p1[0])
            dy = min(p2[1], bb_p2[1]) - max(p1[1], bb_p1[1])

            # Verifica se houve interseção
            if (dx >= 0) and (dy >= 0):
                text_list.append(text)

        return text_list


    # Desenha a área de um segmento na imagem
    def paint_segment(self, center, color):
        center = self._calculate_original_point_position(center)
        cv2.circle(self.mask, center, self.segment_size, color, -1)


    # Retorna se os bounding boxes estão sendo mostrados ou não
    def get_hide_bb(self):
        return  self.hide_bb


    # Define se mostra ou não os bounding boxes
    def set_hide_bb(self, flag):
        self.hide_bb = flag


    # Gera uma imagem com os bounding boxes
    def get_image(self):

        # Reinicia o texto do titulo
        self.title = ""

        # Recupera imagem original
        image = super().get_image()

        # Aplica mascara
        if not self.hide_segment:
            image = self._apply_mask(image)

        # Aplica o zoom
        image = self._apply_zoom(image)

        # Aplica os bounding boxes
        if not self.hide_bb:
            image = self._apply_bb(image)

            # Desenha o bounding box de preview
            if not self.bb_preview is None:
                index = len(self.descriptions) - 1
                self._draw_bb(image, self.bb_preview, color=self.colors['X'])

        # Exibe texto de título da imagem
        # self._apply_title(image)

        # Texto da barra de status
        self.status_text = f'{self.grid.shape[0] * self.grid.shape[1]} recortes na grade'
        self.status_text += f' | resolucao: {self.image_original.shape[1]}x{self.image_original.shape[0]} pixels'
        self.status_text += f' | zoom: {image.shape[1]}x{image.shape[0]} pixels'

        # Retorna a imagem
        return image


    # Retorna um vetor com as imagens fatiadas a partir dos bounding boxes
    def get_images_sliced(self, x_size = None, y_size = None):
        if self.sliced_mode == self.SLICED_BY_GRID:
            return self.get_images_sliced_by_grid(x_size, y_size)
        else:
            return self.get_images_sliced_by_bb(x_size, y_size)


    # Retorna um vetor com as imagens fatiadas a partir dos bounding boxes
    def get_images_sliced_by_bb(self, x_size = None, y_size = None):

        sliced_list = super().get_images_sliced(x_size, y_size)

        # Trata o tamanho da imagem de extração
        x_size = x_size if x_size is not None else self.configuration.bb_x_size
        y_size = y_size if y_size is not None else self.configuration.bb_y_size

        # Desenha os danos na figura
        num = 0
        for bb in self.bb_list:
            cx, cy, sx, sy, text = bb
            num += 1

            # Fatia a imagem
            p1, p2, _, _ = self._calculate_bb_coordinates(bb)
            cropped = self.image_original[p1[1]:p2[1], p1[0]:p2[0]]

            # # Redimensiona a imagem
            # if x_size != sx or y_size != sy:
            #     cropped = cv2.resize(cropped, (x_size, y_size), interpolation=cv2.INTER_AREA)

            num_text = str(num).zfill(3)
            name = f"{num_text}_{text}"

            sliced_list.append((cropped, name))

        return sliced_list


    # Retorna um vetor com as imagens fatiadas a partir da grid
    def get_images_sliced_by_grid(self, x_size = None, y_size = None):

        sliced_list = super().get_images_sliced(x_size, y_size)
        num = 0

        grid = self.grid.reshape((self.grid.shape[0] * self.grid.shape[1], 2, 2))
        for p1, p2 in grid:
            num += 1

            # Fatia a imagem
            cropped = self.image_original[p1[1]:p2[1], p1[0]:p2[0]]

            # Fatia a imagem
            cropped_mask = self.mask[p1[1]:p2[1], p1[0]:p2[0]]
            cropped_mask = np.where(cropped_mask > 0, 1, cropped_mask)
            segment_found = 1 in cropped_mask

            # Calcula percentual de representatividade do dano em relacao a imagem total
            uv = np.unique(cropped_mask, return_counts=True)
            total = uv[1][1] / np.sum(uv[1]) if len(uv[1]) > 1 else 0
            percent_text = '10000' if total == 1 else '0' + '{:.4f}'.format(total)[-4:]

            # # Redimensiona a imagem
            # if x_size != sx or y_size != sy:
            #     cropped = cv2.resize(cropped, (x_size, y_size), interpolation=cv2.INTER_AREA)

            if segment_found:
                text = 'damage'
            else:
                text = 'normal'

            num_text = str(num).zfill(3)
            name = f"{num_text}_{text}_{percent_text}"

            sliced_list.append((cropped, name))

        return sliced_list


    # Adiciona um novo bounding box
    def preview_bb(self, center, size, description):

        if center is not None:

            # Recalcula o centro
            center = self._calculate_original_point_position(center)

            # Armazena pontos x e y
            x = center[0]
            y = center[1]

            # Define bounding box de preview
            self.bb_preview = (x, y, size[0], size[1], description)

        else:
            # Define bounding box de preview
            self.bb_preview = None


    # Adiciona um novo bounding box
    def add_bb(self, center, size, description):

        # Recalcula o centro
        center = self._calculate_original_point_position(center)

        # Armazena pontos x e y
        x = center[0]
        y = center[1]

        # Adiciona valores a lista
        self.bb_list.append((x, y, size[0], size[1], description))


    # Remove um bounding box existente
    def remove_bb(self, center):

        # Recalcula o centro
        center = self._calculate_original_point_position(center)

        dist = None
        best = None

        # Percorre a lista de bounding boxes
        for n, bb in enumerate(self.bb_list):
            bb_center = (bb[0], bb[1])
            d = math.dist(center, bb_center)

            # Armazena indice que possuí a menor distância
            if dist == None:
                dist = d
                best = n
            else:
                if d < dist:
                    dist = d
                    best = n

        # Remove bounding box de melhor correspondencia
        if best is not None:
            self.bb_list.pop(best)

