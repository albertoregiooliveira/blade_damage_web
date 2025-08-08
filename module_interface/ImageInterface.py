########################################################################################################################
# ImageInterface.py
# Desenvolvido por: Alberto Régio A. de Oliveira
#
# Descrição:
#
########################################################################################################################

import cv2
import numpy as np

#=======================================================================================================================
# Classe responsável por controlar a exibicao da imagem
class ImageInterface:

    # Metodo inicial
    def __init__(self, image, configuration=None):

        # Valida e atribui configuracao
        assert configuration is not None, "A configuração não foi definida"
        self.configuration = configuration

        # Define cor
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GRAY = (50, 50, 50)
        self.color = self.COLOR_GREEN

        # Armazena dados da imagem
        self.image_original = image.copy()
        self.image = image
        self.image_size = (self.image.shape[1], self.image.shape[0])

        # Nível do zoom
        self.zoom = 0

        # Exibe/oculta grid
        self.hide_grid = True

        # Exibe/oculta lattice
        self.hide_lattice = True

        # Overlap
        self.__overlap_changed = True

        # Define grid
        self.__grid = self._get_grid()

        # Define texto da barra de status
        self.status_text = ''


        # TEMP TEMP TEMP TEMP ===================
        self.dx, self.dy = 0, 0
        # Recorte selecionado
        self.point_mouse = (0, 0)
        self.highlight_bb = False
        # Cores
        self.colors_grid = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.color_index = 0
        self.color_selected = 1
        # TEMP TEMP TEMP TEMP ===================


    """ 
      Define a propriedade grid.
      Retorna um valor já calculado ou realiza um novo calculo caso algum parâmetro tenha mudado.      
    """
    @property
    def grid(self):

        # Verifica se é necessário recalcular a grid
        if self.__overlap_changed:
            self.__grid = self._get_grid()
            self.__overlap_changed = False

        return self.__grid


    @property
    def grid_count(self):
        return self.grid.shape[0] * self.grid.shape[1]


    @property
    def overlap(self):
        return self.configuration.overlap


    @overlap.setter
    def overlap(self, value):
        self.configuration.overlap = value
        self.__overlap_changed = True


    # Retorna os pontos dos recortes na grid
    def _get_grid(self):

        # Lista de pontos
        grid_points = []

        # Define variáveis internas
        x_size, y_size = self.configuration.bb_size
        overlap = self.overlap

        # Calcula o deslocamento da grade
        ex1 = x_size - overlap
        ey1 = y_size - overlap

        # Calcula quantas grids cabem na tela
        x = self.image_size[0] // ex1
        y = self.image_size[1] // ex1

        # Verifica se a grid no eixo X contemplou toda a area da imagem
        x_remainder = self.image_size[0] % ex1
        if x_remainder > 0: x += 1
        if x > 1:
            nx = ((x * ex1) - self.image_size[0]) // (x - 1)
        else:
            nx = 0

        # Verifica se a grid no eixo Y contemplou toda a area da imagem
        y_remainder = self.image_size[1] % ey1
        if y_remainder > 0: y += 1
        if y > 1:
            ny = ((y * ey1) - self.image_size[1]) // (y - 1)
        else:
            ny = 0

        # Ajusta o deslocamento para gerar grids em toda a area da imagem
        ex1 -= nx
        ey1 -= ny

        grid_points_y = []

        # Calcula os pontos da grid
        for i2 in range(y):

            grid_points_x = []

            for i1 in range(x):

                # Calcula os pontos do eixo X
                x1 = i1 * ex1
                x2 = x1 + x_size
                if x2 > self.image_size[0]:
                    x1 -= x2 - self.image_size[0]
                    x2 -= x2 - self.image_size[0]

                # Calcula os pontos do eixo Y
                y1 = i2 * ey1
                y2 = y1 + y_size
                if y2 > self.image_size[1]:
                    y1 -= y2 - self.image_size[1]
                    y2 -= y2 - self.image_size[1]

                # Define pontos
                p1 = (x1, y1)
                p2 = (x2, y2)

                grid_points_x.append((p1, p2))
                grid_points.append((p1, p2))

            grid_points_y.append(grid_points_x)

        return np.array(grid_points_y)


    # Aplica os grids na imagem
    def _apply_grid(self, image):

        # Define cor da grid
        color = self.COLOR_GRAY

        # Recupera a grid
        grid = self.grid.reshape((self.grid.shape[0] * self.grid.shape[1], 2, 2))

        # Monta grid na imagem
        for i, (p1, p2) in enumerate(grid):

            #TEMP
            p1 = (p1[0]+self.dx, p1[1]+self.dy)
            p2 = (p2[0] + self.dx, p2[1] + self.dy)

            # Desenha retangulo
            cv2.rectangle(image, p1, p2, color)

        # Seleciona grid
        index_grid = self.select_grid(self.point_mouse)
        if index_grid >= 0 and self.highlight_bb:
            p1, p2 = grid[index_grid]
            p1 = (p1[0]+self.dx, p1[1]+self.dy)
            p2 = (p2[0] + self.dx, p2[1] + self.dy)
            color = self.colors_grid[self.color_selected]
            cv2.rectangle(image, p1, p2, color)

        return image


    # Aplica os lattice na imagem
    def _apply_lattice(self, image):

        # Define cor da grid
        # color = self.COLOR_GRAY
        color = self.COLOR_GREEN

        dx, dy = self.dx, self.dy

        # Recupera a grid
        grid = self.grid.reshape((self.grid.shape[0] * self.grid.shape[1], 2, 2))

        # Monta grid na imagem
        for i, (p1, p2) in enumerate(grid):
            const = 4
            center = ((p2 - p1) // 2) + p1
            center = (center[0]+dx, center[1]+dy)
            cv2.circle(image, center, const, color, -1)
            cv2.line(image, (0, center[1]), (self.image_size[0], center[1]), color)
            cv2.line(image, (center[0], 0), (center[0], self.image_size[1]), color)

        return image


    # Coloca rotulos dos recortes nas grids na imagem
    def _apply_grid_text(self, image, p1, p2, i):
        return image


    # Calcula a proporção de crescimento da imagem com zoom em relação a imagem original
    def _calculate_original_ratio(self):

        # Recupera a imagem com zoom
        zimage = self.image_original.copy()
        zimage = self._apply_zoom(zimage)

        # Recupera a imagem original
        oimage = self.image_original

        # Ajusta centro de acordo com o zoom aplicado
        xratio = oimage.shape[1] / zimage.shape[1]
        yratio = oimage.shape[0] / zimage.shape[0]

        return xratio, yratio


    # Calcula um ponto na imagem original baseado em um ponto dado da imagem com zoom
    def _calculate_original_point_position(self, zoom_point):

        xratio, yratio = self._calculate_original_ratio()

        xnew = round(zoom_point[0] * xratio)
        ynew = round(zoom_point[1] * yratio)

        original_point = (xnew, ynew)
        return original_point


    # Calcula a proporção de crescimento da imagem original em relação a imagem com zoom
    def _calculate_zoom_ratio(self):

        # Recupera a imagem com zoom
        zimage = self.image_original.copy()
        zimage = self._apply_zoom(zimage)

        # Recupera a imagem original
        oimage = self.image_original

        # Ajusta centro de acordo com o zoom aplicado
        xratio = zimage.shape[1] / oimage.shape[1]
        yratio = zimage.shape[0] / oimage.shape[0]

        return xratio, yratio


    # Calcula um ponto na imagem com zoom baseado em um ponto dado da imagem original
    def _calculate_zoom_point_position(self, original_point):

        xratio, yratio = self._calculate_zoom_ratio()

        xnew = round(original_point[0] * xratio)
        ynew = round(original_point[1] * yratio)

        zoom_point = (xnew, ynew)
        return zoom_point


    # Aplica zoom a imagem
    def _apply_zoom(self, image):

        for i in range(abs(self.zoom)):
            zoom_factor = 1.2 if self.zoom > 0 else 0.8
            image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)

        return image


    # Aumenta o zoom da imagem
    def zoom_in(self):
        self.zoom += 1


    # Diminui o zoom da imagem
    def zoom_out(self):
        self.zoom -= 1


    # Gera uma imagem
    def get_image(self):

        # Recupera uma cópia da imagem original
        image = self.image_original.copy()

        # Aplica grid
        if not self.hide_grid:
            image = self._apply_grid(image)

        # Aplica lattice
        if not self.hide_lattice:
            image = self._apply_lattice(image)

        return image


    # Retorna um vetor com os recortes
    def get_images_sliced(self, x_size = None, y_size = None):
        return []


    # Seleciona uma grid existente
    def select_grid(self, center):

        import math

        # Recalcula o centro
        center = self._calculate_original_point_position(center)

        dist = None
        best = -1

        # Recupera a grid
        grid = self.grid.reshape((self.grid.shape[0] * self.grid.shape[1], 2, 2))

        # Monta grid na imagem
        for n, (p1, p2) in enumerate(grid):

            grid_center = ((p2 - p1) // 2) + p1
            grid_center = (grid_center[0]+self.dx, grid_center[1]+self.dy)

            d = math.dist(center, grid_center)

            # Armazena indice que possuí a menor distância
            if dist == None:
                dist = d
                best = n
            else:
                if d < dist:
                    dist = d
                    best = n

        return best
