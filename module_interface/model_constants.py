import os
import platform
import numpy as np

IMAGE_DIMENSIONS = [320, 320]

# Usado para os modelos novos, que foram treinados com as classes em ordem alfabetica
CLASSES = ['descarga', 'erosao', 'normal', 'pintura', 'trinca']
CLASSES_LABELS = ['Descarga', 'Erosao', 'Normal', 'Pintura', 'Trinca']

# Usado para os modelos antigos, que foram treinados com as classes em ordem diversa
# CLASSES = ['erosao', 'trinca', 'descarga', 'normal', 'pintura']
# CLASSES_LABELS = ['Erosao', 'Trinca', 'Descarga', 'Normal', 'Pintura']

# Quantidade de repeticoes executadas para cada treinamento de modelo
TRAINING_REPETITIONS = 5

# Random state padrao
RANDOM_STATE = 42

# Define a lista de random states utilizados
RANDOM_STATE_LIST = np.array([93, 692, 100, 562, 729])


# Retorna caminho da pasta notebooks
def get_PATH_NOTEBOOKS():
    if platform.system() == 'Windows':
        return 'D:\\Dropbox\\Projetos\\notebooks\\'
    elif platform.system() == 'Linux':
        return '/home/regio/Dropbox/Projetos/notebooks/'
    else:
        return '/Users/albertoregio/Dropbox/Projetos/notebooks/'


# Retorna caminho dos modelos
def get_PATH_MODEL():
    path = get_PATH_NOTEBOOKS()
    return os.path.join(path, 'blades_models')


# Retorna caminho das imagens
def get_PATH_IMAGES():
    path = get_PATH_NOTEBOOKS()
    return os.path.join(path, 'blades_images')