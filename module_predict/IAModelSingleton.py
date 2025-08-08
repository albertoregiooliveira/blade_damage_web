########################################################################################################################
# IAModel.py
# Desenvolvido por: Alberto Régio A. de Oliveira
#
# Descrição:
#
########################################################################################################################

from module_predict.IAModel import IAModel

#=======================================================================================================================
# Classe responsável carregar o modelo de IA
class IAModelSingleton:

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = IAModel()
            cls._instance.load_model()
        return cls._instance
