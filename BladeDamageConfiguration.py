########################################################################################################################
# BladeDamageConfiguration.py
# Desenvolvido por: Alberto Régio A. de Oliveira
#
# Descrição:
#
########################################################################################################################

#=======================================================================================================================
# Classe responsável por controlar a exibicao do Segment Anything Meta (SAM)
class BladeDamageConfiguration:

    # Metodo inicial
    def __init__(self, folder_root):
        self.folder = folder_root
        self.folder_masks = folder_root
        self.folder_parameter = folder_root
        self.folder_parameter_sam = folder_root
        self.bb_x_size = 320
        self.bb_y_size = 320
        self.extract = None
        self.extract_sam = None
        self.extract_x_size = None
        self.extract_y_size = None
        self.overlap = 32 #10
        self.read_only = False
        self.force_size = False
        self.quiet = None

    @property
    def bb_size(self):
        return self.bb_x_size, self.bb_y_size
