"""
Heart Segmentation Advanced - Utilities Module
=============================================

Este módulo contém utilitários para segmentação cardíaca avançada.

Módulos disponíveis:
- data_utils: Funções para carregamento e pré-processamento de dados
- model_utils: Funções para construção e manipulação de modelos
- training_utils: Funções para treinamento e callbacks
- visualization_utils: Funções para visualização e análise
"""

__version__ = "1.0.0"
__author__ = "Heart Segmentation Team"

# Imports principais
from . import data_utils
from . import model_utils
from . import training_utils
from . import visualization_utils

__all__ = [
    'data_utils',
    'model_utils', 
    'training_utils',
    'visualization_utils'
]
