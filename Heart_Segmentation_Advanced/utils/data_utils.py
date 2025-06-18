"""
Data Utilities for Heart Segmentation Advanced
=============================================

Este módulo contém funções utilitárias para carregamento, 
pré-processamento e manipulação de dados médicos.
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os
from typing import Tuple, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MedicalImageLoader:
    """Carregador de imagens médicas com suporte a múltiplos formatos"""
    
    @staticmethod
    def load_nifti(filepath: str) -> Optional[np.ndarray]:
        """
        Carrega arquivo NIfTI e retorna array numpy
        
        Args:
            filepath: Caminho para arquivo .nii ou .nii.gz
            
        Returns:
            Array numpy com dados da imagem ou None se erro
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            return data
        except Exception as e:
            logger.error(f"Erro ao carregar {filepath}: {e}")
            return None
    
    @staticmethod
    def load_nifti_with_header(filepath: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Carrega arquivo NIfTI com informações do header
        
        Args:
            filepath: Caminho para arquivo .nii ou .nii.gz
            
        Returns:
            Tupla (dados, informações_header)
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            
            header_info = {
                'shape': data.shape,
                'pixdim': img.header['pixdim'][1:4],
                'orientation': str(nib.aff2axcodes(img.affine)),
                'dtype': str(data.dtype),
                'affine': img.affine
            }
            
            return data, header_info
        except Exception as e:
            logger.error(f"Erro ao carregar {filepath}: {e}")
            return None, None

class ImageNormalizer:
    """Normalizadores para imagens médicas"""
    
    @staticmethod
    def percentile_normalize(image: np.ndarray, 
                           lower_percentile: float = 2.0,
                           upper_percentile: float = 98.0) -> np.ndarray:
        """
        Normalização baseada em percentis
        
        Args:
            image: Imagem de entrada
            lower_percentile: Percentil inferior
            upper_percentile: Percentil superior
            
        Returns:
            Imagem normalizada [0, 1]
        """
        p_low = np.percentile(image, lower_percentile)
        p_high = np.percentile(image, upper_percentile)
        
        normalized = np.clip((image - p_low) / (p_high - p_low + 1e-8), 0, 1)
        return normalized.astype(np.float32)
    
    @staticmethod
    def z_score_normalize(image: np.ndarray, 
                         clip_range: Tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
        """
        Normalização Z-score com clipping
        
        Args:
            image: Imagem de entrada
            clip_range: Range para clipping
            
        Returns:
            Imagem normalizada [0, 1]
        """
        mean = np.mean(image)
        std = np.std(image) + 1e-8
        
        z_normalized = (image - mean) / std
        clipped = np.clip(z_normalized, clip_range[0], clip_range[1])
        
        # Normalizar para [0, 1]
        min_val, max_val = clip_range
        normalized = (clipped - min_val) / (max_val - min_val)
        
        return normalized.astype(np.float32)
    
    @staticmethod
    def min_max_normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalização Min-Max
        
        Args:
            image: Imagem de entrada
            
        Returns:
            Imagem normalizada [0, 1]
        """
        min_val = np.min(image)
        max_val = np.max(image)
        
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        return normalized.astype(np.float32)

class DatasetAnalyzer:
    """Analisador de propriedades do dataset"""
    
    @staticmethod
    def analyze_intensity_distribution(images: List[np.ndarray]) -> dict:
        """
        Analisa distribuição de intensidades
        
        Args:
            images: Lista de imagens
            
        Returns:
            Dicionário com estatísticas
        """
        all_values = np.concatenate([img.flatten() for img in images])
        
        stats = {
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'median': float(np.median(all_values)),
            'percentiles': {
                'p1': float(np.percentile(all_values, 1)),
                'p5': float(np.percentile(all_values, 5)),
                'p25': float(np.percentile(all_values, 25)),
                'p75': float(np.percentile(all_values, 75)),
                'p95': float(np.percentile(all_values, 95)),
                'p99': float(np.percentile(all_values, 99))
            }
        }
        
        return stats
    
    @staticmethod
    def analyze_class_distribution(labels: List[np.ndarray]) -> dict:
        """
        Analisa distribuição de classes
        
        Args:
            labels: Lista de máscaras de segmentação
            
        Returns:
            Dicionário com contagens e proporções
        """
        all_labels = np.concatenate([label.flatten() for label in labels])
        unique_classes, counts = np.unique(all_labels, return_counts=True)
        
        total_pixels = len(all_labels)
        
        distribution = {}
        for class_id, count in zip(unique_classes, counts):
            distribution[int(class_id)] = {
                'count': int(count),
                'proportion': float(count / total_pixels),
                'percentage': float(count / total_pixels * 100)
            }
        
        return distribution
    
    @staticmethod
    def analyze_spatial_properties(images: List[np.ndarray]) -> dict:
        """
        Analisa propriedades espaciais
        
        Args:
            images: Lista de imagens
            
        Returns:
            Dicionário com propriedades espaciais
        """
        shapes = [img.shape for img in images]
        
        # Analisar dimensões
        if len(shapes) > 0:
            dimensions = len(shapes[0])
            
            spatial_stats = {
                'dimensions': dimensions,
                'shape_variations': len(set(shapes)),
                'unique_shapes': list(set(shapes))
            }
            
            # Estatísticas por dimensão
            for dim in range(dimensions):
                dim_sizes = [shape[dim] for shape in shapes]
                spatial_stats[f'dim_{dim}'] = {
                    'min': min(dim_sizes),
                    'max': max(dim_sizes),
                    'mean': np.mean(dim_sizes),
                    'std': np.std(dim_sizes)
                }
        else:
            spatial_stats = {'error': 'No images provided'}
        
        return spatial_stats

def find_corresponding_files(images_dir: str, labels_dir: str, 
                           image_suffix: str = '_0000.nii.gz',
                           label_suffix: str = '.nii.gz') -> List[Tuple[str, str]]:
    """
    Encontra pares correspondentes de imagem e label
    
    Args:
        images_dir: Diretório de imagens
        labels_dir: Diretório de labels
        image_suffix: Sufixo dos arquivos de imagem
        label_suffix: Sufixo dos arquivos de label
        
    Returns:
        Lista de tuplas (caminho_imagem, caminho_label)
    """
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        logger.error("Diretórios não encontrados")
        return []
    
    # Listar arquivos de imagem
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]
    
    # Extrair IDs
    image_ids = []
    for f in image_files:
        if image_suffix in f:
            image_id = f.replace(image_suffix, '')
            image_ids.append(image_id)
    
    label_ids = [f.replace(label_suffix, '') for f in label_files]
    
    # Encontrar correspondências
    matched_pairs = []
    for image_id in image_ids:
        if image_id in label_ids:
            img_path = os.path.join(images_dir, f"{image_id}{image_suffix}")
            label_path = os.path.join(labels_dir, f"{image_id}{label_suffix}")
            matched_pairs.append((img_path, label_path))
    
    logger.info(f"Encontrados {len(matched_pairs)} pares correspondentes")
    return matched_pairs

def validate_data_integrity(image_path: str, label_path: str) -> bool:
    """
    Valida integridade de um par imagem-label
    
    Args:
        image_path: Caminho da imagem
        label_path: Caminho do label
        
    Returns:
        True se válido, False caso contrário
    """
    try:
        # Carregar ambos os arquivos
        img_data, img_header = MedicalImageLoader.load_nifti_with_header(image_path)
        label_data, label_header = MedicalImageLoader.load_nifti_with_header(label_path)
        
        if img_data is None or label_data is None:
            return False
        
        # Verificar compatibilidade de dimensões
        if img_data.shape != label_data.shape:
            logger.warning(f"Dimensões incompatíveis: {img_data.shape} vs {label_data.shape}")
            return False
        
        # Verificar se labels têm valores válidos
        unique_labels = np.unique(label_data)
        expected_labels = set([0, 1, 2])  # Background, LV, Myocardium
        
        if not set(unique_labels).issubset(expected_labels):
            logger.warning(f"Labels inesperados encontrados: {unique_labels}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Erro na validação: {e}")
        return False
