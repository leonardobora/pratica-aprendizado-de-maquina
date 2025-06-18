"""
Visualization Utilities for Heart Segmentation Advanced
======================================================

Este módulo contém funções para visualização de dados médicos,
resultados de treinamento e análises.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Union, Dict
import warnings
warnings.filterwarnings('ignore')

# Tentar imports opcionais
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class MedicalImageVisualizer:
    """Visualizador especializado para imagens médicas"""
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8'):
        """
        Inicializa visualizador
        
        Args:
            figsize: Tamanho padrão das figuras
            style: Estilo matplotlib
        """
        self.figsize = figsize
        plt.style.use('default')  # Usar estilo padrão
        sns.set_palette("husl")
    
    def plot_slice_comparison(self, 
                            image: np.ndarray,
                            mask: np.ndarray,
                            slice_idx: int,
                            title: str = "Comparação de Fatia",
                            class_names: List[str] = None,
                            class_colors: List[str] = None) -> None:
        """
        Plota comparação entre imagem e máscara
        
        Args:
            image: Volume de imagem 3D
            mask: Volume de máscara 3D
            slice_idx: Índice da fatia
            title: Título da figura
            class_names: Nomes das classes
            class_colors: Cores das classes
        """
        if class_names is None:
            class_names = ['Background', 'Left Ventricle', 'Myocardium']
        
        if class_colors is None:
            class_colors = ['black', 'red', 'green']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{title} - Fatia {slice_idx}', fontsize=14, fontweight='bold')
        
        # Imagem original
        axes[0].imshow(image[slice_idx], cmap='gray')
        axes[0].set_title('Imagem Original')
        axes[0].axis('off')
        
        # Máscara
        axes[1].imshow(mask[slice_idx], cmap='jet', vmin=0, vmax=len(class_names)-1)
        axes[1].set_title('Segmentação')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image[slice_idx], cmap='gray')
        
        # Criar overlay colorido
        overlay = np.zeros((*mask[slice_idx].shape, 3))
        for i, color in enumerate(class_colors[1:], 1):  # Pular background
            if color == 'red':
                overlay[mask[slice_idx] == i] = [1, 0, 0]
            elif color == 'green':
                overlay[mask[slice_idx] == i] = [0, 1, 0]
            elif color == 'blue':
                overlay[mask[slice_idx] == i] = [0, 0, 1]
        
        axes[2].imshow(overlay, alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_volume_montage(self,
                           volume: np.ndarray,
                           num_slices: int = 12,
                           title: str = "Montagem do Volume") -> None:
        """
        Cria montagem de fatias do volume
        
        Args:
            volume: Volume 3D
            num_slices: Número de fatias a mostrar
            title: Título da figura
        """
        depth = volume.shape[0]
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
        
        cols = 4
        rows = (num_slices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, slice_idx in enumerate(slice_indices):
            row = i // cols
            col = i % cols
            
            axes[row, col].imshow(volume[slice_idx], cmap='gray')
            axes[row, col].set_title(f'Fatia {slice_idx}')
            axes[row, col].axis('off')
        
        # Ocultar eixos não utilizados
        for i in range(num_slices, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self,
                               class_counts: Dict[int, int],
                               class_names: List[str] = None,
                               title: str = "Distribuição de Classes") -> None:
        """
        Plota distribuição de classes
        
        Args:
            class_counts: Dicionário {classe: contagem}
            class_names: Nomes das classes
            title: Título da figura
        """
        if class_names is None:
            class_names = ['Background', 'Left Ventricle', 'Myocardium']
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Gráfico de barras
        colors = ['gray', 'red', 'green'][:len(classes)]
        bars = ax1.bar(classes, counts, color=colors, alpha=0.7)
        ax1.set_xlabel('Classe')
        ax1.set_ylabel('Contagem de Pixels')
        ax1.set_title('Contagem por Classe')
        ax1.set_yscale('log')
        
        # Adicionar valores nas barras
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}',
                    ha='center', va='bottom')
        
        # Gráfico de pizza
        total = sum(counts)
        proportions = [count/total for count in counts]
        labels = [f'{class_names[i]}\n({prop*100:.1f}%)' 
                 for i, prop in zip(classes, proportions)]
        
        ax2.pie(proportions, labels=labels, colors=colors, autopct='',
               startangle=90)
        ax2.set_title('Proporção de Classes')
        
        plt.tight_layout()
        plt.show()

class TrainingVisualizer:
    """Visualizador para resultados de treinamento"""
    
    @staticmethod
    def plot_training_history(history: dict,
                            metrics: List[str] = None,
                            title: str = "Histórico de Treinamento") -> None:
        """
        Plota histórico de treinamento
        
        Args:
            history: Dicionário com histórico (history.history)
            metrics: Lista de métricas para plotar
            title: Título da figura
        """
        if metrics is None:
            # Detectar métricas automaticamente
            available_metrics = [key for key in history.keys() 
                               if not key.startswith('val_')]
            metrics = available_metrics
        
        num_metrics = len(metrics)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1) if num_metrics > 1 else [axes]
        elif num_metrics == 1:
            axes = [axes[0, 0]]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            train_values = history.get(metric, [])
            val_values = history.get(f'val_{metric}', [])
            
            epochs = range(1, len(train_values) + 1)
            
            axes[i].plot(epochs, train_values, 'b-', label=f'Treino {metric}')
            if val_values:
                axes[i].plot(epochs, val_values, 'r-', label=f'Validação {metric}')
            
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_xlabel('Época')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Ocultar eixos não utilizados
        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_learning_curves(train_scores: List[float],
                           val_scores: List[float],
                           metric_name: str = "Score",
                           title: str = "Curvas de Aprendizado") -> None:
        """
        Plota curvas de aprendizado
        
        Args:
            train_scores: Scores de treino
            val_scores: Scores de validação
            metric_name: Nome da métrica
            title: Título da figura
        """
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, 'b-', label=f'Treino {metric_name}', linewidth=2)
        plt.plot(epochs, val_scores, 'r-', label=f'Validação {metric_name}', linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações de máximo/mínimo
        max_val_idx = np.argmax(val_scores)
        plt.annotate(f'Melhor: {val_scores[max_val_idx]:.4f} (época {max_val_idx+1})',
                    xy=(max_val_idx+1, val_scores[max_val_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.show()

class InteractiveVisualizer:
    """Visualizador interativo usando Plotly (se disponível)"""
    
    def __init__(self):
        self.available = PLOTLY_AVAILABLE
        if not self.available:
            print("⚠️ Plotly não disponível - visualizações interativas desabilitadas")
    
    def plot_3d_volume(self,
                      volume: np.ndarray,
                      title: str = "Volume 3D",
                      sample_rate: int = 4) -> None:
        """
        Cria visualização 3D do volume
        
        Args:
            volume: Volume 3D
            title: Título da figura
            sample_rate: Taxa de amostragem para reduzir dados
        """
        if not self.available:
            print("⚠️ Plotly não disponível")
            return
        
        # Reduzir amostragem para performance
        sampled = volume[::sample_rate, ::sample_rate, ::sample_rate]
        
        # Criar coordenadas
        z, y, x = np.meshgrid(
            np.arange(sampled.shape[2]),
            np.arange(sampled.shape[1]),
            np.arange(sampled.shape[0])
        )
        
        # Filtrar valores significativos
        threshold = np.percentile(sampled, 90)
        mask = sampled > threshold
        
        fig = go.Figure(data=go.Scatter3d(
            x=x[mask],
            y=y[mask],
            z=z[mask],
            mode='markers',
            marker=dict(
                size=2,
                color=sampled[mask],
                colorscale='viridis',
                opacity=0.6
            )
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        fig.show()
    
    def plot_interactive_slice(self,
                             volume: np.ndarray,
                             mask: Optional[np.ndarray] = None,
                             title: str = "Fatia Interativa") -> None:
        """
        Cria visualização interativa de fatias
        
        Args:
            volume: Volume 3D
            mask: Máscara opcional
            title: Título da figura
        """
        if not self.available:
            print("⚠️ Plotly não disponível")
            return
        
        # Criar slider para navegar entre fatias
        num_slices = volume.shape[0]
        
        # Criar figura inicial
        fig = go.Figure()
        
        # Adicionar todas as fatias como traces
        for i in range(num_slices):
            fig.add_trace(
                go.Heatmap(
                    z=volume[i],
                    colorscale='gray',
                    visible=(i == num_slices // 2),  # Mostrar fatia do meio inicialmente
                    name=f'Fatia {i}'
                )
            )
        
        # Criar steps para slider
        steps = []
        for i in range(num_slices):
            step = dict(
                method="update",
                label=str(i),
                args=[{"visible": [False] * num_slices}]
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        
        # Adicionar slider
        sliders = [dict(
            active=num_slices // 2,
            currentvalue={"prefix": "Fatia: "},
            steps=steps
        )]
        
        fig.update_layout(
            sliders=sliders,
            title=title
        )
        
        fig.show()

def create_comparison_grid(images: List[np.ndarray],
                         titles: List[str],
                         suptitle: str = "Comparação de Imagens",
                         cmap: str = 'gray') -> None:
    """
    Cria grade de comparação de imagens
    
    Args:
        images: Lista de imagens 2D
        titles: Lista de títulos
        suptitle: Título principal
        cmap: Colormap
    """
    num_images = len(images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if num_images > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(title)
            axes[i].axis('off')
    
    # Ocultar eixos não utilizados
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
