"""
Módulo para processamento de histograma
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from ..utils.image_utils import convert_to_grayscale


def calculate_histogram(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula o histograma de uma imagem
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Tupla com (histograma, bins)
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Calcula o histograma
    hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
    return hist, bins


def calculate_cumulative_histogram(hist: np.ndarray) -> np.ndarray:
    """
    Calcula o histograma cumulativo
    
    Args:
        hist: Histograma normal
        
    Returns:
        Histograma cumulativo
    """
    return np.cumsum(hist)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Equaliza o histograma de uma imagem
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Imagem com histograma equalizado
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Equaliza o histograma
    equalized = cv2.equalizeHist(gray_image)
    return equalized


def stretch_contrast(image: np.ndarray, min_val: int = 0, max_val: int = 255) -> np.ndarray:
    """
    Alarga o contraste de uma imagem
    
    Args:
        image: Imagem de entrada
        min_val: Valor mínimo desejado
        max_val: Valor máximo desejado
        
    Returns:
        Imagem com contraste alargado
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Encontra os valores mínimo e máximo atuais
    current_min = np.min(gray_image)
    current_max = np.max(gray_image)
    
    # Evita divisão por zero
    if current_max == current_min:
        return gray_image
        
    # Aplica a transformação linear
    stretched = ((gray_image - current_min) / (current_max - current_min) * 
                (max_val - min_val) + min_val).astype(np.uint8)
    
    return stretched


def normalize_histogram(image: np.ndarray, target_mean: float = 128, 
                       target_std: float = 64) -> np.ndarray:
    """
    Normaliza o histograma para uma média e desvio padrão específicos
    
    Args:
        image: Imagem de entrada
        target_mean: Média desejada
        target_std: Desvio padrão desejado
        
    Returns:
        Imagem com histograma normalizado
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Calcula estatísticas atuais
    current_mean = np.mean(gray_image)
    current_std = np.std(gray_image)
    
    # Evita divisão por zero
    if current_std == 0:
        return gray_image
        
    # Aplica normalização
    normalized = ((gray_image - current_mean) / current_std * target_std + target_mean)
    
    # Limita os valores ao intervalo [0, 255]
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def get_histogram_statistics(image: np.ndarray) -> dict:
    """
    Calcula estatísticas do histograma
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Dicionário com estatísticas
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    hist, _ = calculate_histogram(gray_image)
    
    # Calcula estatísticas
    total_pixels = np.sum(hist)
    mean_intensity = np.sum(np.arange(256) * hist) / total_pixels
    
    # Calcula variância
    variance = np.sum(((np.arange(256) - mean_intensity) ** 2) * hist) / total_pixels
    std_intensity = np.sqrt(variance)
    
    # Calcula percentis
    cumulative = calculate_cumulative_histogram(hist)
    p25 = np.argmax(cumulative >= 0.25 * total_pixels)
    p50 = np.argmax(cumulative >= 0.50 * total_pixels)
    p75 = np.argmax(cumulative >= 0.75 * total_pixels)
    
    return {
        'mean': float(mean_intensity),
        'std': float(std_intensity),
        'variance': float(variance),
        'min': float(np.min(gray_image)),
        'max': float(np.max(gray_image)),
        'p25': int(p25),
        'p50': int(p50),
        'p75': int(p75),
        'total_pixels': int(total_pixels)
    }


def create_histogram_plot_data(image: np.ndarray) -> dict:
    """
    Cria dados para plotagem do histograma
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Dicionário com dados para plotagem
    """
    hist, bins = calculate_histogram(image)
    cumulative = calculate_cumulative_histogram(hist)
    
    return {
        'histogram': hist,
        'bins': bins,
        'cumulative': cumulative,
        'x_values': np.arange(256),
        'statistics': get_histogram_statistics(image)
    }


def adaptive_histogram_equalization(image: np.ndarray, clip_limit: float = 2.0, 
                                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica equalização adaptativa de histograma (CLAHE)
    
    Args:
        image: Imagem de entrada
        clip_limit: Limite de clipping
        tile_grid_size: Tamanho da grade de tiles
        
    Returns:
        Imagem com CLAHE aplicado
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria o objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Aplica CLAHE
    equalized = clahe.apply(gray_image)
    
    return equalized 