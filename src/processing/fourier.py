"""
Módulo para processamento no domínio da frequência
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from ..utils.image_utils import convert_to_grayscale, normalize_image


def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Calcula a transformada de Fourier 2D
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Espectro de Fourier
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica FFT
    f_transform = fft2(gray_image.astype(np.float32))
    
    # Centraliza o espectro
    f_shift = fftshift(f_transform)
    
    return f_shift


def compute_inverse_fft(f_shift: np.ndarray) -> np.ndarray:
    """
    Calcula a transformada inversa de Fourier
    
    Args:
        f_shift: Espectro de Fourier centralizado
        
    Returns:
        Imagem reconstruída
    """
    # Descentraliza o espectro
    f_transform = ifftshift(f_shift)
    
    # Aplica IFFT
    image = ifft2(f_transform)
    
    # Retorna apenas a parte real
    return np.real(image)


def get_magnitude_spectrum(f_shift: np.ndarray) -> np.ndarray:
    """
    Calcula o espectro de magnitude
    
    Args:
        f_shift: Espectro de Fourier centralizado
        
    Returns:
        Espectro de magnitude
    """
    magnitude = np.abs(f_shift)
    
    # Aplica log para melhor visualização
    magnitude = np.log1p(magnitude)
    
    return magnitude


def get_phase_spectrum(f_shift: np.ndarray) -> np.ndarray:
    """
    Calcula o espectro de fase
    
    Args:
        f_shift: Espectro de Fourier centralizado
        
    Returns:
        Espectro de fase
    """
    phase = np.angle(f_shift)
    return phase


def create_ideal_lowpass_filter(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Cria filtro passa-baixa ideal
    
    Args:
        shape: Forma da imagem (height, width)
        cutoff: Frequência de corte (0-1)
        
    Returns:
        Filtro passa-baixa
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Cria grade de coordenadas
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Calcula distância do centro
    d = np.sqrt((u - ccol)**2 + (v - crow)**2)
    
    # Cria filtro ideal
    cutoff_freq = cutoff * min(rows, cols) / 2
    filter_mask = d <= cutoff_freq
    
    return filter_mask.astype(np.float32)


def create_ideal_highpass_filter(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Cria filtro passa-alta ideal
    
    Args:
        shape: Forma da imagem (height, width)
        cutoff: Frequência de corte (0-1)
        
    Returns:
        Filtro passa-alta
    """
    return 1.0 - create_ideal_lowpass_filter(shape, cutoff)


def create_butterworth_lowpass_filter(shape: Tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    """
    Cria filtro passa-baixa Butterworth
    
    Args:
        shape: Forma da imagem (height, width)
        cutoff: Frequência de corte (0-1)
        order: Ordem do filtro
        
    Returns:
        Filtro passa-baixa Butterworth
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Cria grade de coordenadas
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Calcula distância do centro
    d = np.sqrt((u - ccol)**2 + (v - crow)**2)
    
    # Cria filtro Butterworth
    cutoff_freq = cutoff * min(rows, cols) / 2
    filter_mask = 1.0 / (1.0 + (d / cutoff_freq)**(2 * order))
    
    return filter_mask.astype(np.float32)


def create_butterworth_highpass_filter(shape: Tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    """
    Cria filtro passa-alta Butterworth
    
    Args:
        shape: Forma da imagem (height, width)
        cutoff: Frequência de corte (0-1)
        order: Ordem do filtro
        
    Returns:
        Filtro passa-alta Butterworth
    """
    return 1.0 - create_butterworth_lowpass_filter(shape, cutoff, order)


def create_gaussian_lowpass_filter(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Cria filtro passa-baixa Gaussiano
    
    Args:
        shape: Forma da imagem (height, width)
        cutoff: Frequência de corte (0-1)
        
    Returns:
        Filtro passa-baixa Gaussiano
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Cria grade de coordenadas
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Calcula distância do centro
    d = np.sqrt((u - ccol)**2 + (v - crow)**2)
    
    # Cria filtro Gaussiano
    cutoff_freq = cutoff * min(rows, cols) / 2
    filter_mask = np.exp(-(d**2) / (2 * cutoff_freq**2))
    
    return filter_mask.astype(np.float32)


def create_gaussian_highpass_filter(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Cria filtro passa-alta Gaussiano
    
    Args:
        shape: Forma da imagem (height, width)
        cutoff: Frequência de corte (0-1)
        
    Returns:
        Filtro passa-alta Gaussiano
    """
    return 1.0 - create_gaussian_lowpass_filter(shape, cutoff)


def apply_frequency_filter(image: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    """
    Aplica filtro no domínio da frequência
    
    Args:
        image: Imagem de entrada
        filter_mask: Máscara do filtro
        
    Returns:
        Imagem filtrada
    """
    # Calcula FFT
    f_shift = compute_fft(image)
    
    # Aplica filtro
    filtered_shift = f_shift * filter_mask
    
    # Calcula IFFT
    filtered_image = compute_inverse_fft(filtered_shift)
    
    # Normaliza resultado
    filtered_image = normalize_image(filtered_image)
    
    return filtered_image


def lowpass_filter(image: np.ndarray, cutoff: float = 0.1, filter_type: str = 'gaussian') -> np.ndarray:
    """
    Aplica filtro passa-baixa
    
    Args:
        image: Imagem de entrada
        cutoff: Frequência de corte (0-1)
        filter_type: Tipo de filtro ('ideal', 'butterworth', 'gaussian')
        
    Returns:
        Imagem filtrada
    """
    shape = image.shape[:2]
    
    if filter_type == 'ideal':
        filter_mask = create_ideal_lowpass_filter(shape, cutoff)
    elif filter_type == 'butterworth':
        filter_mask = create_butterworth_lowpass_filter(shape, cutoff)
    elif filter_type == 'gaussian':
        filter_mask = create_gaussian_lowpass_filter(shape, cutoff)
    else:
        raise ValueError(f"Tipo de filtro não suportado: {filter_type}")
    
    return apply_frequency_filter(image, filter_mask)


def highpass_filter(image: np.ndarray, cutoff: float = 0.1, filter_type: str = 'gaussian') -> np.ndarray:
    """
    Aplica filtro passa-alta
    
    Args:
        image: Imagem de entrada
        cutoff: Frequência de corte (0-1)
        filter_type: Tipo de filtro ('ideal', 'butterworth', 'gaussian')
        
    Returns:
        Imagem filtrada
    """
    shape = image.shape[:2]
    
    if filter_type == 'ideal':
        filter_mask = create_ideal_highpass_filter(shape, cutoff)
    elif filter_type == 'butterworth':
        filter_mask = create_butterworth_highpass_filter(shape, cutoff)
    elif filter_type == 'gaussian':
        filter_mask = create_gaussian_highpass_filter(shape, cutoff)
    else:
        raise ValueError(f"Tipo de filtro não suportado: {filter_type}")
    
    return apply_frequency_filter(image, filter_mask)


def bandpass_filter(image: np.ndarray, low_cutoff: float = 0.05, high_cutoff: float = 0.2, 
                   filter_type: str = 'gaussian') -> np.ndarray:
    """
    Aplica filtro passa-banda
    
    Args:
        image: Imagem de entrada
        low_cutoff: Frequência de corte inferior (0-1)
        high_cutoff: Frequência de corte superior (0-1)
        filter_type: Tipo de filtro ('ideal', 'butterworth', 'gaussian')
        
    Returns:
        Imagem filtrada
    """
    # Aplica filtro passa-baixa
    low_filtered = lowpass_filter(image, high_cutoff, filter_type)
    
    # Aplica filtro passa-alta
    band_filtered = highpass_filter(low_filtered, low_cutoff, filter_type)
    
    return band_filtered


def notch_filter(image: np.ndarray, notch_points: list, notch_width: int = 10) -> np.ndarray:
    """
    Aplica filtro notch para remover ruído periódico
    
    Args:
        image: Imagem de entrada
        notch_points: Lista de pontos de notch [(u1, v1), (u2, v2), ...]
        notch_width: Largura do notch
        
    Returns:
        Imagem filtrada
    """
    shape = image.shape[:2]
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Cria máscara de notch
    filter_mask = np.ones(shape, dtype=np.float32)
    
    for u, v in notch_points:
        # Coordenadas no domínio da frequência
        freq_u = u + ccol
        freq_v = v + crow
        
        # Cria notch circular
        u_coords, v_coords = np.meshgrid(np.arange(cols), np.arange(rows))
        distance = np.sqrt((u_coords - freq_u)**2 + (v_coords - freq_v)**2)
        
        # Aplica notch
        notch = distance <= notch_width
        filter_mask[notch] = 0.0
        
        # Aplica notch simétrico
        sym_u = ccol - u
        sym_v = crow - v
        sym_distance = np.sqrt((u_coords - sym_u)**2 + (v_coords - sym_v)**2)
        sym_notch = sym_distance <= notch_width
        filter_mask[sym_notch] = 0.0
    
    return apply_frequency_filter(image, filter_mask)


def get_fourier_analysis(image: np.ndarray) -> dict:
    """
    Realiza análise completa no domínio da frequência
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Dicionário com análise de Fourier
    """
    # Calcula FFT
    f_shift = compute_fft(image)
    
    # Calcula espectros
    magnitude = get_magnitude_spectrum(f_shift)
    phase = get_phase_spectrum(f_shift)
    
    # Calcula estatísticas
    magnitude_stats = {
        'mean': float(np.mean(magnitude)),
        'std': float(np.std(magnitude)),
        'min': float(np.min(magnitude)),
        'max': float(np.max(magnitude))
    }
    
    return {
        'fft': f_shift,
        'magnitude': magnitude,
        'phase': phase,
        'magnitude_stats': magnitude_stats
    } 