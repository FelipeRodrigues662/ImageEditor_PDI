"""
Módulo para filtros espaciais
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from scipy import ndimage
from ..utils.image_utils import convert_to_grayscale, pad_image


def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica convolução 2D em uma imagem
    
    Args:
        image: Imagem de entrada
        kernel: Kernel de convolução
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica convolução
    filtered = cv2.filter2D(gray_image, -1, kernel)
    return filtered


# Filtros Passa-Baixa

def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Aplica filtro de média
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel (deve ser ímpar)
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro de média
    filtered = cv2.blur(gray_image, (kernel_size, kernel_size))
    return filtered


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Aplica filtro de mediana
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel (deve ser ímpar)
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro de mediana
    filtered = cv2.medianBlur(gray_image, kernel_size)
    return filtered


def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Aplica filtro gaussiano
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel (deve ser ímpar)
        sigma: Desvio padrão do gaussiano
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro gaussiano
    filtered = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)
    return filtered


def max_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Aplica filtro de máximo
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel (deve ser ímpar)
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro de máximo
    filtered = ndimage.maximum_filter(gray_image, size=kernel_size)
    return filtered


def min_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Aplica filtro de mínimo
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel (deve ser ímpar)
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro de mínimo
    filtered = ndimage.minimum_filter(gray_image, size=kernel_size)
    return filtered


# Filtros Passa-Alta

def laplacian_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Aplica filtro laplaciano
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel (deve ser ímpar)
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro laplaciano
    filtered = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=kernel_size)
    
    # Normaliza para [0, 255]
    filtered = np.absolute(filtered)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    
    return filtered


def roberts_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica filtro de Roberts
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Kernels de Roberts
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # Aplica convoluções
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    
    # Calcula magnitude do gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normaliza para [0, 255]
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude


def prewitt_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica filtro de Prewitt
    
    Args:
        image: Imagem de entrada
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Kernels de Prewitt
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    # Aplica convoluções
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    
    # Calcula magnitude do gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normaliza para [0, 255]
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude


def sobel_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Aplica filtro de Sobel
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel (deve ser ímpar)
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro de Sobel
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    # Calcula magnitude do gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normaliza para [0, 255]
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude


def unsharp_masking(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, 
                   amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """
    Aplica unsharp masking para realce de bordas
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel gaussiano
        sigma: Desvio padrão do gaussiano
        amount: Intensidade do realce
        threshold: Limiar para aplicar o realce
        
    Returns:
        Imagem com realce aplicado
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro gaussiano
    blurred = gaussian_filter(gray_image, kernel_size, sigma)
    
    # Calcula a diferença
    sharpened = cv2.addWeighted(gray_image, 1.0 + amount, blurred, -amount, 0)
    
    # Aplica limiar se especificado
    if threshold > 0:
        mask = np.abs(gray_image - blurred) > threshold
        sharpened = np.where(mask, sharpened, gray_image)
    
    # Normaliza para [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                    sigma_space: float = 75) -> np.ndarray:
    """
    Aplica filtro bilateral para preservar bordas
    
    Args:
        image: Imagem de entrada
        d: Diâmetro do kernel
        sigma_color: Desvio padrão do filtro de cor
        sigma_space: Desvio padrão do filtro espacial
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro bilateral
    filtered = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)
    return filtered


def wiener_filter(image: np.ndarray, kernel_size: int = 3, noise_var: float = 0.01) -> np.ndarray:
    """
    Aplica filtro de Wiener para redução de ruído
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        noise_var: Variância do ruído
        
    Returns:
        Imagem filtrada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Aplica filtro de Wiener
    filtered = ndimage.wiener(gray_image, (kernel_size, kernel_size), noise_var)
    
    # Normaliza para [0, 255]
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    
    return filtered 