"""
Módulo para operações de morfologia matemática
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from scipy import ndimage
from ..utils.image_utils import convert_to_grayscale


def create_structuring_element(shape: str = 'rect', size: int = 3) -> np.ndarray:
    """
    Cria elemento estruturante
    
    Args:
        shape: Forma do elemento ('rect', 'ellipse', 'cross')
        size: Tamanho do elemento (deve ser ímpar)
        
    Returns:
        Elemento estruturante
    """
    if shape == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        raise ValueError(f"Forma não suportada: {shape}")


def erosion(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    """
    Aplica operação de erosão
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        kernel_shape: Forma do kernel
        
    Returns:
        Imagem erodida
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria elemento estruturante
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Aplica erosão
    eroded = cv2.erode(gray_image, kernel, iterations=1)
    
    return eroded


def dilation(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    """
    Aplica operação de dilatação
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        kernel_shape: Forma do kernel
        
    Returns:
        Imagem dilatada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria elemento estruturante
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Aplica dilatação
    dilated = cv2.dilate(gray_image, kernel, iterations=1)
    
    return dilated


def opening(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    """
    Aplica operação de abertura (erosão seguida de dilatação)
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        kernel_shape: Forma do kernel
        
    Returns:
        Imagem após abertura
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria elemento estruturante
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Aplica abertura
    opened = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    
    return opened


def closing(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    """
    Aplica operação de fechamento (dilatação seguida de erosão)
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        kernel_shape: Forma do kernel
        
    Returns:
        Imagem após fechamento
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria elemento estruturante
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Aplica fechamento
    closed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    return closed


def morphological_gradient(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    """
    Calcula gradiente morfológico (dilatação - erosão)
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        kernel_shape: Forma do kernel
        
    Returns:
        Gradiente morfológico
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria elemento estruturante
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Aplica gradiente morfológico
    gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
    
    return gradient


def top_hat(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    """
    Aplica operação top-hat (imagem - abertura)
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        kernel_shape: Forma do kernel
        
    Returns:
        Resultado do top-hat
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria elemento estruturante
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Aplica top-hat
    tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
    
    return tophat


def black_hat(image: np.ndarray, kernel_size: int = 3, kernel_shape: str = 'rect') -> np.ndarray:
    """
    Aplica operação black-hat (fechamento - imagem)
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        kernel_shape: Forma do kernel
        
    Returns:
        Resultado do black-hat
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria elemento estruturante
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Aplica black-hat
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    
    return blackhat


def hit_or_miss(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Aplica operação hit-or-miss
    
    Args:
        image: Imagem de entrada
        kernel_size: Tamanho do kernel
        
    Returns:
        Resultado da operação hit-or-miss
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Cria kernel hit-or-miss (exemplo simples)
    kernel = np.array([[-1, -1, -1],
                      [-1,  1, -1],
                      [-1, -1, -1]], dtype=np.int8)
    
    # Aplica hit-or-miss
    result = cv2.morphologyEx(gray_image, cv2.MORPH_HITMISS, kernel)
    
    return result


def skeletonize(image: np.ndarray) -> np.ndarray:
    """
    Esqueletiza uma imagem binária
    
    Args:
        image: Imagem binária de entrada
        
    Returns:
        Esqueleto da imagem
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Binariza a imagem se necessário
    if np.max(gray_image) > 1:
        _, binary = cv2.threshold(gray_image, 127, 1, cv2.THRESH_BINARY)
    else:
        binary = gray_image
        
    # Esqueletiza
    skeleton = ndimage.skeletonize(binary).astype(np.uint8) * 255
    
    return skeleton


def distance_transform(image: np.ndarray, distance_type: str = 'euclidean') -> np.ndarray:
    """
    Calcula transformada de distância
    
    Args:
        image: Imagem binária de entrada
        distance_type: Tipo de distância ('euclidean', 'manhattan', 'chessboard')
        
    Returns:
        Transformada de distância
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Binariza a imagem se necessário
    if np.max(gray_image) > 1:
        _, binary = cv2.threshold(gray_image, 127, 1, cv2.THRESH_BINARY)
    else:
        binary = gray_image
        
    # Inverte para que o fundo seja 0
    binary = 1 - binary
        
    # Calcula transformada de distância
    if distance_type == 'euclidean':
        distance = ndimage.distance_transform_edt(binary)
    elif distance_type == 'manhattan':
        distance = ndimage.distance_transform_cdt(binary, metric='taxicab')
    elif distance_type == 'chessboard':
        distance = ndimage.distance_transform_cdt(binary, metric='chessboard')
    else:
        raise ValueError(f"Tipo de distância não suportado: {distance_type}")
    
    # Normaliza para [0, 255]
    distance = (distance / np.max(distance) * 255).astype(np.uint8)
    
    return distance


def watershed_segmentation(image: np.ndarray, markers: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Aplica segmentação por watershed
    
    Args:
        image: Imagem de entrada
        markers: Marcadores para watershed (opcional)
        
    Returns:
        Imagem segmentada
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image
        
    # Se não há marcadores, cria alguns automaticamente
    if markers is None:
        # Aplica filtro gaussiano
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Calcula gradiente
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = (gradient / np.max(gradient) * 255).astype(np.uint8)
        
        # Cria marcadores simples
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operações morfológicas para criar marcadores
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marca componentes
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
    
    # Aplica watershed
    markers = markers.astype(np.int32)
    cv2.watershed(image, markers)
    
    # Normaliza resultado
    result = markers.astype(np.uint8)
    result = (result / np.max(result) * 255).astype(np.uint8)
    
    return result


def morphological_reconstruction(image: np.ndarray, mask: np.ndarray, 
                               operation: str = 'dilation') -> np.ndarray:
    """
    Aplica reconstrução morfológica
    
    Args:
        image: Imagem marcadora
        mask: Imagem máscara
        operation: Tipo de operação ('dilation', 'erosion')
        
    Returns:
        Imagem reconstruída
    """
    # Converte para níveis de cinza se necessário
    if len(image.shape) == 3:
        gray_image = convert_to_grayscale(image)
        mask_image = convert_to_grayscale(mask)
    else:
        gray_image = image
        mask_image = mask
        
    if operation == 'dilation':
        # Reconstrução por dilatação
        result = ndimage.grey_dilation(gray_image, size=(3, 3))
        result = np.minimum(result, mask_image)
        
        # Itera até convergência
        prev_result = None
        while not np.array_equal(result, prev_result):
            prev_result = result.copy()
            result = ndimage.grey_dilation(result, size=(3, 3))
            result = np.minimum(result, mask_image)
            
    elif operation == 'erosion':
        # Reconstrução por erosão
        result = ndimage.grey_erosion(gray_image, size=(3, 3))
        result = np.maximum(result, mask_image)
        
        # Itera até convergência
        prev_result = None
        while not np.array_equal(result, prev_result):
            prev_result = result.copy()
            result = ndimage.grey_erosion(result, size=(3, 3))
            result = np.maximum(result, mask_image)
    else:
        raise ValueError(f"Operação não suportada: {operation}")
    
    return result.astype(np.uint8) 