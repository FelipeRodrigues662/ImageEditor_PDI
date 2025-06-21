"""
Utilitárias para manipulação de imagens
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Carrega uma imagem do disco
    
    Args:
        file_path: Caminho para o arquivo de imagem
        
    Returns:
        Imagem carregada em formato numpy array ou None se erro
    """
    try:
        # Carrega a imagem
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            return None
            
        # Converte BGR para RGB se necessário
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return image
    except Exception as e:
        print(f"Erro ao carregar imagem: {e}")
        return None


def save_image(image: np.ndarray, file_path: str) -> bool:
    """
    Salva uma imagem no disco
    
    Args:
        image: Imagem em formato numpy array
        file_path: Caminho onde salvar a imagem
        
    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        # Converte RGB para BGR se necessário
        if len(image.shape) == 3:
            image_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_save = image
            
        # Salva a imagem
        success = cv2.imwrite(file_path, image_save)
        return success
    except Exception as e:
        print(f"Erro ao salvar imagem: {e}")
        return False


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem RGB para níveis de cinza
    
    Args:
        image: Imagem RGB
        
    Returns:
        Imagem em níveis de cinza
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Redimensiona uma imagem mantendo a proporção
    
    Args:
        image: Imagem original
        max_size: Tamanho máximo (largura ou altura)
        
    Returns:
        Imagem redimensionada
    """
    height, width = image.shape[:2]
    
    if width <= max_size and height <= max_size:
        return image
        
    # Calcula a nova dimensão mantendo proporção
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
        
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza uma imagem para o intervalo [0, 255]
    
    Args:
        image: Imagem original
        
    Returns:
        Imagem normalizada
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max > image_min:
            normalized = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
        else:
            normalized = image.astype(np.uint8)
    else:
        normalized = image.astype(np.uint8)
        
    return normalized


def get_image_info(image: np.ndarray) -> dict:
    """
    Retorna informações sobre a imagem
    
    Args:
        image: Imagem
        
    Returns:
        Dicionário com informações da imagem
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
        'std_value': float(np.std(image))
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
        info['type'] = 'RGB'
    else:
        info['channels'] = 1
        info['type'] = 'Grayscale'
        
    return info


def create_histogram(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula o histograma de uma imagem
    
    Args:
        image: Imagem em níveis de cinza
        
    Returns:
        Tupla com (valores do histograma, bins)
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
        
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return hist, bins


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Aplica uma máscara em uma imagem
    
    Args:
        image: Imagem original
        mask: Máscara binária
        
    Returns:
        Imagem com máscara aplicada
    """
    if len(image.shape) == 3:
        mask = np.stack([mask] * 3, axis=-1)
        
    return image * mask


def pad_image(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Adiciona padding à imagem para convolução
    
    Args:
        image: Imagem original
        kernel_size: Tamanho do kernel
        
    Returns:
        Imagem com padding
    """
    pad_size = kernel_size // 2
    return cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                             cv2.BORDER_REFLECT)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Recorta uma região da imagem
    
    Args:
        image: Imagem original
        x, y: Coordenadas do canto superior esquerdo
        width, height: Largura e altura da região
        
    Returns:
        Região recortada
    """
    return image[y:y+height, x:x+width]


def rotate_image(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Rotaciona uma imagem
    
    Args:
        image: Imagem original
        angle: Ângulo de rotação em graus
        center: Centro de rotação (opcional)
        
    Returns:
        Imagem rotacionada
    """
    height, width = image.shape[:2]
    
    if center is None:
        center = (width // 2, height // 2)
        
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def flip_image(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Inverte uma imagem
    
    Args:
        image: Imagem original
        direction: Direção da inversão ('horizontal' ou 'vertical')
        
    Returns:
        Imagem invertida
    """
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    elif direction == 'vertical':
        return cv2.flip(image, 0)
    else:
        return image 