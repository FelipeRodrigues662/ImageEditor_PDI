"""
Sistema de histórico para desfazer/refazer ações
"""
import numpy as np
from typing import List, Optional, Callable, Any


class HistoryManager:
    """Gerenciador de histórico para desfazer/refazer ações"""
    
    def __init__(self, max_history: int = 50):
        """
        Inicializa o gerenciador de histórico
        
        Args:
            max_history: Número máximo de ações no histórico
        """
        self.history: List[dict] = []
        self.current_index: int = -1
        self.max_history: int = max_history
        
    def add_action(self, action_name: str, image: np.ndarray, 
                   params: Optional[dict] = None, 
                   callback: Optional[Callable] = None):
        """
        Adiciona uma nova ação ao histórico
        
        Args:
            action_name: Nome da ação realizada
            image: Imagem resultante da ação
            params: Parâmetros usados na ação
            callback: Função de callback para refazer a ação
        """
        # Remove ações futuras se estamos no meio do histórico
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Adiciona nova ação
        action = {
            'name': action_name,
            'image': image.copy(),
            'params': params or {},
            'callback': callback
        }
        
        self.history.append(action)
        self.current_index += 1
        
        # Remove ações antigas se exceder o limite
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1
            
    def can_undo(self) -> bool:
        """Verifica se é possível desfazer"""
        return self.current_index > 0
        
    def can_redo(self) -> bool:
        """Verifica se é possível refazer"""
        return self.current_index < len(self.history) - 1
        
    def undo(self) -> Optional[dict]:
        """
        Desfaz a última ação
        
        Returns:
            Dicionário com informações da ação desfeita ou None
        """
        if not self.can_undo():
            return None
            
        self.current_index -= 1
        return self.history[self.current_index]
        
    def redo(self) -> Optional[dict]:
        """
        Refaz a próxima ação
        
        Returns:
            Dicionário com informações da ação refeita ou None
        """
        if not self.can_redo():
            return None
            
        self.current_index += 1
        return self.history[self.current_index]
        
    def get_current_image(self) -> Optional[np.ndarray]:
        """Retorna a imagem atual"""
        if self.current_index >= 0 and self.current_index < len(self.history):
            return self.history[self.current_index]['image'].copy()
        return None
        
    def get_current_action_name(self) -> Optional[str]:
        """Retorna o nome da ação atual"""
        if self.current_index >= 0 and self.current_index < len(self.history):
            return self.history[self.current_index]['name']
        return None
        
    def clear(self):
        """Limpa todo o histórico"""
        self.history.clear()
        self.current_index = -1
        
    def get_history_info(self) -> dict:
        """Retorna informações sobre o histórico"""
        return {
            'total_actions': len(self.history),
            'current_index': self.current_index,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo(),
            'current_action': self.get_current_action_name()
        } 