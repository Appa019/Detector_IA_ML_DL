"""
Classe abstrata base para todos os detectores do ensemble.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image


@dataclass
class ResultadoDeteccao:
    """Resultado padronizado de um detector."""
    score: float  # 0.0 (real) a 1.0 (IA)
    confianca: float  # confianca do modelo na predicao
    id_modelo: str
    nome_modelo: str
    metadados: Dict[str, Any] = field(default_factory=dict)
    mapa_calor: Optional[np.ndarray] = None  # GradCAM ou similar
    tempo_inferencia_ms: float = 0.0


class DetectorBase(ABC):
    """
    Classe abstrata que todos os detectores do ensemble devem implementar.

    Cada detector e responsavel por:
    1. Carregar seu modelo na GPU (com FP16)
    2. Preprocessar a entrada
    3. Executar inferencia
    4. Retornar um ResultadoDeteccao padronizado
    5. Liberar recursos da GPU
    """

    def __init__(self, id_modelo: str, nome_modelo: str):
        self.id_modelo = id_modelo
        self.nome_modelo = nome_modelo
        self._modelo = None
        self._carregado = False

    @property
    def carregado(self) -> bool:
        return self._carregado

    @abstractmethod
    def carregar(self, dispositivo: str = "cuda") -> None:
        """Carrega o modelo na memoria/GPU."""
        ...

    @abstractmethod
    def descarregar(self) -> None:
        """Remove o modelo da memoria/GPU."""
        ...

    @abstractmethod
    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """
        Executa deteccao em uma imagem.

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score entre 0.0 (real) e 1.0 (IA).
        """
        ...

    def treinar(self, dados_treino: Any, dados_validacao: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Metodo para fine-tuning futuro com datasets proprios.
        Implementacao padrao levanta NotImplementedError.

        Returns:
            Dicionario com metricas de treino.
        """
        raise NotImplementedError(
            f"Fine-tuning ainda nao implementado para {self.nome_modelo}. "
            f"Preparado para implementacao futura."
        )

    def __repr__(self) -> str:
        status = "carregado" if self._carregado else "nao carregado"
        return f"{self.__class__.__name__}(id='{self.id_modelo}', status='{status}')"
