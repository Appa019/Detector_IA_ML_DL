"""
Configuracoes globais do sistema de deteccao de conteudo gerado por IA.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict


# Diretorio raiz do projeto
DIRETORIO_RAIZ = Path(__file__).parent.parent

# Diretorio de cache dos modelos
DIRETORIO_MODELOS = DIRETORIO_RAIZ / "models_cache"

# Precisao padrao (FP16 para economizar VRAM)
PRECISAO_PADRAO = "fp16"

# Dispositivo padrao
DISPOSITIVO_PADRAO = "cuda"


@dataclass
class ConfiguracaoModelo:
    """Configuracao individual de um modelo do ensemble."""
    nome: str
    peso_ensemble: float
    vram_estimada_mb: int
    velocidade_fps: int
    habilitado: bool = True


@dataclass
class ConfiguracaoEnsemble:
    """Configuracao do sistema de ensemble."""

    # Pesos dos modelos no ensemble (devem somar 1.0)
    # Fallback quando meta-learner nao esta treinado
    pesos: Dict[str, float] = field(default_factory=lambda: {
        "spatial_vit": 0.18,
        "sdxl_detector": 0.18,
        "ai_image_detector": 0.20,
        "siglip_detector": 0.18,
        "frequency_analyzer": 0.10,
        "pixel_stats": 0.04,
        "ela_score": 0.06,
        "wavelet_score": 0.06,
    })

    # Temperatura para calibracao (Temperature Scaling)
    temperatura: float = 1.5

    # Thresholds de classificacao
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "provavelmente_real": 0.25,
        "possivelmente_real": 0.50,
        "possivelmente_ia": 0.75,
        "provavelmente_ia": 1.00,
    })


@dataclass
class ConfiguracaoVideo:
    """Configuracao para processamento de video."""
    # Intervalo de amostragem de frames (a cada N frames)
    intervalo_frames: int = 10
    # Maximo de frames a processar
    max_frames: int = 100
    # Tamanho minimo de rosto para deteccao (pixels)
    tamanho_minimo_rosto: int = 40
    # Confianca minima para deteccao facial
    confianca_minima_rosto: float = 0.9


@dataclass
class ConfiguracaoImagem:
    """Configuracao para processamento de imagens."""
    # Tamanho de entrada padrao para modelos
    tamanho_entrada: int = 224
    # Normalizacao ImageNet
    media_normalizacao: tuple = (0.485, 0.456, 0.406)
    desvio_normalizacao: tuple = (0.229, 0.224, 0.225)


@dataclass
class ConfiguracaoApp:
    """Configuracao geral da aplicacao."""
    titulo: str = "Detector de Conteudo Gerado por IA"
    descricao: str = "Analise de imagens e videos usando ensemble de modelos de Deep Learning"
    max_tamanho_upload_mb: int = 200
    extensoes_imagem: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
    extensoes_video: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")


# Instancias globais de configuracao
CONFIG_ENSEMBLE = ConfiguracaoEnsemble()
CONFIG_VIDEO = ConfiguracaoVideo()
CONFIG_IMAGEM = ConfiguracaoImagem()
CONFIG_APP = ConfiguracaoApp()
