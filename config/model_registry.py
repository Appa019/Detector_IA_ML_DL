"""
Registro centralizado de modelos disponiveis para o ensemble.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RegistroModelo:
    """Informacoes de um modelo registrado."""
    id: str
    nome_exibicao: str
    hub_id: str  # ID no HuggingFace Hub ou identificador local
    arquitetura: str
    vram_fp16_mb: int
    papel: str
    tipo: str  # "imagem" ou "video" ou "ambos"
    habilitado: bool = True
    caminho_local: Optional[str] = None


# Registro global de modelos
REGISTRO_MODELOS: Dict[str, RegistroModelo] = {
    "spatial_vit": RegistroModelo(
        id="spatial_vit",
        nome_exibicao="ViT Espacial (Deep Fake Detector v2)",
        hub_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
        arquitetura="ViT-base-patch16-224",
        vram_fp16_mb=250,
        papel="Detector principal - analise espacial de texturas e artefatos",
        tipo="imagem",
    ),
    "clip_detector": RegistroModelo(
        id="clip_detector",
        nome_exibicao="CLIP UniversalFakeDetect (ViT-L/14)",
        hub_id="openai/clip-vit-large-patch14",
        arquitetura="CLIP ViT-L/14 + sonda linear",
        vram_fp16_mb=1700,
        papel="Generalizacao cross-domain universal - detecta geradores nao vistos no treino",
        tipo="imagem",
        habilitado=False,
    ),
    "frequency_analyzer": RegistroModelo(
        id="frequency_analyzer",
        nome_exibicao="Analisador de Frequencia (FFT/DCT)",
        hub_id="local",
        arquitetura="FFT/DCT + MLP",
        vram_fp16_mb=50,
        papel="Detecta artefatos de frequencia de GANs e modelos de difusao",
        tipo="imagem",
    ),
    "dinov2_detector": RegistroModelo(
        id="dinov2_detector",
        nome_exibicao="DINOv2-Small (Detector Universal)",
        hub_id="facebook/dinov2-small",
        arquitetura="DINOv2-Small + cabeca linear",
        vram_fp16_mb=350,
        papel="Detector universal de alta precisao - features auto-supervisionadas",
        tipo="imagem",
        habilitado=False,
    ),
    "sdxl_detector": RegistroModelo(
        id="sdxl_detector",
        nome_exibicao="SDXL Detector (Organika)",
        hub_id="Organika/sdxl-detector",
        arquitetura="Fine-tuned SDXL classifier",
        vram_fp16_mb=1740,
        papel="Detector especializado em imagens de modelos de difusao (SDXL)",
        tipo="imagem",
    ),
    "ai_image_detector": RegistroModelo(
        id="ai_image_detector",
        nome_exibicao="AI Image Detector (ViT)",
        hub_id="umm-maybe/AI-image-detector",
        arquitetura="ViT (Vision Transformer)",
        vram_fp16_mb=280,
        papel="Detector treinado em Midjourney/SD/DALL-E",
        tipo="imagem",
        habilitado=True,
    ),
    "siglip2_detector": RegistroModelo(
        id="siglip2_detector",
        nome_exibicao="SigLIP2 3-Classes",
        hub_id="prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
        arquitetura="SigLIP2 (google/siglip2-base-patch16-224)",
        vram_fp16_mb=180,
        papel="Classificador 3-classes: IA / Deepfake / Real",
        tipo="imagem",
        habilitado=False,
    ),
    "siglip_detector": RegistroModelo(
        id="siglip_detector",
        nome_exibicao="SigLIP AI vs Human",
        hub_id="Ateeqq/ai-vs-human-image-detector",
        arquitetura="SigLIP (SiglipForImageClassification)",
        vram_fp16_mb=180,
        papel="Detector binario IA vs Humano",
        tipo="imagem",
        habilitado=True,
    ),
    "efficientnet_video": RegistroModelo(
        id="efficientnet_video",
        nome_exibicao="EfficientNet-B4 (Video Deepfake)",
        hub_id="efficientnet-b4",
        arquitetura="EfficientNet-B4",
        vram_fp16_mb=150,
        papel="Deteccao de deepfakes em video (face swap, reenactment)",
        tipo="video",
    ),
}


def obter_modelos_por_tipo(tipo: str) -> Dict[str, RegistroModelo]:
    """Retorna modelos filtrados por tipo (imagem/video/ambos)."""
    return {
        k: v for k, v in REGISTRO_MODELOS.items()
        if v.habilitado and v.tipo in (tipo, "ambos")
    }


def obter_modelo(id_modelo: str) -> Optional[RegistroModelo]:
    """Retorna um modelo pelo ID."""
    return REGISTRO_MODELOS.get(id_modelo)


def calcular_vram_total() -> int:
    """Calcula VRAM total necessaria (pico sequencial, nao soma)."""
    return max(
        m.vram_fp16_mb for m in REGISTRO_MODELOS.values() if m.habilitado
    )
