"""
Detector EfficientNet-B4 para deepfakes em video.

Arquitetura EfficientNet-B4 pre-treinada no ImageNet com cabeca de classificacao
substituida por uma Linear(1792, 2) para o problema binario real/deepfake.

Projetado principalmente para frames de video (deepfakes faciais: face swap,
reenactment, puppeteering), mas tambem funciona para imagens estaticas.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B4_Weights

from config.settings import DIRETORIO_MODELOS
from models.base import DetectorBase, ResultadoDeteccao
from utils.gpu_manager import gerenciador_gpu

logger = logging.getLogger(__name__)

# Caminho dos pesos fine-tuned (opcional — usa ImageNet se nao existir)
_CAMINHO_PESOS_FINETUNED = (
    DIRETORIO_MODELOS / "efficientnet_b4" / "efficientnet_b4_deepfake.pth"
)

# Tamanho de entrada esperado pelo EfficientNet-B4
_TAMANHO_ENTRADA = 380

# Numero de features na ultima camada do EfficientNet-B4
_DIM_FEATURES_B4 = 1792

# Normalizacao ImageNet (usada durante pre-treino do EfficientNet)
_MEDIA_IMAGENET = (0.485, 0.456, 0.406)
_DESVIO_IMAGENET = (0.229, 0.224, 0.225)


def _construir_modelo_efficientnet(num_classes: int = 2) -> nn.Module:
    """
    Constroi EfficientNet-B4 com cabeca de classificacao personalizada.

    A backbone EfficientNet-B4 e carregada com pesos ImageNet; a cabeca
    original (nn.Linear(1792, 1000)) e substituida por nn.Linear(1792, 2)
    para classificacao binaria real/deepfake.

    Args:
        num_classes: Numero de classes de saida (padrao: 2 = real/deepfake).

    Returns:
        Modelo EfficientNet-B4 com cabeca substituida, pronto para fine-tuning.
    """
    # Pesos ImageNet mais recentes disponiveis
    pesos_imagenet = EfficientNet_B4_Weights.IMAGENET1K_V1
    modelo = models.efficientnet_b4(weights=pesos_imagenet)

    # Substitui cabeca classificadora original (1000 classes) por binaria
    # A ultima camada do EfficientNet-B4 e modelo.classifier[1]
    in_features = modelo.classifier[1].in_features  # 1792 para B4
    modelo.classifier[1] = nn.Linear(in_features, num_classes)

    logger.debug(
        f"EfficientNet-B4 construido: features={in_features}, "
        f"classes={num_classes}. Cabeca: {modelo.classifier[1]}"
    )

    return modelo


def _construir_transformacao_entrada() -> transforms.Compose:
    """
    Cria pipeline de transformacao para pre-processamento de imagens.

    Redimensiona para 380x380 (tamanho nativo do EfficientNet-B4),
    converte para tensor e normaliza com estatisticas ImageNet.

    Returns:
        Compose de transforms aplicado antes da inferencia.
    """
    return transforms.Compose([
        transforms.Resize((_TAMANHO_ENTRADA, _TAMANHO_ENTRADA), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEDIA_IMAGENET, std=_DESVIO_IMAGENET),
    ])


class DetectorEfficientNet(DetectorBase):
    """
    Detector de deepfakes baseado em EfficientNet-B4 com fine-tuning.

    Utiliza a backbone EfficientNet-B4 pre-treinada no ImageNet (1.2M imagens)
    com a cabeca classificadora substituida para deteccao binaria.

    Quando pesos fine-tuned estao disponiveis em `models_cache/efficientnet_b4/`,
    o modelo alcanca alta precisao em deepfakes faciais. Sem fine-tuning, os
    pesos ImageNet com cabeca aleatoria retornam score ~0.5 (incerto).

    Atributos:
        _transformacao: Pipeline de pre-processamento de imagens.
        _dispositivo: Dispositivo PyTorch (cuda ou cpu) em uso.
        _pesos_fine_tuned_carregados: Indica se pesos especializados foram carregados.
    """

    def __init__(self) -> None:
        super().__init__(
            id_modelo="efficientnet_video",
            nome_modelo="EfficientNet-B4 (Video Deepfake)",
        )
        self._transformacao: Optional[transforms.Compose] = None
        self._dispositivo: Optional[torch.device] = None
        self._pesos_fine_tuned_carregados: bool = False

    def carregar(self, dispositivo: str = "cuda") -> None:
        """
        Carrega EfficientNet-B4 com pesos ImageNet e cabeca de classificacao.

        Se pesos fine-tuned estiverem disponiveis em:
            `models_cache/efficientnet_b4/efficientnet_b4_deepfake.pth`
        eles sao carregados sobre a backbone. Caso contrario, o modelo usa
        apenas pesos ImageNet com cabeca nao treinada (score ~0.5 ate fine-tuning).

        Args:
            dispositivo: Dispositivo PyTorch alvo ('cuda' ou 'cpu').

        Raises:
            RuntimeError: Se a construcao do modelo falhar.
        """
        logger.info(f"Carregando {self.nome_modelo}...")
        inicio = time.time()

        self._dispositivo = gerenciador_gpu.dispositivo
        self._transformacao = _construir_transformacao_entrada()

        try:
            modelo_raw = _construir_modelo_efficientnet(num_classes=2)

            # Tenta carregar pesos fine-tuned (especializados para deepfakes)
            caminho_pesos = Path(_CAMINHO_PESOS_FINETUNED)
            if caminho_pesos.exists():
                try:
                    estado_salvo = torch.load(
                        caminho_pesos,
                        map_location="cpu",
                        weights_only=True,
                    )
                    modelo_raw.load_state_dict(estado_salvo)
                    self._pesos_fine_tuned_carregados = True
                    logger.info(f"Pesos fine-tuned carregados de: {caminho_pesos}")
                except Exception as excecao_pesos:
                    logger.warning(
                        f"Falha ao carregar pesos fine-tuned ({caminho_pesos}): "
                        f"{excecao_pesos}. "
                        "Usando backbone ImageNet com cabeca nao treinada."
                    )
                    self._pesos_fine_tuned_carregados = False
            else:
                self._pesos_fine_tuned_carregados = False
                logger.warning(
                    f"Pesos fine-tuned nao encontrados em '{caminho_pesos}'. "
                    "Usando backbone ImageNet. Score sera ~0.5 ate fine-tuning."
                )

            # Move para GPU em FP16
            self._modelo = gerenciador_gpu.mover_para_gpu(modelo_raw, fp16=True)
            self._carregado = True

            tempo_carregamento = time.time() - inicio
            status_pesos = "fine-tuned" if self._pesos_fine_tuned_carregados else "ImageNet (sem fine-tune)"
            logger.info(
                f"{self.nome_modelo} ({status_pesos}) carregado em {tempo_carregamento:.2f}s. "
                f"Dispositivo: {self._dispositivo}. "
                f"VRAM usada: {gerenciador_gpu.obter_vram_usada():.0f}MB"
            )

        except Exception as excecao:
            logger.error(f"Falha ao carregar {self.nome_modelo}: {excecao}")
            self._carregado = False
            raise RuntimeError(
                f"Nao foi possivel carregar o EfficientNet-B4: {excecao}"
            ) from excecao

    def descarregar(self) -> None:
        """Remove o modelo da GPU e libera toda VRAM associada."""
        if self._modelo is not None:
            del self._modelo
            self._modelo = None

        if self._transformacao is not None:
            self._transformacao = None

        self._pesos_fine_tuned_carregados = False
        self._carregado = False
        gerenciador_gpu.limpar_vram()
        logger.info(f"{self.nome_modelo} descarregado da VRAM.")

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """
        Executa inferencia EfficientNet-B4 e retorna score de deepfake.

        Pre-processamento:
        - Redimensiona para 380x380 (tamanho nativo do B4)
        - Normaliza com estatisticas ImageNet
        - Converte para FP16 na GPU

        Inferencia com torch.no_grad() + autocast FP16 para eficiencia maxima.

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score 0.0 (real) a 1.0 (deepfake/IA).
            Score ~0.5 indica incerteza (modelo sem fine-tuning).
        """
        if not self._carregado or self._modelo is None or self._transformacao is None:
            logger.error(f"{self.nome_modelo} nao esta carregado. Retornando score incerto.")
            return ResultadoDeteccao(
                score=0.5,
                confianca=0.0,
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados={"erro": "Modelo nao carregado"},
            )

        inicio = time.time()

        try:
            # Garante formato RGB antes do pre-processamento
            imagem_rgb = imagem.convert("RGB")

            # Aplica transformacoes: resize 380x380, ToTensor, Normalize
            tensor_entrada = self._transformacao(imagem_rgb)
            tensor_entrada = tensor_entrada.unsqueeze(0)  # Adiciona dimensao de batch: [1, C, H, W]
            tensor_entrada = tensor_entrada.to(self._dispositivo).half()

            # Inferencia com no_grad e autocast FP16
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = self._modelo(tensor_entrada)

            # Softmax para probabilidades calibradas
            logits_fp32 = logits.float()  # Converte para FP32 para softmax estavel
            probabilidades = torch.softmax(logits_fp32, dim=-1).squeeze()

            # Indice 1 = classe deepfake/IA (convencao: 0=real, 1=falso)
            score_ia = float(probabilidades[1].item()) if len(probabilidades) > 1 else 0.5
            confianca = float(probabilidades.max().item())

            # Confianca reduzida se pesos nao foram fine-tuned
            if not self._pesos_fine_tuned_carregados:
                confianca *= 0.5
                logger.debug(
                    f"{self.nome_modelo}: Confianca reduzida (sem fine-tuning). "
                    f"score_ia={score_ia:.4f}"
                )

            tempo_ms = (time.time() - inicio) * 1000.0

            metadados = {
                "arquitetura": "EfficientNet-B4",
                "tamanho_entrada": f"{_TAMANHO_ENTRADA}x{_TAMANHO_ENTRADA}",
                "features_backbone": _DIM_FEATURES_B4,
                "pesos_fine_tuned": self._pesos_fine_tuned_carregados,
                "caminho_pesos": str(_CAMINHO_PESOS_FINETUNED),
                "probabilidade_real": float(probabilidades[0].item()) if len(probabilidades) > 1 else 0.5,
                "probabilidade_fake": score_ia,
                "dispositivo": str(self._dispositivo),
            }

            logger.debug(
                f"{self.nome_modelo}: score_ia={score_ia:.4f}, "
                f"confianca={confianca:.4f}, tempo={tempo_ms:.1f}ms, "
                f"fine_tuned={self._pesos_fine_tuned_carregados}"
            )

            return ResultadoDeteccao(
                score=max(0.0, min(1.0, score_ia)),
                confianca=max(0.0, min(1.0, confianca)),
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados=metadados,
                tempo_inferencia_ms=tempo_ms,
            )

        except Exception as excecao:
            tempo_ms = (time.time() - inicio) * 1000.0
            logger.error(
                f"Erro durante inferencia em {self.nome_modelo}: {excecao}", exc_info=True
            )
            return ResultadoDeteccao(
                score=0.5,
                confianca=0.0,
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados={"erro": str(excecao)},
                tempo_inferencia_ms=tempo_ms,
            )
