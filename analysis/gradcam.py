"""
Geracao de mapas de ativacao (GradCAM) para modelos ViT.

Utiliza a biblioteca pytorch_grad_cam para produzir visualizacoes
que destacam as regioes da imagem mais influentes na decisao do modelo.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Tenta importar pytorch_grad_cam; disponibiliza fallback se ausente
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    _GRAD_CAM_DISPONIVEL = True
except ImportError:
    logger.warning(
        "pytorch_grad_cam nao instalado. GradCAM sera desabilitado. "
        "Instale com: pip install grad-cam"
    )
    _GRAD_CAM_DISPONIVEL = False


class GeradorGradCAM:
    """
    Gera mapas de ativacao GradCAM para modelos ViT (Vision Transformer).

    O GradCAM identifica quais regioes da imagem mais contribuiram para
    a predicao do modelo, auxiliando na explicabilidade da decisao de
    "imagem real" ou "gerada por IA".

    Para ViTs, a camada alvo tipica e a ultima LayerNorm antes do bloco
    de atencao final (e.g., model.vit.encoder.layer[-1].layernorm_before).
    """

    def __init__(
        self,
        modelo: nn.Module,
        camada_alvo: Optional[nn.Module] = None,
    ) -> None:
        """
        Inicializa o gerador GradCAM.

        Args:
            modelo: Modelo PyTorch (preferencialmente um ViT) ja carregado.
            camada_alvo: Modulo da rede que sera usado para extrair gradientes.
                         Se None, tenta inferir automaticamente para ViT.
        """
        self._modelo = modelo
        self._camada_alvo = camada_alvo or self._inferir_camada_alvo(modelo)
        self._grad_cam: Optional[object] = None  # instancia de GradCAM

        if _GRAD_CAM_DISPONIVEL and self._camada_alvo is not None:
            self._inicializar_grad_cam()
        else:
            logger.warning(
                "GradCAM nao sera inicializado (biblioteca ausente ou camada "
                "alvo nao encontrada)."
            )

    def gerar(
        self,
        imagem_tensor: torch.Tensor,
        indice_classe: Optional[int] = None,
    ) -> np.ndarray:
        """
        Gera o mapa de calor GradCAM para uma imagem.

        Args:
            imagem_tensor: Tensor de entrada com shape (1, 3, H, W) em
                           float32 ou float16, na mesma GPU do modelo.
            indice_classe: Indice da classe para calcular o gradiente.
                           Se None, usa a classe com maior logit.

        Returns:
            Mapa de calor normalizado entre 0.0 e 1.0, shape (H, W).
            Retorna array de zeros se GradCAM nao estiver disponivel.
        """
        altura = imagem_tensor.shape[2]
        largura = imagem_tensor.shape[3]
        mapa_zeros = np.zeros((altura, largura), dtype=np.float32)

        if not _GRAD_CAM_DISPONIVEL or self._grad_cam is None:
            logger.debug("GradCAM indisponivel; retornando mapa vazio.")
            return mapa_zeros

        try:
            # pytorch_grad_cam espera float32 na CPU ou GPU
            tensor_f32 = imagem_tensor.float()

            alvos = (
                [ClassifierOutputTarget(indice_classe)]
                if indice_classe is not None
                else None
            )

            # generate_cam retorna shape (N, H, W)
            mapa_calor = self._grad_cam(
                input_tensor=tensor_f32,
                targets=alvos,
            )
            # Pega o primeiro item do batch
            mapa_normalizado = mapa_calor[0]
            return mapa_normalizado.astype(np.float32)

        except Exception as erro:
            logger.error(
                "Erro ao gerar GradCAM: %s", erro, exc_info=True
            )
            return mapa_zeros

    def sobrepor_heatmap(
        self,
        imagem: np.ndarray,
        mapa_calor: np.ndarray,
        alfa: float = 0.4,
    ) -> np.ndarray:
        """
        Sobrepos o mapa de calor sobre a imagem original.

        Aplica o colormap JET ao mapa de calor e combina com a imagem
        original usando blend linear com fator alfa.

        Args:
            imagem: Imagem original em RGB, shape (H, W, 3), dtype uint8.
            mapa_calor: Mapa de calor normalizado 0.0-1.0, shape (H, W).
            alfa: Peso do mapa de calor na mistura (0 = so imagem,
                  1 = so mapa). Padrao: 0.4.

        Returns:
            Imagem RGB resultante da sobreposicao, shape (H, W, 3),
            dtype uint8.
        """
        if imagem.dtype != np.uint8:
            imagem = np.clip(imagem, 0, 255).astype(np.uint8)

        altura_orig, largura_orig = imagem.shape[:2]

        # Redimensiona mapa de calor para coincidir com a imagem
        if mapa_calor.shape != (altura_orig, largura_orig):
            mapa_calor = cv2.resize(
                mapa_calor,
                (largura_orig, altura_orig),
                interpolation=cv2.INTER_LINEAR,
            )

        # Converte mapa 0.0-1.0 para uint8 (0-255) e aplica colormap
        mapa_uint8 = np.uint8(255 * np.clip(mapa_calor, 0.0, 1.0))
        mapa_colorido_bgr = cv2.applyColorMap(mapa_uint8, cv2.COLORMAP_JET)

        # cv2 usa BGR; converte para RGB para manter consistencia
        mapa_colorido_rgb = cv2.cvtColor(mapa_colorido_bgr, cv2.COLOR_BGR2RGB)

        # Blend linear: imagem * (1 - alfa) + mapa * alfa
        imagem_float = imagem.astype(np.float32)
        mapa_float = mapa_colorido_rgb.astype(np.float32)
        resultado = cv2.addWeighted(
            imagem_float, 1.0 - alfa,
            mapa_float, alfa,
            0,
        )

        return np.clip(resultado, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Metodos auxiliares privados
    # ------------------------------------------------------------------

    def _inicializar_grad_cam(self) -> None:
        """Cria a instancia de GradCAM com a camada alvo configurada."""
        try:
            self._grad_cam = GradCAM(
                model=self._modelo,
                target_layers=[self._camada_alvo],
            )
            logger.info(
                "GradCAM inicializado com camada: %s",
                self._camada_alvo.__class__.__name__,
            )
        except Exception as erro:
            logger.error(
                "Falha ao inicializar GradCAM: %s", erro, exc_info=True
            )
            self._grad_cam = None

    @staticmethod
    def _inferir_camada_alvo(modelo: nn.Module) -> Optional[nn.Module]:
        """
        Tenta encontrar automaticamente a camada alvo para GradCAM em ViTs.

        Estrategia de busca (em ordem de prioridade):
        1. model.vit.encoder.layer[-1].layernorm_before  (HuggingFace ViT)
        2. Ultimo bloco do encoder que tenha 'norm' ou 'layernorm'
        3. None se nenhuma camada compativel for encontrada

        Args:
            modelo: Modelo PyTorch a ser inspecionado.

        Returns:
            Modulo encontrado ou None.
        """
        # Tentativa 1: ViT HuggingFace padrao
        try:
            camada = modelo.vit.encoder.layer[-1].layernorm_before
            logger.debug(
                "Camada alvo GradCAM inferida: vit.encoder.layer[-1].layernorm_before"
            )
            return camada
        except AttributeError:
            pass

        # Tentativa 2: Busca generica por 'norm' em blocos do encoder
        candidatos: list[nn.Module] = []
        for nome, modulo in modelo.named_modules():
            nome_lower = nome.lower()
            if any(
                kw in nome_lower
                for kw in ("layernorm", "norm", "ln_")
            ):
                candidatos.append(modulo)

        if candidatos:
            camada_selecionada = candidatos[-1]
            logger.debug(
                "Camada alvo GradCAM inferida por busca generica: %s",
                camada_selecionada.__class__.__name__,
            )
            return camada_selecionada

        logger.warning(
            "Nao foi possivel inferir a camada alvo para GradCAM. "
            "Passe camada_alvo explicitamente no construtor."
        )
        return None
