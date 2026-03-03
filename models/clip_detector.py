"""
Detector CLIP UniversalFakeDetect baseado em 'openai/clip-vit-large-patch14'.

Usa o backbone CLIP ViT-L/14 congelado para extrair features visuais
universais, seguido de uma sonda linear treinavel para classificacao
real/IA. Abordagem inspirada em UniversalFakeDetect (Ojha et al. 2023),
que demonstrou generalizacao superior a geradores nao vistos no treino.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config.settings import DIRETORIO_MODELOS
from models.base import DetectorBase, ResultadoDeteccao
from utils.gpu_manager import gerenciador_gpu

logger = logging.getLogger(__name__)

_HUB_ID = "openai/clip-vit-large-patch14"
_DIM_EMBEDDING = 768  # Dimensao do embedding visual do CLIP ViT-L/14
_CAMINHO_CABECA = DIRETORIO_MODELOS / "clip_ufd" / "cabeca_clip_ufd.pth"


class _SondaLinear(nn.Module):
    """Sonda linear para classificacao binaria sobre embeddings CLIP."""

    def __init__(self, dim_entrada: int = _DIM_EMBEDDING, num_classes: int = 2) -> None:
        super().__init__()
        self.classificador = nn.Sequential(
            nn.LayerNorm(dim_entrada),
            nn.Linear(dim_entrada, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classificador(x)


class DetectorCLIP(DetectorBase):
    """
    Detector UniversalFakeDetect baseado em embeddings CLIP ViT-L/14.

    Utiliza o backbone CLIP pre-treinado congelado para extrair embeddings
    visuais ricos em semantica, seguido de uma sonda linear treinavel
    para distinguir imagens reais de geradas por IA.

    A generalizacao cross-domain e o principal diferencial: CLIP aprende
    representacoes universais que expoe inconsistencias em imagens
    sinteticas mesmo de geradores nao vistos durante o treinamento.

    Atributos:
        _processador: CLIPProcessor para pre-processamento compativel.
        _sonda: Sonda linear para classificacao.
        _dispositivo: Dispositivo PyTorch (cuda ou cpu) em uso.
        _sonda_treinada: Indica se a sonda foi carregada com pesos treinados.
    """

    def __init__(self) -> None:
        super().__init__(
            id_modelo="clip_detector",
            nome_modelo="CLIP UniversalFakeDetect (ViT-L/14)",
        )
        self._processador: Optional[CLIPProcessor] = None
        self._sonda: Optional[_SondaLinear] = None
        self._dispositivo: Optional[torch.device] = None
        self._sonda_treinada: bool = False

    def carregar(self, dispositivo: str = "cuda") -> None:
        """
        Carrega o backbone CLIP ViT-L/14 congelado + sonda linear.

        Args:
            dispositivo: Dispositivo PyTorch alvo ('cuda' ou 'cpu').

        Raises:
            RuntimeError: Se o download ou carregamento do modelo falhar.
        """
        logger.info(f"Carregando {self.nome_modelo} de '{_HUB_ID}'...")
        inicio = time.time()

        self._dispositivo = gerenciador_gpu.dispositivo

        try:
            self._processador = CLIPProcessor.from_pretrained(_HUB_ID)

            modelo_raw = CLIPModel.from_pretrained(
                _HUB_ID,
                torch_dtype=torch.float16,
            )

            # Congela backbone CLIP — apenas a sonda sera treinavel
            for param in modelo_raw.parameters():
                param.requires_grad = False

            self._modelo = gerenciador_gpu.mover_para_gpu(modelo_raw, fp16=True)

            # Sonda linear
            self._sonda = _SondaLinear(dim_entrada=_DIM_EMBEDDING, num_classes=2)
            self._carregar_sonda()
            self._sonda = self._sonda.half().to(self._dispositivo)
            self._sonda.eval()

            self._carregado = True

            tempo_carregamento = time.time() - inicio
            logger.info(
                f"{self.nome_modelo} carregado em {tempo_carregamento:.2f}s. "
                f"Sonda treinada: {self._sonda_treinada}. "
                f"Dispositivo: {self._dispositivo}. "
                f"VRAM usada: {gerenciador_gpu.obter_vram_usada():.0f}MB"
            )

        except Exception as excecao:
            logger.error(f"Falha ao carregar {self.nome_modelo}: {excecao}")
            self._carregado = False
            raise RuntimeError(
                f"Nao foi possivel carregar o modelo CLIP de '{_HUB_ID}': {excecao}"
            ) from excecao

    def _carregar_sonda(self) -> None:
        """Carrega pesos pre-treinados da sonda se disponiveis."""
        caminho = Path(_CAMINHO_CABECA)
        if caminho.exists():
            try:
                estado = torch.load(caminho, map_location="cpu", weights_only=True)
                self._sonda.load_state_dict(estado)
                self._sonda_treinada = True
                logger.info(f"Sonda CLIP UFD carregada de: {caminho}")
            except Exception as excecao:
                logger.warning(
                    f"Falha ao carregar sonda CLIP ({caminho}): {excecao}. "
                    "Usando pesos aleatorios (resultados nao confiaveis)."
                )
                self._sonda_treinada = False
        else:
            self._sonda_treinada = False
            logger.warning(
                f"Pesos da sonda CLIP nao encontrados em '{caminho}'. "
                "Usando pesos aleatorios ate treinamento ser realizado."
            )

    def descarregar(self) -> None:
        """Remove o modelo da GPU e libera toda VRAM associada."""
        if self._modelo is not None:
            del self._modelo
            self._modelo = None

        if self._sonda is not None:
            del self._sonda
            self._sonda = None

        if self._processador is not None:
            del self._processador
            self._processador = None

        self._sonda_treinada = False
        self._carregado = False
        gerenciador_gpu.limpar_vram()
        logger.info(f"{self.nome_modelo} descarregado da VRAM.")

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """
        Executa inferencia CLIP UniversalFakeDetect na imagem.

        Pipeline:
        1. Preprocessa imagem com CLIPProcessor
        2. Extrai embedding visual via get_image_features()
        3. Normaliza L2
        4. Classifica com sonda linear → softmax → score

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score 0.0 (real) a 1.0 (gerado por IA).
        """
        if not self._carregado or self._modelo is None or self._sonda is None:
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
            imagem_rgb = imagem.convert("RGB")

            entradas = self._processador(images=imagem_rgb, return_tensors="pt")
            pixel_values = entradas["pixel_values"].to(self._dispositivo).half()

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Extrai features visuais globais do CLIP
                    features_imagem = self._modelo.get_image_features(pixel_values=pixel_values)

                # Normaliza L2 (padrao UniversalFakeDetect)
                features_normalizadas = nn.functional.normalize(features_imagem, p=2, dim=-1)

                # Classifica com sonda linear
                logits = self._sonda(features_normalizadas)

            probabilidades = torch.softmax(logits.float(), dim=-1).squeeze()

            # Indice 1 = classe IA/Fake
            score_ia = float(probabilidades[1].item()) if len(probabilidades) > 1 else 0.5
            confianca = float(probabilidades.max().item())

            # Se sonda nao treinada, reduz confianca
            if not self._sonda_treinada:
                confianca *= 0.3

            tempo_ms = (time.time() - inicio) * 1000.0

            metadados = {
                "hub_id": _HUB_ID,
                "arquitetura": "CLIP ViT-L/14 + sonda linear (UniversalFakeDetect)",
                "sonda_treinada": self._sonda_treinada,
                "dim_embedding": _DIM_EMBEDDING,
                "dispositivo": str(self._dispositivo),
            }

            logger.debug(
                f"{self.nome_modelo}: score_ia={score_ia:.4f}, "
                f"confianca={confianca:.4f}, sonda_treinada={self._sonda_treinada}, "
                f"tempo={tempo_ms:.1f}ms"
            )

            return ResultadoDeteccao(
                score=score_ia,
                confianca=confianca,
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
