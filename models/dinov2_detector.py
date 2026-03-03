"""
Detector DINOv2-Small para deteccao de conteudo gerado por IA.

DINOv2 (Meta AI) aprende features visuais universais via auto-supervisao,
capturando padroes de textura e estrutura que distinguem imagens reais
de geradas por IA com alta precisao (99.01% em benchmarks, Nature 2025).

Backbone congelado + cabeca linear treinavel (sonda linear).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config.settings import DIRETORIO_MODELOS
from models.base import DetectorBase, ResultadoDeteccao
from utils.gpu_manager import gerenciador_gpu

logger = logging.getLogger(__name__)

_HUB_ID = "facebook/dinov2-small"
_DIM_EMBEDDING = 384  # Dimensao do CLS token do DINOv2-Small
_CAMINHO_CABECA = DIRETORIO_MODELOS / "dinov2" / "cabeca_dinov2.pth"


class _CabecaClassificacao(nn.Module):
    """Cabeca linear para classificacao binaria sobre embeddings DINOv2."""

    def __init__(self, dim_entrada: int = _DIM_EMBEDDING, num_classes: int = 2) -> None:
        super().__init__()
        self.classificador = nn.Sequential(
            nn.LayerNorm(dim_entrada),
            nn.Linear(dim_entrada, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classificador(x)


class DetectorDINOv2(DetectorBase):
    """
    Detector baseado em DINOv2-Small para analise de conteudo gerado por IA.

    Utiliza o backbone DINOv2-Small (86M params) congelado para extrair
    o CLS token como embedding global da imagem, seguido de uma cabeca
    linear treinavel para classificacao real/IA.

    Atributos:
        _processador: AutoImageProcessor para pre-processamento.
        _cabeca: Cabeca linear para classificacao.
        _dispositivo: Dispositivo PyTorch em uso.
        _cabeca_treinada: Indica se a cabeca foi carregada com pesos treinados.
    """

    def __init__(self) -> None:
        super().__init__(
            id_modelo="dinov2_detector",
            nome_modelo="DINOv2-Small (Detector Universal)",
        )
        self._processador: Optional[AutoImageProcessor] = None
        self._cabeca: Optional[_CabecaClassificacao] = None
        self._dispositivo: Optional[torch.device] = None
        self._cabeca_treinada: bool = False

    def carregar(self, dispositivo: str = "cuda") -> None:
        """
        Carrega o backbone DINOv2-Small congelado + cabeca linear.

        Args:
            dispositivo: Dispositivo PyTorch alvo ('cuda' ou 'cpu').

        Raises:
            RuntimeError: Se o download ou carregamento do modelo falhar.
        """
        logger.info(f"Carregando {self.nome_modelo} de '{_HUB_ID}'...")
        inicio = time.time()

        self._dispositivo = gerenciador_gpu.dispositivo

        try:
            self._processador = AutoImageProcessor.from_pretrained(_HUB_ID)

            modelo_raw = AutoModel.from_pretrained(
                _HUB_ID,
                torch_dtype=torch.float16,
            )

            # Congela backbone — apenas a cabeca sera treinavel
            for param in modelo_raw.parameters():
                param.requires_grad = False

            self._modelo = gerenciador_gpu.mover_para_gpu(modelo_raw, fp16=True)

            # Cabeca de classificacao
            self._cabeca = _CabecaClassificacao(dim_entrada=_DIM_EMBEDDING, num_classes=2)
            self._carregar_cabeca()
            self._cabeca = self._cabeca.half().to(self._dispositivo)
            self._cabeca.eval()

            self._carregado = True
            tempo_carregamento = time.time() - inicio
            logger.info(
                f"{self.nome_modelo} carregado em {tempo_carregamento:.2f}s. "
                f"Cabeca treinada: {self._cabeca_treinada}. "
                f"Dispositivo: {self._dispositivo}. "
                f"VRAM usada: {gerenciador_gpu.obter_vram_usada():.0f}MB"
            )

        except Exception as excecao:
            logger.error(f"Falha ao carregar {self.nome_modelo}: {excecao}")
            self._carregado = False
            raise RuntimeError(
                f"Nao foi possivel carregar o modelo DINOv2: {excecao}"
            ) from excecao

    def _carregar_cabeca(self) -> None:
        """Carrega pesos pre-treinados da cabeca se disponiveis."""
        caminho = Path(_CAMINHO_CABECA)
        if caminho.exists():
            try:
                estado = torch.load(caminho, map_location="cpu", weights_only=True)
                self._cabeca.load_state_dict(estado)
                self._cabeca_treinada = True
                logger.info(f"Cabeca DINOv2 carregada de: {caminho}")
            except Exception as excecao:
                logger.warning(
                    f"Falha ao carregar cabeca DINOv2 ({caminho}): {excecao}. "
                    "Usando pesos aleatorios (resultados nao confiaveis)."
                )
                self._cabeca_treinada = False
        else:
            self._cabeca_treinada = False
            logger.warning(
                f"Pesos da cabeca DINOv2 nao encontrados em '{caminho}'. "
                "Usando pesos aleatorios ate treinamento ser realizado."
            )

    def descarregar(self) -> None:
        """Remove o modelo da GPU e libera toda VRAM associada."""
        if self._modelo is not None:
            del self._modelo
            self._modelo = None

        if self._cabeca is not None:
            del self._cabeca
            self._cabeca = None

        if self._processador is not None:
            del self._processador
            self._processador = None

        self._cabeca_treinada = False
        self._carregado = False
        gerenciador_gpu.limpar_vram()
        logger.info(f"{self.nome_modelo} descarregado da VRAM.")

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """
        Executa inferencia DINOv2 na imagem e retorna score de IA.

        Pipeline:
        1. Preprocessa imagem para 224x224
        2. Extrai CLS token via backbone DINOv2 congelado
        3. Classifica com cabeca linear → softmax → score

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score 0.0 (real) a 1.0 (gerado por IA).
        """
        if not self._carregado or self._modelo is None or self._cabeca is None:
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
            entradas = {
                chave: tensor.to(self._dispositivo).half()
                for chave, tensor in entradas.items()
            }

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    saidas = self._modelo(**entradas)

                # Extrai CLS token (primeiro token do last_hidden_state)
                cls_token = saidas.last_hidden_state[:, 0, :]

                # Classifica com cabeca linear
                logits = self._cabeca(cls_token)

            probabilidades = torch.softmax(logits.float(), dim=-1).squeeze()

            # Indice 1 = classe IA/Fake
            score_ia = float(probabilidades[1].item()) if len(probabilidades) > 1 else 0.5
            confianca = float(probabilidades.max().item())

            # Se cabeca nao treinada, reduz confianca
            if not self._cabeca_treinada:
                confianca *= 0.3

            tempo_ms = (time.time() - inicio) * 1000.0

            metadados = {
                "hub_id": _HUB_ID,
                "arquitetura": "DINOv2-Small + cabeca linear",
                "cabeca_treinada": self._cabeca_treinada,
                "dim_embedding": _DIM_EMBEDDING,
                "dispositivo": str(self._dispositivo),
            }

            logger.debug(
                f"{self.nome_modelo}: score_ia={score_ia:.4f}, "
                f"confianca={confianca:.4f}, treinada={self._cabeca_treinada}, "
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
