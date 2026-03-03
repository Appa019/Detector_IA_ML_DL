"""
Detector SDXL baseado em 'Organika/sdxl-detector'.

Modelo fine-tuned especificamente para detectar imagens geradas por
Stable Diffusion XL (SDXL) e geradores similares de difusao.
Segue o mesmo padrao do DetectorViTEspacial (AutoModelForImageClassification).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from models.base import DetectorBase, ResultadoDeteccao
from utils.gpu_manager import gerenciador_gpu

logger = logging.getLogger(__name__)

_HUB_ID = "Organika/sdxl-detector"


class DetectorSDXL(DetectorBase):
    """
    Detector fine-tuned para imagens geradas por SDXL e modelos de difusao.

    Utiliza um classificador de imagem fine-tuned do HuggingFace Hub
    para distinguir imagens reais de geradas por Stable Diffusion XL.

    Atributos:
        _processador: AutoImageProcessor para pre-processamento.
        _dispositivo: Dispositivo PyTorch em uso.
    """

    def __init__(self) -> None:
        super().__init__(
            id_modelo="sdxl_detector",
            nome_modelo="SDXL Detector (Organika)",
        )
        self._processador: Optional[AutoImageProcessor] = None
        self._dispositivo: Optional[torch.device] = None

    def carregar(self, dispositivo: str = "cuda") -> None:
        """
        Carrega o processador e modelo SDXL Detector do HuggingFace Hub em FP16.

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

            modelo_raw = AutoModelForImageClassification.from_pretrained(
                _HUB_ID,
                torch_dtype=torch.float16,
                ignore_mismatched_sizes=True,
            )

            self._modelo = gerenciador_gpu.mover_para_gpu(modelo_raw, fp16=True)
            self._carregado = True

            tempo_carregamento = time.time() - inicio
            logger.info(
                f"{self.nome_modelo} carregado em {tempo_carregamento:.2f}s. "
                f"Dispositivo: {self._dispositivo}. "
                f"VRAM usada: {gerenciador_gpu.obter_vram_usada():.0f}MB"
            )

        except Exception as excecao:
            logger.error(f"Falha ao carregar {self.nome_modelo}: {excecao}")
            self._carregado = False
            raise RuntimeError(
                f"Nao foi possivel carregar o modelo SDXL de '{_HUB_ID}': {excecao}"
            ) from excecao

    def descarregar(self) -> None:
        """Remove o modelo da GPU e libera toda VRAM associada."""
        if self._modelo is not None:
            del self._modelo
            self._modelo = None

        if self._processador is not None:
            del self._processador
            self._processador = None

        self._carregado = False
        gerenciador_gpu.limpar_vram()
        logger.info(f"{self.nome_modelo} descarregado da VRAM.")

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """
        Executa inferencia SDXL Detector na imagem e retorna score de IA.

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score 0.0 (real) a 1.0 (gerado por IA).
        """
        if not self._carregado or self._modelo is None or self._processador is None:
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

            logits = saidas.logits.float()
            probabilidades = torch.softmax(logits, dim=-1).squeeze()

            mapa_rotulos = self._modelo.config.id2label
            score_ia = self._extrair_score_ia(probabilidades, mapa_rotulos)
            confianca = float(probabilidades.max().item())

            tempo_ms = (time.time() - inicio) * 1000.0

            metadados = {
                "hub_id": _HUB_ID,
                "arquitetura": "SDXL fine-tuned classifier",
                "rotulos": {str(idx): rotulo for idx, rotulo in mapa_rotulos.items()},
                "probabilidades": {
                    rotulo: float(probabilidades[idx].item())
                    for idx, rotulo in mapa_rotulos.items()
                },
                "dispositivo": str(self._dispositivo),
            }

            logger.debug(
                f"{self.nome_modelo}: score_ia={score_ia:.4f}, "
                f"confianca={confianca:.4f}, tempo={tempo_ms:.1f}ms"
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

    def _extrair_score_ia(
        self,
        probabilidades: torch.Tensor,
        mapa_rotulos: dict[int, str],
    ) -> float:
        """
        Extrai a probabilidade da classe de imagem gerada por IA.

        Busca por palavras-chave nos rotulos do modelo para identificar
        qual classe representa conteudo sintetico/falso/artificial.

        Args:
            probabilidades: Tensor 1D com probabilidades de cada classe.
            mapa_rotulos: Dicionario {indice: nome_rotulo} do modelo.

        Returns:
            Score entre 0.0 (real) e 1.0 (gerado por IA).
        """
        _termos_ia = {"fake", "ai", "synthetic", "generated", "artificial", "sdxl", "diffusion"}

        score_acumulado = 0.0

        for indice, rotulo in mapa_rotulos.items():
            rotulo_normalizado = rotulo.lower().replace("-", " ").replace("_", " ")
            if any(termo in rotulo_normalizado for termo in _termos_ia):
                score_acumulado += float(probabilidades[indice].item())

        if score_acumulado == 0.0:
            logger.warning(
                f"Rotulo de IA nao identificado em {mapa_rotulos}. "
                "Usando indice 1 como fallback."
            )
            score_acumulado = (
                float(probabilidades[1].item()) if len(probabilidades) > 1
                else float(probabilidades[0].item())
            )

        return max(0.0, min(1.0, score_acumulado))
