"""
Detector SigLIP2 3-classes baseado em 'prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2'.

Classificador com 3 classes: AI / Deepfake / Real.
Score final = 1.0 - prob_real (combina prob_ai + prob_deepfake como "nao real").
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

_HUB_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"


class DetectorSigLIP2(DetectorBase):
    """
    Detector SigLIP2 com 3 classes: AI, Deepfake e Real.

    Utiliza google/siglip2-base-patch16-224 fine-tuned para classificar
    imagens em tres categorias. O score final combina as probabilidades
    de AI e Deepfake como indicadores de conteudo sintetico.

    Atributos:
        _processador: AutoImageProcessor para pre-processamento.
        _dispositivo: Dispositivo PyTorch em uso.
    """

    def __init__(self) -> None:
        super().__init__(
            id_modelo="siglip2_detector",
            nome_modelo="SigLIP2 3-Classes (IA/Deepfake/Real)",
        )
        self._processador: Optional[AutoImageProcessor] = None
        self._dispositivo: Optional[torch.device] = None

    def carregar(self, dispositivo: str = "cuda") -> None:
        """
        Carrega o processador e modelo SigLIP2 do HuggingFace Hub em FP16.

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
                f"Nao foi possivel carregar o modelo SigLIP2 de '{_HUB_ID}': {excecao}"
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
        Executa inferencia SigLIP2 3-classes na imagem.

        O score final e calculado como 1.0 - prob_real, combinando
        as probabilidades de AI e Deepfake como "nao real".

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score 0.0 (real) a 1.0 (gerado por IA).
            Metadados incluem probabilidades individuais de cada classe.
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

            # Identifica probabilidades por classe
            prob_ai, prob_deepfake, prob_real = self._extrair_probabilidades_3classes(
                probabilidades, mapa_rotulos
            )

            # Score final: 1.0 - prob_real (combina AI + Deepfake)
            score_ia = max(0.0, min(1.0, 1.0 - prob_real))
            confianca = float(probabilidades.max().item())

            tempo_ms = (time.time() - inicio) * 1000.0

            metadados = {
                "hub_id": _HUB_ID,
                "arquitetura": "SigLIP2 (google/siglip2-base-patch16-224)",
                "rotulos": {str(idx): rotulo for idx, rotulo in mapa_rotulos.items()},
                "probabilidades": {
                    rotulo: float(probabilidades[idx].item())
                    for idx, rotulo in mapa_rotulos.items()
                },
                "prob_ai": prob_ai,
                "prob_deepfake": prob_deepfake,
                "prob_real": prob_real,
                "dispositivo": str(self._dispositivo),
            }

            logger.debug(
                f"{self.nome_modelo}: score_ia={score_ia:.4f}, "
                f"prob_ai={prob_ai:.4f}, prob_deepfake={prob_deepfake:.4f}, "
                f"prob_real={prob_real:.4f}, tempo={tempo_ms:.1f}ms"
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

    def _extrair_probabilidades_3classes(
        self,
        probabilidades: torch.Tensor,
        mapa_rotulos: dict[int, str],
    ) -> tuple[float, float, float]:
        """
        Extrai probabilidades das 3 classes: AI, Deepfake e Real.

        Args:
            probabilidades: Tensor 1D com probabilidades de cada classe.
            mapa_rotulos: Dicionario {indice: nome_rotulo} do modelo.

        Returns:
            Tupla (prob_ai, prob_deepfake, prob_real).
        """
        _termos_ai = {"ai", "artificial", "generated", "synthetic"}
        _termos_deepfake = {"deepfake", "fake", "manipulated"}
        _termos_real = {"real", "human", "authentic", "natural", "genuine"}

        prob_ai = 0.0
        prob_deepfake = 0.0
        prob_real = 0.0

        for indice, rotulo in mapa_rotulos.items():
            rotulo_normalizado = rotulo.lower().replace("-", " ").replace("_", " ")
            prob = float(probabilidades[indice].item())

            if any(termo in rotulo_normalizado for termo in _termos_real):
                prob_real += prob
            elif any(termo in rotulo_normalizado for termo in _termos_deepfake):
                prob_deepfake += prob
            elif any(termo in rotulo_normalizado for termo in _termos_ai):
                prob_ai += prob
            else:
                # Rotulo nao reconhecido - trata como AI por seguranca
                logger.warning(f"Rotulo nao reconhecido: '{rotulo}'. Atribuido a AI.")
                prob_ai += prob

        # Se nenhuma classe foi identificada, usa fallback por posicao
        if prob_ai == 0.0 and prob_deepfake == 0.0 and prob_real == 0.0:
            logger.warning(
                f"Nenhum rotulo identificado em {mapa_rotulos}. "
                "Usando fallback posicional: [0]=AI, [1]=Deepfake, [2]=Real"
            )
            if len(probabilidades) >= 3:
                prob_ai = float(probabilidades[0].item())
                prob_deepfake = float(probabilidades[1].item())
                prob_real = float(probabilidades[2].item())
            elif len(probabilidades) == 2:
                prob_ai = float(probabilidades[0].item())
                prob_real = float(probabilidades[1].item())

        return prob_ai, prob_deepfake, prob_real
