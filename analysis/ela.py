"""
Error Level Analysis (ELA) para deteccao de conteudo gerado por IA.

Tecnica forense classica que re-comprime a imagem em JPEG e compara
com o original. Regioes manipuladas ou geradas por IA apresentam
niveis de erro diferentes de regioes fotograficas autenticas.

Imagens de IA tendem a ter ELA mais uniforme (erro homogeneo),
enquanto fotos reais apresentam variabilidade natural.
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image
from scipy.stats import skew

logger = logging.getLogger(__name__)


class AnalisadorELA:
    """
    Realiza Error Level Analysis (ELA) para deteccao de manipulacao/geracao.

    A ELA re-salva a imagem em JPEG com uma qualidade especifica e mede
    a diferenca pixel-a-pixel. Regioes com niveis de compressao
    inconsistentes indicam edicao ou geracao artificial.
    """

    # Fator de amplificacao para visualizacao do mapa ELA
    FATOR_AMPLIFICACAO: int = 15

    # Limiar para considerar um pixel como "erro alto" (apos normalizacao)
    LIMIAR_ERRO_ALTO: float = 0.3

    def calcular_ela(
        self,
        imagem: np.ndarray,
        qualidade_jpeg: int = 95,
    ) -> np.ndarray:
        """
        Calcula o mapa de Error Level Analysis da imagem.

        Re-salva a imagem como JPEG na qualidade especificada (em buffer
        de memoria) e calcula a diferenca absoluta pixel-a-pixel.

        Args:
            imagem: Array numpy (H, W, 3) em uint8 RGB.
            qualidade_jpeg: Qualidade de re-compressao JPEG (1-100).

        Returns:
            Mapa ELA (H, W) normalizado [0, 1].
        """
        imagem_pil = Image.fromarray(
            np.clip(imagem, 0, 255).astype(np.uint8)
        )

        # Re-comprime em JPEG via buffer de memoria
        buffer = io.BytesIO()
        imagem_pil.save(buffer, format="JPEG", quality=qualidade_jpeg)
        buffer.seek(0)
        imagem_recomprimida = np.array(Image.open(buffer))

        # Diferenca absoluta
        diferenca = np.abs(
            imagem[:, :, :3].astype(np.float32)
            - imagem_recomprimida[:, :, :3].astype(np.float32)
        )

        # Converte para escala de cinza (media dos canais)
        mapa_ela = diferenca.mean(axis=2)

        # Amplifica para melhor visualizacao
        mapa_ela = mapa_ela * self.FATOR_AMPLIFICACAO

        # Normaliza para [0, 1]
        valor_max = mapa_ela.max()
        if valor_max > 1e-8:
            mapa_ela = mapa_ela / valor_max

        return mapa_ela

    def calcular_score_ela(self, imagem: np.ndarray) -> float:
        """
        Calcula score ELA combinando analise em multiplas qualidades.

        Analisa a distribuicao do erro em 2 qualidades JPEG (90% e 75%).
        Imagens de IA tendem a ter ELA mais uniforme (erro homogeneo),
        fotos reais tendem a ter ELA mais variavel.

        Args:
            imagem: Array numpy (H, W, 3) em uint8 RGB.

        Returns:
            Score entre 0.0 (provavel real) e 1.0 (provavel IA).
        """
        try:
            mapa_q90 = self.calcular_ela(imagem, qualidade_jpeg=90)
            mapa_q75 = self.calcular_ela(imagem, qualidade_jpeg=75)

            # --- Feature 1: Uniformidade do ELA (desvio padrao) ---
            # IA: desvio baixo (erro uniforme) -> score alto
            desvio_q90 = float(np.std(mapa_q90))
            desvio_q75 = float(np.std(mapa_q75))
            desvio_medio = (desvio_q90 + desvio_q75) / 2.0

            # Normaliza: desvio < 0.08 -> uniforme (IA), > 0.25 -> variavel (real)
            score_uniformidade = 1.0 - float(
                np.clip((desvio_medio - 0.08) / 0.17, 0.0, 1.0)
            )

            # --- Feature 2: Skewness da distribuicao de erro ---
            # IA: distribuicao mais simetrica (skewness baixo)
            # Real: distribuicao mais assimetrica (cauda longa)
            skew_q90 = float(skew(mapa_q90.ravel()))
            skew_q75 = float(skew(mapa_q75.ravel()))
            skew_medio = abs((skew_q90 + skew_q75) / 2.0)

            # Normaliza: skew < 0.5 -> simetrico (IA), > 2.0 -> assimetrico (real)
            score_skewness = 1.0 - float(
                np.clip((skew_medio - 0.5) / 1.5, 0.0, 1.0)
            )

            # --- Feature 3: Percentual de pixels com erro alto ---
            # IA: poucos pixels com erro muito alto (geracao uniforme)
            pct_alto_q90 = float((mapa_q90 > self.LIMIAR_ERRO_ALTO).mean())
            pct_alto_q75 = float((mapa_q75 > self.LIMIAR_ERRO_ALTO).mean())
            pct_medio = (pct_alto_q90 + pct_alto_q75) / 2.0

            # Normaliza: < 5% erro alto -> IA, > 25% -> real
            score_erro_alto = 1.0 - float(
                np.clip((pct_medio - 0.05) / 0.20, 0.0, 1.0)
            )

            # Ponderacao das features
            score_final = (
                0.40 * score_uniformidade
                + 0.30 * score_skewness
                + 0.30 * score_erro_alto
            )

            return float(np.clip(score_final, 0.0, 1.0))

        except Exception as erro:
            logger.error(
                "Erro ao calcular score ELA: %s", erro, exc_info=True
            )
            return 0.5
