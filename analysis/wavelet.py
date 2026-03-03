"""
Analise Wavelet para deteccao de conteudo gerado por IA.

Decomposicao multi-escala que captura artefatos em diferentes resolucoes.
Imagens de IA tem coeficientes de detalhe com distribuicao diferente
de imagens fotograficas reais — menos energia em detalhes finos e
kurtosis mais baixa nos coeficientes de alta frequencia.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


class AnalisadorWavelet:
    """
    Realiza analise wavelet para deteccao de conteudo gerado por IA.

    Utiliza a Transformada Wavelet Discreta 2D (DWT) para decompor
    a imagem em sub-bandas de frequencia e extrair features
    estatisticas que diferenciam imagens reais de geradas.
    """

    # Wavelet padrao (Daubechies-1 / Haar)
    WAVELET_PADRAO: str = "db1"

    # Numero de niveis de decomposicao
    NIVEIS_PADRAO: int = 3

    def decompor_wavelet(
        self,
        imagem: np.ndarray,
        wavelet: str = WAVELET_PADRAO,
        niveis: int = NIVEIS_PADRAO,
    ) -> list:
        """
        Aplica DWT 2D na imagem em escala de cinza.

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W).
            wavelet: Tipo de wavelet a usar. Padrao: "db1".
            niveis: Numero de niveis de decomposicao. Padrao: 3.

        Returns:
            Lista de coeficientes retornada por pywt.wavedec2:
            [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        """
        imagem_cinza = self._converter_para_cinza(imagem)
        coeficientes = pywt.wavedec2(
            imagem_cinza, wavelet=wavelet, level=niveis
        )
        return coeficientes

    def extrair_features_wavelet(self, imagem: np.ndarray) -> dict:
        """
        Extrai features estatisticas dos coeficientes wavelet.

        Para cada nivel e sub-banda (LH, HL, HH) calcula:
        media, desvio padrao, skewness e kurtosis dos coeficientes,
        alem da energia relativa por sub-banda.

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W).

        Returns:
            Dicionario com features nomeadas (~36 features).
        """
        coeficientes = self.decompor_wavelet(imagem)
        features: dict[str, float] = {}

        # Energia total para calculo de energia relativa
        energia_total = 0.0
        for i in range(1, len(coeficientes)):
            for sub_banda in coeficientes[i]:
                energia_total += float(np.sum(sub_banda ** 2))
        energia_total = max(energia_total, 1e-8)

        nomes_sub_bandas = ["LH", "HL", "HH"]

        for nivel_idx in range(1, len(coeficientes)):
            nivel = len(coeficientes) - nivel_idx  # Nivel de resolucao
            detalhes = coeficientes[nivel_idx]

            for sb_idx, nome_sb in enumerate(nomes_sub_bandas):
                coefs = detalhes[sb_idx].ravel().astype(np.float64)
                prefixo = f"nivel{nivel}_{nome_sb}"

                features[f"{prefixo}_media"] = float(np.mean(coefs))
                features[f"{prefixo}_desvio"] = float(np.std(coefs))
                features[f"{prefixo}_skewness"] = float(skew(coefs))
                features[f"{prefixo}_kurtosis"] = float(kurtosis(coefs))

                energia_sb = float(np.sum(coefs ** 2))
                features[f"{prefixo}_energia_relativa"] = energia_sb / energia_total

        return features

    def calcular_score_wavelet(self, imagem: np.ndarray) -> float:
        """
        Calcula score heuristico baseado em features wavelet.

        Heuristica:
        - IA: menos energia em detalhes finos (niveis altos), kurtosis mais baixa
        - Real: mais energia de alta frequencia, distribuicao mais "heavy-tailed"

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W).

        Returns:
            Score entre 0.0 (provavel real) e 1.0 (provavel IA).
        """
        try:
            features = self.extrair_features_wavelet(imagem)

            # --- Feature 1: Energia de detalhes finos (nivel 1 = alta freq) ---
            energia_fina = 0.0
            for nome_sb in ["LH", "HL", "HH"]:
                chave = f"nivel1_{nome_sb}_energia_relativa"
                energia_fina += features.get(chave, 0.0)

            # IA: energia fina baixa. Normaliza: < 0.15 -> IA, > 0.50 -> real
            score_energia = 1.0 - float(
                np.clip((energia_fina - 0.15) / 0.35, 0.0, 1.0)
            )

            # --- Feature 2: Kurtosis media dos detalhes finos ---
            kurtosis_fina = 0.0
            contagem = 0
            for nome_sb in ["LH", "HL", "HH"]:
                chave = f"nivel1_{nome_sb}_kurtosis"
                if chave in features:
                    kurtosis_fina += features[chave]
                    contagem += 1
            kurtosis_media = kurtosis_fina / max(contagem, 1)

            # IA: kurtosis baixa (distribuicao mais gaussiana)
            # Real: kurtosis alta (heavy-tailed, texturas naturais)
            # Normaliza: < 2.0 -> IA, > 10.0 -> real
            score_kurtosis = 1.0 - float(
                np.clip((kurtosis_media - 2.0) / 8.0, 0.0, 1.0)
            )

            # --- Feature 3: Razao de energia entre niveis ---
            # IA tende a ter decaimento mais suave entre niveis
            energia_grossa = 0.0
            for nome_sb in ["LH", "HL", "HH"]:
                chave_n3 = f"nivel3_{nome_sb}_energia_relativa"
                energia_grossa += features.get(chave_n3, 0.0)

            razao_niveis = energia_fina / max(energia_grossa, 1e-8)
            # IA: razao mais baixa. Normaliza: < 1.0 -> IA, > 5.0 -> real
            score_razao = 1.0 - float(
                np.clip((razao_niveis - 1.0) / 4.0, 0.0, 1.0)
            )

            # Ponderacao
            score_final = (
                0.40 * score_energia
                + 0.35 * score_kurtosis
                + 0.25 * score_razao
            )

            return float(np.clip(score_final, 0.0, 1.0))

        except Exception as erro:
            logger.error(
                "Erro ao calcular score wavelet: %s", erro, exc_info=True
            )
            return 0.5

    def gerar_mapa_detalhes(self, imagem: np.ndarray) -> np.ndarray:
        """
        Reconstroi imagem apenas com coeficientes de detalhe (sem aproximacao).

        Mostra onde esta a textura e o detalhe fino da imagem.
        Regioes com poucos detalhes podem indicar geracao artificial.

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W).

        Returns:
            Mapa (H, W) normalizado [0, 1] mostrando detalhes.
        """
        coeficientes = self.decompor_wavelet(imagem)

        # Zera a aproximacao (nivel mais baixo)
        coefs_sem_aprox = [np.zeros_like(coeficientes[0])] + list(
            coeficientes[1:]
        )

        # Reconstroi com coeficientes de detalhe apenas
        reconstruida = pywt.waverec2(coefs_sem_aprox, wavelet=self.WAVELET_PADRAO)

        # Toma o valor absoluto e normaliza
        mapa = np.abs(reconstruida)
        valor_max = mapa.max()
        if valor_max > 1e-8:
            mapa = mapa / valor_max

        # Garante que tenha o mesmo tamanho da imagem original
        imagem_cinza = self._converter_para_cinza(imagem)
        if mapa.shape != imagem_cinza.shape:
            mapa = cv2.resize(
                mapa.astype(np.float32),
                (imagem_cinza.shape[1], imagem_cinza.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return mapa.astype(np.float32)

    @staticmethod
    def _converter_para_cinza(imagem: np.ndarray) -> np.ndarray:
        """Converte imagem para escala de cinza float64."""
        if imagem.ndim == 2:
            return imagem.astype(np.float64)

        if imagem.ndim == 3:
            n_canais = imagem.shape[2]
            if n_canais == 1:
                return imagem[:, :, 0].astype(np.float64)
            if n_canais >= 3:
                imagem_u8 = np.clip(imagem, 0, 255).astype(np.uint8)
                cinza = cv2.cvtColor(imagem_u8, cv2.COLOR_RGB2GRAY)
                return cinza.astype(np.float64)

        raise ValueError(f"Shape de imagem nao suportado: {imagem.shape}")
