"""
Analise espectral de imagens usando FFT e DCT.

Imagens geradas por IA tendem a apresentar distribuicoes espectrais
distintas das imagens reais — em especial, menor energia em altas
frequencias e artefatos periodicos caracteristicos.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from scipy.fft import dctn

logger = logging.getLogger(__name__)


class AnalisadorEspectral:
    """
    Realiza analise espectral de imagens via FFT e DCT.

    Imagens geradas por IA tendem a ter:
    - Menos energia em altas frequencias (superficies mais "lisas")
    - Curtose e assimetria distintas no espectro de magnitude
    - Coeficientes DCT com distribuicao diferente das imagens fotograficas
    """

    # Limiar padrao que separa baixas de altas frequencias
    # (fracao do raio maximo no espaco espectral)
    LIMIAR_FREQUENCIA_PADRAO: float = 0.5

    def calcular_fft_2d(self, imagem: np.ndarray) -> np.ndarray:
        """
        Calcula o espectro de magnitude 2D via FFT.

        A imagem e convertida para escala de cinza se tiver 3 canais.
        O espectro e centrado (DC no meio) e convertido para escala
        logaritmica para facilitar visualizacao e analise.

        Args:
            imagem: Array numpy com shape (H, W) ou (H, W, C).

        Returns:
            Espectro de magnitude centrado em escala log, shape (H, W).
        """
        imagem_cinza = self._converter_para_cinza(imagem)

        # FFT 2D e centraliza componente DC
        espectro_fft = np.fft.fft2(imagem_cinza.astype(np.float64))
        espectro_centralizado = np.fft.fftshift(espectro_fft)

        # Magnitude em escala logaritmica (evita dominancia do DC)
        magnitude = np.abs(espectro_centralizado)
        espectro_log = np.log1p(magnitude)

        return espectro_log

    def calcular_dct_2d(self, imagem: np.ndarray) -> np.ndarray:
        """
        Calcula a Transformada Discreta do Cosseno 2D (DCT-II).

        Usa scipy.fft.dctn com norm='ortho' para normalizacao ortonormal,
        o que facilita a comparacao de coeficientes entre imagens de
        tamanhos diferentes.

        Args:
            imagem: Array numpy com shape (H, W) ou (H, W, C).

        Returns:
            Coeficientes DCT 2D, shape (H, W).
        """
        imagem_cinza = self._converter_para_cinza(imagem)
        coeficientes_dct = dctn(imagem_cinza.astype(np.float64), norm="ortho")
        return coeficientes_dct

    def calcular_media_azimuthal(self, espectro: np.ndarray) -> np.ndarray:
        """
        Calcula o perfil azimutal medio do espectro 2D.

        Agrupa os valores do espectro por distancia radial ao centro
        (componente DC) e calcula a media em cada anel. O resultado
        representa a distribuicao de energia por frequencia radial,
        independente da direcao.

        Args:
            espectro: Espectro 2D (e.g., saida de calcular_fft_2d),
                      shape (H, W).

        Returns:
            Perfil azimutal 1D, com comprimento igual ao raio maximo
            em pixels.
        """
        altura, largura = espectro.shape
        centro_y, centro_x = altura // 2, largura // 2

        # Mapa de distancias radiais ao centro
        indices_y = np.arange(altura) - centro_y
        indices_x = np.arange(largura) - centro_x
        grade_x, grade_y = np.meshgrid(indices_x, indices_y)
        mapa_raios = np.sqrt(grade_x**2 + grade_y**2).astype(int)

        raio_maximo = min(centro_y, centro_x)
        perfil_azimuthal = np.zeros(raio_maximo, dtype=np.float64)

        for raio in range(raio_maximo):
            mascara = mapa_raios == raio
            if mascara.any():
                perfil_azimuthal[raio] = espectro[mascara].mean()

        return perfil_azimuthal

    def extrair_features_frequencia(self, imagem: np.ndarray) -> dict:
        """
        Extrai um conjunto de features estatisticas do espectro de frequencias.

        As features cobrem:
        - Estatisticas globais do espectro FFT (media, desvio, assimetria, curtose)
        - Razao de energia alta/baixa frequencia
        - Caracteristicas do perfil azimutal (inclinacao espectral, energia media)
        - Estatisticas dos coeficientes DCT (energia no canto superior esquerdo
          versus demais regioes)

        Args:
            imagem: Array numpy com shape (H, W) ou (H, W, C).

        Returns:
            Dicionario com features numericas extraidas do espectro.
        """
        try:
            espectro_fft = self.calcular_fft_2d(imagem)
            coef_dct = self.calcular_dct_2d(imagem)
            perfil_azimuthal = self.calcular_media_azimuthal(espectro_fft)

            valores_espectro = espectro_fft.ravel()

            # Estatisticas globais do espectro FFT
            media_espectro = float(np.mean(valores_espectro))
            desvio_espectro = float(np.std(valores_espectro))
            assimetria_espectro = float(stats.skew(valores_espectro))
            curtose_espectro = float(stats.kurtosis(valores_espectro))

            # Razao alta/baixa frequencia
            razao_hf_lf = self.calcular_razao_frequencia(
                espectro_fft, self.LIMIAR_FREQUENCIA_PADRAO
            )

            # Features do perfil azimutal
            if len(perfil_azimuthal) > 1:
                inclinacao_espectral = self._estimar_inclinacao_espectral(
                    perfil_azimuthal
                )
                media_perfil_hf = float(
                    np.mean(perfil_azimuthal[len(perfil_azimuthal) // 2 :])
                )
                media_perfil_lf = float(
                    np.mean(perfil_azimuthal[: len(perfil_azimuthal) // 2])
                )
            else:
                inclinacao_espectral = 0.0
                media_perfil_hf = 0.0
                media_perfil_lf = 0.0

            # Estatisticas dos coeficientes DCT
            # Coeficientes de baixa freq. ficam no canto superior esquerdo
            altura_dct, largura_dct = coef_dct.shape
            regiao_lf = coef_dct[
                : altura_dct // 4,
                : largura_dct // 4,
            ]
            energia_lf_dct = float(np.mean(np.abs(regiao_lf)))
            energia_total_dct = float(np.mean(np.abs(coef_dct)))
            razao_energia_dct = (
                energia_lf_dct / (energia_total_dct + 1e-8)
            )

            return {
                # Espectro FFT global
                "media_espectro": media_espectro,
                "desvio_espectro": desvio_espectro,
                "assimetria_espectro": assimetria_espectro,
                "curtose_espectro": curtose_espectro,
                # Distribuicao de energia em frequencias
                "razao_hf_lf": razao_hf_lf,
                # Perfil azimutal
                "inclinacao_espectral": inclinacao_espectral,
                "media_perfil_alta_freq": media_perfil_hf,
                "media_perfil_baixa_freq": media_perfil_lf,
                # DCT
                "energia_lf_dct": energia_lf_dct,
                "energia_total_dct": energia_total_dct,
                "razao_energia_dct": razao_energia_dct,
            }

        except Exception as erro:
            logger.error(
                "Erro ao extrair features de frequencia: %s", erro, exc_info=True
            )
            # Retorna features zeradas para nao interromper o pipeline
            return self._features_vazias()

    def calcular_razao_frequencia(
        self, espectro: np.ndarray, limiar: float = 0.5
    ) -> float:
        """
        Calcula a razao de energia entre altas e baixas frequencias.

        Imagens geradas por IA tipicamente tem razao menor (espectro
        mais concentrado em baixas frequencias, superficies mais lisas).

        Args:
            espectro: Espectro 2D centrado (saida de calcular_fft_2d),
                      shape (H, W).
            limiar: Fracao do raio maximo que separa baixas de altas
                    frequencias. Padrao: 0.5 (metade do raio).

        Returns:
            Razao energia_alta_freq / (energia_total + eps). Valores
            menores indicam maior concentracao em baixas frequencias.
        """
        if espectro.ndim != 2:
            raise ValueError(
                f"Espectro deve ser 2D, recebido shape: {espectro.shape}"
            )

        altura, largura = espectro.shape
        centro_y, centro_x = altura // 2, largura // 2
        raio_maximo = min(centro_y, centro_x)
        limiar_pixels = int(raio_maximo * limiar)

        indices_y = np.arange(altura) - centro_y
        indices_x = np.arange(largura) - centro_x
        grade_x, grade_y = np.meshgrid(indices_x, indices_y)
        distancia = np.sqrt(grade_x**2 + grade_y**2)

        mascara_hf = distancia > limiar_pixels
        energia_hf = float(np.sum(espectro[mascara_hf]))
        energia_total = float(np.sum(espectro))

        return energia_hf / (energia_total + 1e-8)

    # ------------------------------------------------------------------
    # Metodos auxiliares privados
    # ------------------------------------------------------------------

    @staticmethod
    def _converter_para_cinza(imagem: np.ndarray) -> np.ndarray:
        """Converte imagem RGB/RGBA para escala de cinza se necessario."""
        if imagem.ndim == 2:
            return imagem

        if imagem.ndim == 3:
            n_canais = imagem.shape[2]
            if n_canais == 1:
                return imagem[:, :, 0]
            if n_canais >= 3:
                # Pesos luminance (ITU-R BT.601)
                return (
                    0.299 * imagem[:, :, 0]
                    + 0.587 * imagem[:, :, 1]
                    + 0.114 * imagem[:, :, 2]
                )

        raise ValueError(
            f"Shape de imagem nao suportado para conversao em cinza: {imagem.shape}"
        )

    @staticmethod
    def _estimar_inclinacao_espectral(perfil: np.ndarray) -> float:
        """
        Estima a inclinacao espectral via regressao linear em escala log-log.

        A inclinacao (expoente) do perfil azimutal em log-log e um descritor
        classico da estrutura de frequencias de uma imagem.
        """
        comprimento = len(perfil)
        if comprimento < 2:
            return 0.0

        frequencias = np.arange(1, comprimento + 1, dtype=np.float64)
        valores_positivos = np.maximum(perfil, 1e-10)

        log_freq = np.log(frequencias)
        log_amplitude = np.log(valores_positivos)

        # Regressao linear: log(A) = a * log(f) + b
        coeficientes = np.polyfit(log_freq, log_amplitude, deg=1)
        inclinacao = float(coeficientes[0])
        return inclinacao

    @staticmethod
    def _features_vazias() -> dict:
        """Retorna dicionario de features com valores zerados como fallback."""
        return {
            "media_espectro": 0.0,
            "desvio_espectro": 0.0,
            "assimetria_espectro": 0.0,
            "curtose_espectro": 0.0,
            "razao_hf_lf": 0.0,
            "inclinacao_espectral": 0.0,
            "media_perfil_alta_freq": 0.0,
            "media_perfil_baixa_freq": 0.0,
            "energia_lf_dct": 0.0,
            "energia_total_dct": 0.0,
            "razao_energia_dct": 0.0,
        }
