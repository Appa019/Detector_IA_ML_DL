"""
Analise estatistica em nivel de pixel para deteccao de conteudo gerado por IA.

Imagens fotograficas reais exibem padroes de ruido caracteristicos de sensores
fisicos (ruido gaussiano, JPEG artifacts, etc.), enquanto imagens geradas por
IA tendem a ter superficies artificialmente lisas e estatisticas locais
excessivamente uniformes.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from analysis.srm_kernels import aplicar_filtros_srm

logger = logging.getLogger(__name__)


class AnalisadorPixels:
    """
    Realiza analise estatistica em nivel de pixel para deteccao de IA.

    Metricas computadas:
    - Histogramas por canal RGB
    - Estatisticas locais (media/desvio por bloco)
    - Impressao digital de ruido (noise fingerprint) via filtro passa-alta
    - Score heuristico combinado 0.0-1.0

    Score mais alto indica maior probabilidade de imagem gerada por IA.
    """

    # Numero de bins para histogramas de intensidade
    BINS_HISTOGRAMA: int = 256

    # Tamanho padrao do bloco para analise de estatisticas locais (pixels)
    TAMANHO_BLOCO_PADRAO: int = 32

    # Kernel Laplaciano para extracao de ruido de alta frequencia
    KERNEL_LAPLACIANO: np.ndarray = np.array(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )

    def calcular_histograma_rgb(self, imagem: np.ndarray) -> dict:
        """
        Calcula histogramas de intensidade para cada canal RGB.

        Args:
            imagem: Array numpy (H, W, 3) em uint8 representando imagem RGB.

        Returns:
            Dicionario com chaves 'vermelho', 'verde', 'azul', cada uma
            contendo um array de 256 contagens normalizadas (soma = 1.0).
        """
        if imagem.ndim != 3 or imagem.shape[2] < 3:
            raise ValueError(
                f"Imagem deve ter 3 canais RGB. Shape recebido: {imagem.shape}"
            )

        nomes_canais = ["vermelho", "verde", "azul"]
        histogramas: dict[str, np.ndarray] = {}

        for indice_canal, nome_canal in enumerate(nomes_canais):
            canal = imagem[:, :, indice_canal]
            histograma, _ = np.histogram(
                canal.ravel(),
                bins=self.BINS_HISTOGRAMA,
                range=(0, 256),
            )
            # Normaliza para soma = 1.0 (distribuicao de probabilidade)
            total_pixels = float(histograma.sum())
            histogramas[nome_canal] = (
                histograma / total_pixels if total_pixels > 0 else histograma.astype(float)
            )

        return histogramas

    def calcular_estatisticas_locais(
        self,
        imagem: np.ndarray,
        tamanho_bloco: int = TAMANHO_BLOCO_PADRAO,
    ) -> dict:
        """
        Divide a imagem em blocos e calcula media e desvio por bloco.

        Imagens de IA tendem a ter estatisticas locais mais uniformes
        (menor variacao entre blocos) do que imagens fotograficas reais,
        que apresentam variabilidade natural de textura e iluminacao.

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W) em uint8.
            tamanho_bloco: Lado do bloco quadrado em pixels. Padrao: 32.

        Returns:
            Dicionario com:
            - 'medias_blocos': array de medias de intensidade por bloco
            - 'desvios_blocos': array de desvios padrao por bloco
            - 'media_das_medias': media global das medias de bloco
            - 'desvio_das_medias': desvio padrao das medias de bloco
            - 'media_dos_desvios': media dos desvios por bloco
            - 'desvio_dos_desvios': desvio padrao dos desvios por bloco
            - 'coef_variacao_global': cv das medias de bloco (std/mean)
            - 'uniformidade_local': 1 - (desvio_dos_desvios / media_dos_desvios + eps)
        """
        imagem_cinza = self._converter_para_cinza_uint8(imagem)
        altura, largura = imagem_cinza.shape

        medias: list[float] = []
        desvios: list[float] = []

        for topo in range(0, altura - tamanho_bloco + 1, tamanho_bloco):
            for esquerda in range(0, largura - tamanho_bloco + 1, tamanho_bloco):
                bloco = imagem_cinza[
                    topo : topo + tamanho_bloco,
                    esquerda : esquerda + tamanho_bloco,
                ].astype(np.float32)
                medias.append(float(bloco.mean()))
                desvios.append(float(bloco.std()))

        if not medias:
            return self._estatisticas_locais_vazias()

        arr_medias = np.array(medias, dtype=np.float64)
        arr_desvios = np.array(desvios, dtype=np.float64)

        media_das_medias = float(np.mean(arr_medias))
        desvio_das_medias = float(np.std(arr_medias))
        media_dos_desvios = float(np.mean(arr_desvios))
        desvio_dos_desvios = float(np.std(arr_desvios))

        coef_variacao_global = desvio_das_medias / (media_das_medias + 1e-8)
        uniformidade_local = 1.0 - desvio_dos_desvios / (media_dos_desvios + 1e-8)
        uniformidade_local = float(np.clip(uniformidade_local, 0.0, 1.0))

        return {
            "medias_blocos": arr_medias,
            "desvios_blocos": arr_desvios,
            "media_das_medias": media_das_medias,
            "desvio_das_medias": desvio_das_medias,
            "media_dos_desvios": media_dos_desvios,
            "desvio_dos_desvios": desvio_dos_desvios,
            "coef_variacao_global": float(coef_variacao_global),
            "uniformidade_local": uniformidade_local,
        }

    def calcular_noise_print(self, imagem: np.ndarray) -> np.ndarray:
        """
        Extrai a impressao digital de ruido da imagem (noise fingerprint).

        Aplica os 30 filtros SRM (Spatial Rich Model) e toma a media dos
        residuos como noise print. Captura padroes de ruido do sensor/algoritmo
        de geracao que diferem entre imagens reais e geradas por IA.

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W) em uint8.

        Returns:
            Residuo de ruido normalizado [-1.0, 1.0], shape (H, W).
        """
        imagem_cinza = self._converter_para_cinza_uint8(imagem)

        # Aplica 30 filtros SRM e toma a media dos residuos
        residuos_srm = aplicar_filtros_srm(imagem_cinza)  # (30, H, W)
        residuo = residuos_srm.mean(axis=0)  # (H, W)

        # Normaliza para [-1, 1]
        valor_maximo = np.abs(residuo).max()
        if valor_maximo > 1e-8:
            residuo_normalizado = residuo / valor_maximo
        else:
            residuo_normalizado = residuo

        return residuo_normalizado

    def calcular_score_pixels(self, imagem: np.ndarray) -> float:
        """
        Combina metricas de pixel em um score heuristico 0.0-1.0.

        Score mais proximo de 1.0 indica maior probabilidade de imagem
        gerada por IA. A heuristica considera:
        1. Uniformidade local (blocos com desvio muito similar -> IA)
        2. Amplitude do ruido (ruido muito baixo -> IA)
        3. Coeficiente de variacao global (muito uniforme -> IA)

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W) em uint8.

        Returns:
            Score entre 0.0 (provavel real) e 1.0 (provavel IA).
        """
        try:
            # --- Feature 1: uniformidade das estatisticas locais ---
            stats_locais = self.calcular_estatisticas_locais(imagem)
            uniformidade_local: float = stats_locais["uniformidade_local"]

            # --- Feature 2: nivel de ruido via noise fingerprint ---
            residuo = self.calcular_noise_print(imagem)
            amplitude_ruido = float(np.std(residuo))
            # Normaliza: assume que residuo < 0.05 e "muito liso" (IA)
            # e > 0.20 e "ruidoso" (real). Clamp para [0, 1].
            score_ruido_ia = 1.0 - float(np.clip(amplitude_ruido / 0.20, 0.0, 1.0))

            # --- Feature 3: coeficiente de variacao global ---
            coef_var: float = stats_locais["coef_variacao_global"]
            # CV muito baixo (imagem muito uniforme) e indicativo de IA
            # Normaliza: CV < 0.05 -> IA, CV > 0.30 -> real
            score_cv_ia = 1.0 - float(np.clip((coef_var - 0.05) / 0.25, 0.0, 1.0))

            # Ponderacao empirica das features
            pesos = np.array([0.4, 0.35, 0.25])
            scores = np.array([uniformidade_local, score_ruido_ia, score_cv_ia])
            score_final = float(np.dot(pesos, scores))

            return float(np.clip(score_final, 0.0, 1.0))

        except Exception as erro:
            logger.error(
                "Erro ao calcular score de pixels: %s", erro, exc_info=True
            )
            return 0.0

    # ------------------------------------------------------------------
    # Analise de consistencia de ruido
    # ------------------------------------------------------------------

    def calcular_mapa_variancia_ruido(
        self,
        imagem: np.ndarray,
        tamanho_bloco: int = 16,
    ) -> np.ndarray:
        """
        Gera mapa de variancia de ruido por bloco com sobreposicao de 50%.

        Estima o ruido local via MAD (Median Absolute Deviation) do
        Laplaciano em cada bloco. Regioes com variancia muito diferente
        da mediana indicam inconsistencia (possivel manipulacao ou geracao).

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W) em uint8.
            tamanho_bloco: Lado do bloco quadrado em pixels. Padrao: 16.

        Returns:
            Mapa 2D com variancia de ruido por regiao.
        """
        imagem_cinza = self._converter_para_cinza_uint8(imagem)
        altura, largura = imagem_cinza.shape

        # Aplica filtro Laplaciano para extrair ruido de alta frequencia
        laplaciano = cv2.filter2D(
            imagem_cinza.astype(np.float32), -1, self.KERNEL_LAPLACIANO
        )

        # Stride com 50% de sobreposicao
        passo = max(1, tamanho_bloco // 2)
        linhas_blocos = max(1, (altura - tamanho_bloco) // passo + 1)
        colunas_blocos = max(1, (largura - tamanho_bloco) // passo + 1)

        mapa_variancia = np.zeros(
            (linhas_blocos, colunas_blocos), dtype=np.float32
        )

        for i in range(linhas_blocos):
            for j in range(colunas_blocos):
                topo = i * passo
                esquerda = j * passo
                bloco = laplaciano[
                    topo : topo + tamanho_bloco,
                    esquerda : esquerda + tamanho_bloco,
                ]
                # MAD (Median Absolute Deviation) como estimativa robusta
                mediana_bloco = float(np.median(np.abs(bloco)))
                mapa_variancia[i, j] = mediana_bloco

        return mapa_variancia

    def calcular_mapa_inconsistencia(
        self,
        imagem: np.ndarray,
        tamanho_bloco: int = 16,
    ) -> np.ndarray:
        """
        Calcula mapa de inconsistencia de ruido normalizado.

        Regioes com variancia de ruido muito diferente da mediana global
        sao marcadas como inconsistentes. O mapa e interpolado para a
        resolucao original da imagem.

        Args:
            imagem: Array numpy (H, W, 3) ou (H, W) em uint8.
            tamanho_bloco: Lado do bloco quadrado em pixels. Padrao: 16.

        Returns:
            Mapa (H, W) normalizado [0, 1]. Regioes brilhantes = inconsistentes.
        """
        mapa_variancia = self.calcular_mapa_variancia_ruido(
            imagem, tamanho_bloco
        )

        # Normaliza relativo a mediana
        mediana_global = float(np.median(mapa_variancia))
        if mediana_global > 1e-8:
            mapa_inconsistencia = np.abs(mapa_variancia - mediana_global) / mediana_global
        else:
            mapa_inconsistencia = np.zeros_like(mapa_variancia)

        # Normaliza para [0, 1]
        valor_max = mapa_inconsistencia.max()
        if valor_max > 1e-8:
            mapa_inconsistencia = mapa_inconsistencia / valor_max

        # Upsample para resolucao original
        imagem_cinza = self._converter_para_cinza_uint8(imagem)
        altura, largura = imagem_cinza.shape
        mapa_full = cv2.resize(
            mapa_inconsistencia,
            (largura, altura),
            interpolation=cv2.INTER_LINEAR,
        )

        return np.clip(mapa_full, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Metodos auxiliares privados
    # ------------------------------------------------------------------

    @staticmethod
    def _converter_para_cinza_uint8(imagem: np.ndarray) -> np.ndarray:
        """Converte imagem para escala de cinza uint8 se necessario."""
        if imagem.ndim == 2:
            return imagem.astype(np.uint8)

        if imagem.ndim == 3:
            n_canais = imagem.shape[2]
            if n_canais == 1:
                return imagem[:, :, 0].astype(np.uint8)
            if n_canais >= 3:
                # cv2.cvtColor espera uint8
                imagem_u8 = np.clip(imagem, 0, 255).astype(np.uint8)
                return cv2.cvtColor(imagem_u8, cv2.COLOR_RGB2GRAY)

        raise ValueError(
            f"Shape de imagem nao suportado: {imagem.shape}"
        )

    @staticmethod
    def _estatisticas_locais_vazias() -> dict:
        """Retorna estrutura de estatisticas locais vazia como fallback."""
        vazio = np.array([], dtype=np.float64)
        return {
            "medias_blocos": vazio,
            "desvios_blocos": vazio,
            "media_das_medias": 0.0,
            "desvio_das_medias": 0.0,
            "media_dos_desvios": 0.0,
            "desvio_dos_desvios": 0.0,
            "coef_variacao_global": 0.0,
            "uniformidade_local": 0.0,
        }
