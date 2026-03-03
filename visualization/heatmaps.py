"""
Visualizacoes baseadas em Matplotlib e OpenCV para mapas de calor,
espectro de frequencia e analise de ruido de imagens.

Todas as funcoes retornam objetos matplotlib.figure.Figure, prontos
para uso com st.pyplot(figura) no Streamlit.

O backend 'Agg' e configurado aqui para evitar erros em ambientes sem
display grafico (servidores, containers Docker, Streamlit Cloud).
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # Backend nao-interativo — deve vir antes de pyplot
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import cv2

from analysis.ela import AnalisadorELA
from analysis.frequency import AnalisadorEspectral
from analysis.pixel_stats import AnalisadorPixels
from analysis.wavelet import AnalisadorWavelet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de estilo
# ---------------------------------------------------------------------------

# Cor de fundo dos graficos — compativel com dashboards escuros
COR_FUNDO: str = "#1a1a2e"
COR_TEXTO: str = "#e0e0e0"
COR_EIXO: str = "#555577"

# Resolucao padrao das figuras geradas
DPI_PADRAO: int = 100

# Estilo global aplicado a todas as figuras desta classe
ESTILO_MATPLOTLIB: dict = {
    "figure.facecolor": COR_FUNDO,
    "axes.facecolor": "#16213e",
    "axes.edgecolor": COR_EIXO,
    "axes.labelcolor": COR_TEXTO,
    "xtick.color": COR_TEXTO,
    "ytick.color": COR_TEXTO,
    "text.color": COR_TEXTO,
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
}


def _aplicar_estilo() -> None:
    """Aplica as configuracoes de estilo global do modulo ao Matplotlib."""
    plt.rcParams.update(ESTILO_MATPLOTLIB)


# ---------------------------------------------------------------------------
# Funcoes auxiliares internas
# ---------------------------------------------------------------------------


def _redimensionar_para_altura(
    imagem: np.ndarray, altura_alvo: int = 224
) -> np.ndarray:
    """
    Redimensiona a imagem mantendo a proporcao original, fixando a altura.

    Args:
        imagem: Array numpy (H, W, 3) ou (H, W).
        altura_alvo: Altura desejada em pixels. Padrao: 224.

    Returns:
        Imagem redimensionada como array numpy uint8.
    """
    altura_orig, largura_orig = imagem.shape[:2]
    if altura_orig == 0:
        return imagem
    fator = altura_alvo / altura_orig
    nova_largura = max(1, int(largura_orig * fator))
    return cv2.resize(
        imagem,
        (nova_largura, altura_alvo),
        interpolation=cv2.INTER_LINEAR,
    )


def _garantir_uint8_rgb(imagem: np.ndarray) -> np.ndarray:
    """
    Converte qualquer array numpy para uint8 RGB (H, W, 3).

    Trata imagens em escala de cinza, RGBA e arrays float [0,1] ou [0,255].

    Args:
        imagem: Array numpy com qualquer dtype e numero de canais.

    Returns:
        Array numpy uint8 com shape (H, W, 3).
    """
    # Normaliza dtype para float32 primeiro
    if imagem.dtype != np.uint8:
        imagem_f = imagem.astype(np.float32)
        # Se os valores estiverem em [0, 1], converte para [0, 255]
        if imagem_f.max() <= 1.0 + 1e-6:
            imagem_f = imagem_f * 255.0
        imagem = np.clip(imagem_f, 0, 255).astype(np.uint8)

    # Garante 3 canais RGB
    if imagem.ndim == 2:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2RGB)
    elif imagem.ndim == 3 and imagem.shape[2] == 1:
        imagem = cv2.cvtColor(imagem[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif imagem.ndim == 3 and imagem.shape[2] == 4:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGBA2RGB)

    return imagem


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------


class VisualizadorMapasCalor:
    """
    Gerador de visualizacoes de mapas de calor e analise de imagem.

    Utiliza Matplotlib (backend Agg) para gerar figuras estaticas
    compatíveis com st.pyplot() no Streamlit. As figuras devem ser
    fechadas pelo chamador apos o uso para liberar memoria
    (plt.close(figura)).
    """

    def __init__(self) -> None:
        self._analisador_espectral = AnalisadorEspectral()
        self._analisador_pixels = AnalisadorPixels()
        self._analisador_ela = AnalisadorELA()
        self._analisador_wavelet = AnalisadorWavelet()
        _aplicar_estilo()

    def criar_visualizacao_gradcam(
        self,
        imagem: np.ndarray,
        mapa_calor: np.ndarray,
        titulo: str = "GradCAM",
    ) -> matplotlib.figure.Figure:
        """
        Cria visualizacao lado a lado: imagem original e sobreposicao GradCAM.

        O mapa de calor e redimensionado para coincidir com a imagem original,
        colorizado com o colormap JET e misturado com alpha = 0.4.

        Args:
            imagem: Imagem original como array numpy (H, W, 3) uint8 RGB.
            mapa_calor: Mapa de ativacao normalizado [0, 1] com shape (H, W).
                        Tipicamente gerado por GeradorGradCAM.gerar().
            titulo: Titulo geral da figura. Padrao: "GradCAM".

        Returns:
            Figura Matplotlib com dois paineis: original e sobreposicao.
        """
        imagem_rgb = _garantir_uint8_rgb(imagem)
        altura, largura = imagem_rgb.shape[:2]

        # Redimensiona mapa de calor se necessario
        if mapa_calor.shape != (altura, largura):
            mapa_calor_redim = cv2.resize(
                mapa_calor.astype(np.float32),
                (largura, altura),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            mapa_calor_redim = mapa_calor.astype(np.float32)

        # Aplica colormap JET ao mapa [0,1] -> RGB
        mapa_uint8 = np.uint8(255 * np.clip(mapa_calor_redim, 0.0, 1.0))
        mapa_bgr = cv2.applyColorMap(mapa_uint8, cv2.COLORMAP_JET)
        mapa_rgb = cv2.cvtColor(mapa_bgr, cv2.COLOR_BGR2RGB)

        # Blend linear
        alfa = 0.4
        sobreposicao = cv2.addWeighted(
            imagem_rgb.astype(np.float32), 1.0 - alfa,
            mapa_rgb.astype(np.float32), alfa,
            0,
        )
        sobreposicao = np.clip(sobreposicao, 0, 255).astype(np.uint8)

        figura, eixos = plt.subplots(
            1, 2, figsize=(10, 4), dpi=DPI_PADRAO
        )
        figura.patch.set_facecolor(COR_FUNDO)

        eixos[0].imshow(imagem_rgb)
        eixos[0].set_title("Imagem Original", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[0].axis("off")

        eixos[1].imshow(sobreposicao)
        eixos[1].set_title("Sobreposicao GradCAM", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[1].axis("off")

        # Barra de cores do mapa de calor ao lado direito
        mapa_normalizado = plt.cm.ScalarMappable(
            cmap="jet", norm=plt.Normalize(vmin=0, vmax=1)
        )
        mapa_normalizado.set_array([])
        barra_cores = figura.colorbar(
            mapa_normalizado, ax=eixos[1], fraction=0.046, pad=0.04
        )
        barra_cores.set_label("Ativacao", color=COR_TEXTO, fontsize=10)
        barra_cores.ax.yaxis.set_tick_params(color=COR_TEXTO)
        plt.setp(barra_cores.ax.yaxis.get_ticklabels(), color=COR_TEXTO)

        figura.suptitle(titulo, color=COR_TEXTO, fontsize=14, fontweight="bold", y=1.02)
        figura.tight_layout()

        return figura

    def criar_visualizacao_espectro(
        self,
        imagem: np.ndarray,
    ) -> matplotlib.figure.Figure:
        """
        Cria grade 2x2 com analise espectral completa da imagem.

        Paineis:
        - Superior esquerdo: imagem original RGB.
        - Superior direito: espectro de magnitude FFT em escala log (colormap hot).
        - Inferior esquerdo: espectro de fase FFT (colormap twilight).
        - Inferior direito: perfil azimutal medio (grafico de linha).

        Args:
            imagem: Imagem como array numpy (H, W, 3) ou (H, W) em uint8.

        Returns:
            Figura Matplotlib 2x2 com a analise espectral.
        """
        imagem_rgb = _garantir_uint8_rgb(imagem)

        try:
            espectro_magnitude = self._analisador_espectral.calcular_fft_2d(imagem_rgb)
            perfil_azimuthal = self._analisador_espectral.calcular_media_azimuthal(
                espectro_magnitude
            )

            # Espectro de fase (nao normalizado pelo AnalisadorEspectral — calculado aqui)
            imagem_cinza = (
                0.299 * imagem_rgb[:, :, 0]
                + 0.587 * imagem_rgb[:, :, 1]
                + 0.114 * imagem_rgb[:, :, 2]
            )
            espectro_bruto = np.fft.fftshift(np.fft.fft2(imagem_cinza.astype(np.float64)))
            espectro_fase = np.angle(espectro_bruto)

        except Exception as erro:
            logger.error("Erro ao calcular espectro FFT: %s", erro, exc_info=True)
            # Fallback: paineis vazios
            espectro_magnitude = np.zeros((64, 64))
            espectro_fase = np.zeros((64, 64))
            perfil_azimuthal = np.zeros(32)

        figura, eixos = plt.subplots(
            2, 2, figsize=(11, 9), dpi=DPI_PADRAO
        )
        figura.patch.set_facecolor(COR_FUNDO)

        # Painel 1: imagem original
        eixos[0, 0].imshow(imagem_rgb)
        eixos[0, 0].set_title("Imagem Original", color=COR_TEXTO, fontsize=11, pad=6)
        eixos[0, 0].axis("off")

        # Painel 2: magnitude FFT
        img_magnitude = eixos[0, 1].imshow(
            espectro_magnitude, cmap="hot", origin="upper"
        )
        eixos[0, 1].set_title(
            "Espectro de Magnitude FFT (log)", color=COR_TEXTO, fontsize=11, pad=6
        )
        eixos[0, 1].axis("off")
        barra_mag = figura.colorbar(img_magnitude, ax=eixos[0, 1], fraction=0.046, pad=0.04)
        barra_mag.set_label("log(1 + |F|)", color=COR_TEXTO, fontsize=9)
        plt.setp(barra_mag.ax.yaxis.get_ticklabels(), color=COR_TEXTO)
        barra_mag.ax.yaxis.set_tick_params(color=COR_TEXTO)

        # Painel 3: espectro de fase
        img_fase = eixos[1, 0].imshow(
            espectro_fase, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="upper"
        )
        eixos[1, 0].set_title(
            "Espectro de Fase FFT", color=COR_TEXTO, fontsize=11, pad=6
        )
        eixos[1, 0].axis("off")
        barra_fase = figura.colorbar(img_fase, ax=eixos[1, 0], fraction=0.046, pad=0.04)
        barra_fase.set_label("Fase (rad)", color=COR_TEXTO, fontsize=9)
        plt.setp(barra_fase.ax.yaxis.get_ticklabels(), color=COR_TEXTO)
        barra_fase.ax.yaxis.set_tick_params(color=COR_TEXTO)

        # Painel 4: perfil azimutal
        eixo_perfil = eixos[1, 1]
        if len(perfil_azimuthal) > 1:
            frequencias = np.arange(len(perfil_azimuthal))
            eixo_perfil.plot(
                frequencias,
                perfil_azimuthal,
                color="#00d4ff",
                linewidth=1.8,
                label="Perfil azimutal",
            )
            eixo_perfil.fill_between(
                frequencias, perfil_azimuthal, alpha=0.2, color="#00d4ff"
            )
            eixo_perfil.axvline(
                x=len(perfil_azimuthal) // 2,
                color="#ff6b6b",
                linestyle="--",
                linewidth=1,
                label="Freq. media",
            )
            eixo_perfil.legend(
                fontsize=9, facecolor="#16213e", edgecolor=COR_EIXO, labelcolor=COR_TEXTO
            )
        else:
            eixo_perfil.text(
                0.5, 0.5, "Sem dados",
                ha="center", va="center", transform=eixo_perfil.transAxes,
                color="#888888", fontsize=12,
            )

        eixo_perfil.set_title(
            "Perfil Azimutal Medio", color=COR_TEXTO, fontsize=11, pad=6
        )
        eixo_perfil.set_xlabel("Frequencia radial (px)", color=COR_TEXTO, fontsize=9)
        eixo_perfil.set_ylabel("Amplitude media", color=COR_TEXTO, fontsize=9)
        eixo_perfil.grid(True, alpha=0.3, color=COR_EIXO)
        eixo_perfil.set_facecolor("#16213e")

        figura.suptitle(
            "Analise Espectral de Frequencias",
            color=COR_TEXTO, fontsize=14, fontweight="bold", y=1.01,
        )
        figura.tight_layout()

        return figura

    def criar_visualizacao_noise_print(
        self,
        imagem: np.ndarray,
    ) -> matplotlib.figure.Figure:
        """
        Cria visualizacao lado a lado da imagem original e do residuo de ruido.

        O residuo de ruido (noise fingerprint) e calculado via filtro Laplaciano
        e exibido com o colormap RdBu centralizado em zero — azul para valores
        negativos, branco para zero e vermelho para valores positivos.

        Args:
            imagem: Imagem original como array numpy (H, W, 3) uint8 RGB.

        Returns:
            Figura Matplotlib com dois paineis: original e residuo de ruido.
        """
        imagem_rgb = _garantir_uint8_rgb(imagem)

        try:
            residuo = self._analisador_pixels.calcular_noise_print(imagem_rgb)
        except Exception as erro:
            logger.error("Erro ao calcular noise print: %s", erro, exc_info=True)
            residuo = np.zeros(imagem_rgb.shape[:2], dtype=np.float32)

        figura, eixos = plt.subplots(1, 2, figsize=(10, 4), dpi=DPI_PADRAO)
        figura.patch.set_facecolor(COR_FUNDO)

        eixos[0].imshow(imagem_rgb)
        eixos[0].set_title("Imagem Original", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[0].axis("off")

        # Mapa de residuo: centralizado em 0 com RdBu
        valor_abs_max = max(float(np.abs(residuo).max()), 1e-8)
        img_ruido = eixos[1].imshow(
            residuo,
            cmap="RdBu",
            vmin=-valor_abs_max,
            vmax=valor_abs_max,
            interpolation="nearest",
        )
        eixos[1].set_title("Residuo de Ruido (Noise Print)", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[1].axis("off")

        barra = figura.colorbar(img_ruido, ax=eixos[1], fraction=0.046, pad=0.04)
        barra.set_label("Amplitude do Residuo", color=COR_TEXTO, fontsize=10)
        plt.setp(barra.ax.yaxis.get_ticklabels(), color=COR_TEXTO)
        barra.ax.yaxis.set_tick_params(color=COR_TEXTO)

        # Anotacao com desvio padrao do ruido
        desvio_ruido = float(np.std(residuo))
        figura.text(
            0.75, -0.02,
            f"Desvio padrao do ruido: {desvio_ruido:.4f}",
            ha="center", color="#aaaaaa", fontsize=9,
            transform=figura.transFigure,
        )

        figura.suptitle(
            "Impressao Digital de Ruido (Noise Fingerprint)",
            color=COR_TEXTO, fontsize=14, fontweight="bold", y=1.02,
        )
        figura.tight_layout()

        return figura

    def criar_visualizacao_ela(
        self,
        imagem: np.ndarray,
    ) -> matplotlib.figure.Figure:
        """
        Cria visualizacao lado a lado do ELA em duas qualidades.

        Layout 1x3: Original | ELA Q95 | ELA Q75.
        Colormap inferno — regioes com alto erro aparecem brilhantes.

        Args:
            imagem: Imagem original como array numpy (H, W, 3) uint8 RGB.

        Returns:
            Figura Matplotlib com tres paineis de ELA.
        """
        imagem_rgb = _garantir_uint8_rgb(imagem)

        try:
            mapa_q95 = self._analisador_ela.calcular_ela(imagem_rgb, qualidade_jpeg=95)
            mapa_q75 = self._analisador_ela.calcular_ela(imagem_rgb, qualidade_jpeg=75)
            score_ela = self._analisador_ela.calcular_score_ela(imagem_rgb)
        except Exception as erro:
            logger.error("Erro ao calcular ELA: %s", erro, exc_info=True)
            mapa_q95 = np.zeros(imagem_rgb.shape[:2], dtype=np.float32)
            mapa_q75 = np.zeros(imagem_rgb.shape[:2], dtype=np.float32)
            score_ela = 0.5

        figura, eixos = plt.subplots(1, 3, figsize=(14, 4), dpi=DPI_PADRAO)
        figura.patch.set_facecolor(COR_FUNDO)

        eixos[0].imshow(imagem_rgb)
        eixos[0].set_title("Imagem Original", color=COR_TEXTO, fontsize=11, pad=6)
        eixos[0].axis("off")

        img_q95 = eixos[1].imshow(mapa_q95, cmap="inferno", vmin=0, vmax=1)
        eixos[1].set_title("ELA (Q=95%)", color=COR_TEXTO, fontsize=11, pad=6)
        eixos[1].axis("off")
        barra_q95 = figura.colorbar(img_q95, ax=eixos[1], fraction=0.046, pad=0.04)
        barra_q95.set_label("Nivel de Erro", color=COR_TEXTO, fontsize=9)
        plt.setp(barra_q95.ax.yaxis.get_ticklabels(), color=COR_TEXTO)
        barra_q95.ax.yaxis.set_tick_params(color=COR_TEXTO)

        img_q75 = eixos[2].imshow(mapa_q75, cmap="inferno", vmin=0, vmax=1)
        eixos[2].set_title("ELA (Q=75%)", color=COR_TEXTO, fontsize=11, pad=6)
        eixos[2].axis("off")
        barra_q75 = figura.colorbar(img_q75, ax=eixos[2], fraction=0.046, pad=0.04)
        barra_q75.set_label("Nivel de Erro", color=COR_TEXTO, fontsize=9)
        plt.setp(barra_q75.ax.yaxis.get_ticklabels(), color=COR_TEXTO)
        barra_q75.ax.yaxis.set_tick_params(color=COR_TEXTO)

        interpretacao = "Provavel IA" if score_ela > 0.6 else "Provavel Real" if score_ela < 0.4 else "Inconclusivo"
        figura.suptitle(
            f"Error Level Analysis — Score ELA: {score_ela:.2f} ({interpretacao})",
            color=COR_TEXTO, fontsize=13, fontweight="bold", y=1.02,
        )
        figura.tight_layout()

        return figura

    def criar_visualizacao_consistencia_ruido(
        self,
        imagem: np.ndarray,
    ) -> matplotlib.figure.Figure:
        """
        Cria visualizacao do mapa de inconsistencia de ruido.

        Layout 1x2: Original | Mapa de Inconsistencia.
        Colormap YlOrRd — amarelo = consistente, vermelho = inconsistente.

        Args:
            imagem: Imagem original como array numpy (H, W, 3) uint8 RGB.

        Returns:
            Figura Matplotlib com mapa de inconsistencia de ruido.
        """
        imagem_rgb = _garantir_uint8_rgb(imagem)

        try:
            mapa_inconsistencia = self._analisador_pixels.calcular_mapa_inconsistencia(
                imagem_rgb
            )
            pct_inconsistente = float((mapa_inconsistencia > 0.5).mean()) * 100
        except Exception as erro:
            logger.error("Erro ao calcular inconsistencia de ruido: %s", erro, exc_info=True)
            mapa_inconsistencia = np.zeros(imagem_rgb.shape[:2], dtype=np.float32)
            pct_inconsistente = 0.0

        figura, eixos = plt.subplots(1, 2, figsize=(10, 4), dpi=DPI_PADRAO)
        figura.patch.set_facecolor(COR_FUNDO)

        eixos[0].imshow(imagem_rgb)
        eixos[0].set_title("Imagem Original", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[0].axis("off")

        img_incons = eixos[1].imshow(mapa_inconsistencia, cmap="YlOrRd", vmin=0, vmax=1)
        eixos[1].set_title("Mapa de Inconsistencia de Ruido", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[1].axis("off")

        barra = figura.colorbar(img_incons, ax=eixos[1], fraction=0.046, pad=0.04)
        barra.set_label("Inconsistencia", color=COR_TEXTO, fontsize=10)
        plt.setp(barra.ax.yaxis.get_ticklabels(), color=COR_TEXTO)
        barra.ax.yaxis.set_tick_params(color=COR_TEXTO)

        figura.text(
            0.75, -0.02,
            f"Regioes inconsistentes: {pct_inconsistente:.1f}%",
            ha="center", color="#aaaaaa", fontsize=9,
            transform=figura.transFigure,
        )

        figura.suptitle(
            "Consistencia de Ruido",
            color=COR_TEXTO, fontsize=14, fontweight="bold", y=1.02,
        )
        figura.tight_layout()

        return figura

    def criar_visualizacao_regioes_suspeitas(
        self,
        imagem: np.ndarray,
        mapa_calor_gradcam: np.ndarray | None = None,
    ) -> matplotlib.figure.Figure:
        """
        Cria visualizacao principal de fusao de multiplos sinais forenses.

        Funde ELA, inconsistencia de ruido e GradCAM (se disponivel)
        em um unico overlay semi-transparente sobre a imagem original.
        Contornos destacam regioes com score > 0.7.

        Pesos: 40% ELA + 35% ruido + 25% GradCAM.

        Args:
            imagem: Imagem original como array numpy (H, W, 3) uint8 RGB.
            mapa_calor_gradcam: Mapa GradCAM (H, W) normalizado [0, 1], opcional.

        Returns:
            Figura Matplotlib com overlay de regioes suspeitas.
        """
        imagem_rgb = _garantir_uint8_rgb(imagem)
        altura, largura = imagem_rgb.shape[:2]

        # Calcula mapas individuais
        try:
            mapa_ela = self._analisador_ela.calcular_ela(imagem_rgb, qualidade_jpeg=90)
        except Exception:
            mapa_ela = np.zeros((altura, largura), dtype=np.float32)

        try:
            mapa_ruido = self._analisador_pixels.calcular_mapa_inconsistencia(imagem_rgb)
        except Exception:
            mapa_ruido = np.zeros((altura, largura), dtype=np.float32)

        # Garante dimensoes consistentes
        if mapa_ela.shape != (altura, largura):
            mapa_ela = cv2.resize(mapa_ela, (largura, altura), interpolation=cv2.INTER_LINEAR)
        if mapa_ruido.shape != (altura, largura):
            mapa_ruido = cv2.resize(mapa_ruido, (largura, altura), interpolation=cv2.INTER_LINEAR)

        # Fusao ponderada
        if mapa_calor_gradcam is not None:
            if mapa_calor_gradcam.shape != (altura, largura):
                mapa_gradcam = cv2.resize(
                    mapa_calor_gradcam.astype(np.float32),
                    (largura, altura),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                mapa_gradcam = mapa_calor_gradcam.astype(np.float32)
            mapa_fundido = (
                0.40 * mapa_ela + 0.35 * mapa_ruido + 0.25 * mapa_gradcam
            )
        else:
            # Sem GradCAM: redistribui os pesos
            mapa_fundido = 0.55 * mapa_ela + 0.45 * mapa_ruido

        mapa_fundido = np.clip(mapa_fundido, 0.0, 1.0)

        # Score combinado
        score_combinado = float(mapa_fundido.mean())

        # Overlay semi-transparente
        mapa_uint8 = np.uint8(255 * mapa_fundido)
        mapa_colorido_bgr = cv2.applyColorMap(mapa_uint8, cv2.COLORMAP_HOT)
        mapa_colorido_rgb = cv2.cvtColor(mapa_colorido_bgr, cv2.COLOR_BGR2RGB)

        alfa = 0.4
        sobreposicao = cv2.addWeighted(
            imagem_rgb.astype(np.float32), 1.0 - alfa,
            mapa_colorido_rgb.astype(np.float32), alfa,
            0,
        )
        sobreposicao = np.clip(sobreposicao, 0, 255).astype(np.uint8)

        # Contornos nas regioes com score > 0.7
        mascara_alta = (mapa_fundido > 0.7).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara_alta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            sobreposicao = cv2.drawContours(sobreposicao, contornos, -1, (255, 0, 0), 2)

        figura, eixos = plt.subplots(1, 2, figsize=(11, 5), dpi=DPI_PADRAO)
        figura.patch.set_facecolor(COR_FUNDO)

        eixos[0].imshow(imagem_rgb)
        eixos[0].set_title("Imagem Original", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[0].axis("off")

        eixos[1].imshow(sobreposicao)
        eixos[1].set_title("Regioes Suspeitas (Fusao)", color=COR_TEXTO, fontsize=12, pad=8)
        eixos[1].axis("off")

        # Barra de cores
        mapa_normalizado = plt.cm.ScalarMappable(
            cmap="hot", norm=plt.Normalize(vmin=0, vmax=1)
        )
        mapa_normalizado.set_array([])
        barra = figura.colorbar(mapa_normalizado, ax=eixos[1], fraction=0.046, pad=0.04)
        barra.set_label("Suspeita", color=COR_TEXTO, fontsize=10)
        plt.setp(barra.ax.yaxis.get_ticklabels(), color=COR_TEXTO)
        barra.ax.yaxis.set_tick_params(color=COR_TEXTO)

        componentes = "ELA + Ruido"
        if mapa_calor_gradcam is not None:
            componentes += " + GradCAM"

        figura.suptitle(
            f"Regioes Suspeitas — Score: {score_combinado:.2f} ({componentes})",
            color=COR_TEXTO, fontsize=13, fontweight="bold", y=1.02,
        )
        figura.tight_layout()

        return figura

    def criar_visualizacao_histograma_rgb(
        self,
        imagem: np.ndarray,
    ) -> matplotlib.figure.Figure:
        """
        Cria grafico com histogramas RGB sobrepostos e preenchidos por canal.

        Cada canal e plotado com sua cor caracteristica (vermelho, verde, azul)
        com preenchimento translucido para facilitar a leitura da sobreposicao.
        Util para identificar anomalias de distribuicao de cor em imagens geradas.

        Args:
            imagem: Imagem como array numpy (H, W, 3) uint8 RGB.

        Returns:
            Figura Matplotlib com histogramas RGB sobrepostos.
        """
        imagem_rgb = _garantir_uint8_rgb(imagem)

        try:
            histogramas = self._analisador_pixels.calcular_histograma_rgb(imagem_rgb)
        except Exception as erro:
            logger.error("Erro ao calcular histograma RGB: %s", erro, exc_info=True)
            histogramas = {
                "vermelho": np.zeros(256),
                "verde": np.zeros(256),
                "azul": np.zeros(256),
            }

        figura, eixo = plt.subplots(1, 1, figsize=(8, 4), dpi=DPI_PADRAO)
        figura.patch.set_facecolor(COR_FUNDO)
        eixo.set_facecolor("#16213e")

        # Configuracoes de cada canal: (chave_dict, cor_linha, cor_fill, rotulo)
        configuracoes_canais: list[tuple[str, str, str, str]] = [
            ("vermelho", "#ff4444", "rgba(255,68,68,0.15)", "Canal Vermelho (R)"),
            ("verde", "#44ff88", "rgba(68,255,136,0.15)", "Canal Verde (G)"),
            ("azul", "#4488ff", "rgba(68,136,255,0.15)", "Canal Azul (B)"),
        ]

        intensidades = np.arange(256)

        for chave, cor_linha, _, rotulo in configuracoes_canais:
            valores = histogramas.get(chave, np.zeros(256))
            eixo.plot(
                intensidades, valores,
                color=cor_linha, linewidth=1.5, label=rotulo, alpha=0.9,
            )
            eixo.fill_between(
                intensidades, valores, alpha=0.15, color=cor_linha
            )

        eixo.set_title(
            "Distribuicao de Intensidade por Canal RGB",
            color=COR_TEXTO, fontsize=13, pad=10,
        )
        eixo.set_xlabel("Intensidade (0-255)", color=COR_TEXTO, fontsize=10)
        eixo.set_ylabel("Frequencia Relativa", color=COR_TEXTO, fontsize=10)
        eixo.set_xlim(0, 255)
        eixo.set_ylim(bottom=0)
        eixo.grid(True, alpha=0.3, color=COR_EIXO)
        eixo.legend(
            fontsize=10, facecolor="#16213e", edgecolor=COR_EIXO, labelcolor=COR_TEXTO
        )

        figura.tight_layout()

        return figura
