"""
Visualizacoes de timeline para analise frame a frame de videos.

Combina Plotly (graficos interativos) e Matplotlib (grade de frames) para
apresentar a evolucao temporal dos scores de deteccao de IA ao longo do video.

Estrutura esperada de cada entrada em `timeline`:
    {
        "indice_frame": int,     # Posicao do frame no video
        "score": float,          # Score de IA entre 0.0 e 1.0
        "classificacao": str,    # Texto da classificacao
        "tem_rosto": bool,       # Se rosto foi detectado no frame
        "num_rostos": int,       # Quantidade de rostos detectados
    }
"""

from __future__ import annotations

import logging
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")  # Backend nao-interativo
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import CONFIG_ENSEMBLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de estilo compartilhadas
# ---------------------------------------------------------------------------

THRESHOLD_REAL: float = CONFIG_ENSEMBLE.thresholds["provavelmente_real"]
THRESHOLD_POSS_REAL: float = CONFIG_ENSEMBLE.thresholds["possivelmente_real"]
THRESHOLD_POSS_IA: float = CONFIG_ENSEMBLE.thresholds["possivelmente_ia"]

COR_VERDE: str = "#2ecc71"
COR_VERDE_CLARO: str = "#82e0aa"
COR_LARANJA: str = "#f39c12"
COR_VERMELHO: str = "#e74c3c"
COR_CINZA: str = "#808080"

TEMPLATE_PLOTLY: str = "plotly_dark"

# Estilo Matplotlib
COR_FUNDO_MPL: str = "#1a1a2e"
COR_TEXTO_MPL: str = "#e0e0e0"
COR_EIXO_MPL: str = "#555577"
DPI_PADRAO: int = 100


# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------


class EntradaTimeline(TypedDict):
    """Estrutura de cada entrada na lista de timeline de frames."""

    indice_frame: int
    score: float
    classificacao: str
    tem_rosto: bool
    num_rostos: int


# ---------------------------------------------------------------------------
# Funcoes auxiliares internas
# ---------------------------------------------------------------------------


def _cor_para_score(score: float) -> str:
    """Retorna a cor hex da faixa de classificacao do score [0, 1]."""
    if score < THRESHOLD_REAL:
        return COR_VERDE
    if score < THRESHOLD_POSS_REAL:
        return COR_VERDE_CLARO
    if score < THRESHOLD_POSS_IA:
        return COR_LARANJA
    return COR_VERMELHO


def _classificacao_para_cor_rgba(score: float, opacidade: float = 0.3) -> str:
    """Retorna cor RGBA como string CSS para regioes de fundo do Plotly."""
    cor_hex = _cor_para_score(score)
    # Converte hex para r,g,b
    r = int(cor_hex[1:3], 16)
    g = int(cor_hex[3:5], 16)
    b = int(cor_hex[5:7], 16)
    return f"rgba({r},{g},{b},{opacidade})"


def _figura_plotly_vazia(mensagem: str) -> go.Figure:
    """Retorna uma figura Plotly vazia com mensagem de aviso."""
    figura = go.Figure()
    figura.add_annotation(
        text=mensagem, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font={"size": 14, "color": "#888888"},
    )
    figura.update_layout(
        template=TEMPLATE_PLOTLY, height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"}, margin={"t": 30, "b": 30, "l": 30, "r": 30},
    )
    return figura


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------


class TimelineVideo:
    """
    Gerador de visualizacoes temporais para analise de video frame a frame.

    Metodos que retornam go.Figure sao usados com st.plotly_chart().
    Metodos que retornam matplotlib.figure.Figure sao usados com st.pyplot().
    """

    @staticmethod
    def criar_timeline_scores(timeline: list[EntradaTimeline]) -> go.Figure:
        """
        Cria grafico de linha interativo com a evolucao dos scores ao longo
        dos frames analisados.

        O grafico exibe:
        - Linha de score com coloracao continua por faixa de classificacao.
        - Marcadores especiais (estrela) nos frames onde rostos foram detectados.
        - Linhas horizontais tracejadas nos thresholds de classificacao.
        - Regioes de fundo coloridas por faixa (verde/amarelo/laranja/vermelho).

        Args:
            timeline: Lista de dicionarios com dados por frame. Cada item deve
                      conter: indice_frame, score, classificacao, tem_rosto,
                      num_rostos.

        Returns:
            Figura Plotly interativa com a timeline de scores.
        """
        if not timeline:
            return _figura_plotly_vazia(
                "Nenhum frame disponivel para exibir na timeline"
            )

        indices = [entrada["indice_frame"] for entrada in timeline]
        scores_pct = [entrada["score"] * 100 for entrada in timeline]
        classificacoes = [entrada["classificacao"] for entrada in timeline]
        tem_rosto = [entrada.get("tem_rosto", False) for entrada in timeline]
        num_rostos = [entrada.get("num_rostos", 0) for entrada in timeline]
        cores_pontos = [_cor_para_score(entrada["score"]) for entrada in timeline]

        figura = go.Figure()

        # Regioes de fundo por faixa de classificacao
        regioes_fundo = [
            (0, THRESHOLD_REAL * 100, COR_VERDE, "Real"),
            (THRESHOLD_REAL * 100, THRESHOLD_POSS_REAL * 100, COR_VERDE_CLARO, "Pos. Real"),
            (THRESHOLD_POSS_REAL * 100, THRESHOLD_POSS_IA * 100, COR_LARANJA, "Pos. IA"),
            (THRESHOLD_POSS_IA * 100, 100, COR_VERMELHO, "Provavelmente IA"),
        ]
        for y_inicio, y_fim, cor, nome_faixa in regioes_fundo:
            figura.add_hrect(
                y0=y_inicio, y1=y_fim,
                fillcolor=cor, opacity=0.06,
                layer="below", line_width=0,
                annotation_text=nome_faixa,
                annotation_font_size=10,
                annotation_font_color="#666666",
                annotation_position="right",
            )

        # Linha de score continua
        figura.add_trace(
            go.Scatter(
                x=indices,
                y=scores_pct,
                mode="lines",
                name="Score de IA",
                line={"color": "#00aaff", "width": 2},
                hovertemplate=(
                    "<b>Frame %{x}</b><br>"
                    "Score: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Pontos coloridos por classificacao
        figura.add_trace(
            go.Scatter(
                x=indices,
                y=scores_pct,
                mode="markers",
                name="Frames analisados",
                marker={
                    "color": cores_pontos,
                    "size": 7,
                    "line": {"color": "#222222", "width": 0.5},
                },
                customdata=list(zip(classificacoes, num_rostos)),
                hovertemplate=(
                    "<b>Frame %{x}</b><br>"
                    "Score: %{y:.1f}%<br>"
                    "Classificacao: %{customdata[0]}<br>"
                    "Rostos detectados: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Marcadores especiais para frames com rostos detectados
        indices_com_rosto = [idx for idx, r in zip(indices, tem_rosto) if r]
        scores_com_rosto = [
            score for score, r in zip(scores_pct, tem_rosto) if r
        ]

        if indices_com_rosto:
            figura.add_trace(
                go.Scatter(
                    x=indices_com_rosto,
                    y=scores_com_rosto,
                    mode="markers",
                    name="Rosto detectado",
                    marker={
                        "symbol": "star",
                        "size": 14,
                        "color": "#ffdd44",
                        "line": {"color": "#ffffff", "width": 1},
                    },
                    hovertemplate=(
                        "<b>Frame %{x}</b> — Rosto detectado<br>"
                        "Score: %{y:.1f}%<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # Linhas de threshold como referencias horizontais
        for valor_thr, nome_thr, cor_thr in [
            (THRESHOLD_REAL * 100, "Real / Pos. Real", COR_VERDE_CLARO),
            (THRESHOLD_POSS_REAL * 100, "Pos. Real / Pos. IA", COR_LARANJA),
            (THRESHOLD_POSS_IA * 100, "Pos. IA / Provavel IA", COR_VERMELHO),
        ]:
            figura.add_hline(
                y=valor_thr,
                line_dash="dot",
                line_color=cor_thr,
                line_width=1.2,
                annotation_text=f" {nome_thr} ({valor_thr:.0f}%)",
                annotation_font_size=10,
                annotation_font_color=cor_thr,
                annotation_position="left",
            )

        figura.update_layout(
            title={
                "text": "Timeline de Deteccao - Frame a Frame",
                "font": {"size": 16},
                "x": 0.5,
                "xanchor": "center",
            },
            template=TEMPLATE_PLOTLY,
            xaxis={
                "title": "Indice do Frame",
                "showgrid": True,
                "gridcolor": "#333344",
            },
            yaxis={
                "title": "Score de IA (%)",
                "range": [-2, 105],
                "ticksuffix": "%",
                "showgrid": True,
                "gridcolor": "#333344",
            },
            height=420,
            margin={"t": 60, "b": 60, "l": 80, "r": 120},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.25,
                "xanchor": "center",
                "x": 0.5,
            },
            hovermode="x unified",
        )

        return figura

    @staticmethod
    def criar_grid_frames_suspeitos(
        frames_suspeitos: list[dict],
        frames_imagens: dict[int, np.ndarray] | None = None,
    ) -> matplotlib.figure.Figure:
        """
        Cria grade com os frames mais suspeitos do video, ordenados por score.

        Exibe ate 10 frames em uma grade de 2 linhas por 5 colunas. Cada
        frame mostra o score sobreposto e a classificacao abaixo. Se as
        imagens dos frames nao forem fornecidas, exibe placeholders coloridos.

        Args:
            frames_suspeitos: Lista de dicionarios com dados dos frames
                              suspeitos. Campos esperados: indice_frame, score,
                              classificacao, tem_rosto, num_rostos. Ja deve
                              estar ordenada por score decrescente.
            frames_imagens: Mapa opcional {indice_frame: array_numpy_rgb}
                            com as imagens dos frames suspeitos.

        Returns:
            Figura Matplotlib com a grade de frames suspeitos.
        """
        if not frames_suspeitos:
            figura, eixo = plt.subplots(1, 1, figsize=(8, 3), dpi=DPI_PADRAO)
            figura.patch.set_facecolor(COR_FUNDO_MPL)
            eixo.set_facecolor(COR_FUNDO_MPL)
            eixo.text(
                0.5, 0.5,
                "Nenhum frame suspeito identificado",
                ha="center", va="center",
                color="#888888", fontsize=13,
                transform=eixo.transAxes,
            )
            eixo.axis("off")
            figura.tight_layout()
            return figura

        # Limita a 10 frames; ordena por score decrescente (mais suspeitos primeiro)
        frames_exibir = sorted(
            frames_suspeitos, key=lambda f: f["score"], reverse=True
        )[:10]

        num_frames = len(frames_exibir)
        num_colunas = min(5, num_frames)
        num_linhas = max(1, (num_frames + num_colunas - 1) // num_colunas)

        plt.rcParams.update({
            "figure.facecolor": COR_FUNDO_MPL,
            "axes.facecolor": "#16213e",
            "text.color": COR_TEXTO_MPL,
        })

        figura, eixos = plt.subplots(
            num_linhas,
            num_colunas,
            figsize=(num_colunas * 3.2, num_linhas * 3.8),
            dpi=DPI_PADRAO,
        )
        figura.patch.set_facecolor(COR_FUNDO_MPL)

        # Normaliza eixos para array 2D sempre
        if num_linhas == 1 and num_colunas == 1:
            eixos = np.array([[eixos]])
        elif num_linhas == 1:
            eixos = np.array([eixos])
        elif num_colunas == 1:
            eixos = np.array([[e] for e in eixos])

        for posicao, dados_frame in enumerate(frames_exibir):
            linha = posicao // num_colunas
            coluna = posicao % num_colunas
            eixo = eixos[linha, coluna]

            indice_frame: int = dados_frame["indice_frame"]
            score: float = dados_frame["score"]
            classificacao: str = dados_frame.get("classificacao", "")
            tem_rosto: bool = dados_frame.get("tem_rosto", False)

            cor_score = _cor_para_score(score)

            # Exibe imagem real ou placeholder colorido
            if frames_imagens and indice_frame in frames_imagens:
                imagem = frames_imagens[indice_frame]
                if imagem.dtype != np.uint8:
                    imagem = np.clip(imagem, 0, 255).astype(np.uint8)
                eixo.imshow(imagem)
            else:
                # Placeholder: fundo colorido proporcional ao score
                r = int(cor_score[1:3], 16) / 255
                g = int(cor_score[3:5], 16) / 255
                b = int(cor_score[5:7], 16) / 255
                placeholder = np.ones((64, 64, 3)) * [r, g, b]
                placeholder = (placeholder * 0.25 + 0.1)  # Escurece
                eixo.imshow(np.clip(placeholder, 0, 1))
                eixo.text(
                    0.5, 0.5,
                    f"Frame {indice_frame}\n(sem imagem)",
                    ha="center", va="center",
                    color=COR_TEXTO_MPL, fontsize=8,
                    transform=eixo.transAxes,
                )

            # Score sobreposto na parte superior da imagem
            eixo.text(
                0.05, 0.95,
                f"{score * 100:.1f}%",
                ha="left", va="top",
                color="#ffffff",
                fontsize=11, fontweight="bold",
                transform=eixo.transAxes,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": cor_score,
                    "alpha": 0.85,
                    "edgecolor": "none",
                },
            )

            # Icone de rosto se detectado
            if tem_rosto:
                eixo.text(
                    0.95, 0.95,
                    "rosto",
                    ha="right", va="top",
                    color="#ffdd44",
                    fontsize=8,
                    transform=eixo.transAxes,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "#222222",
                        "alpha": 0.8,
                        "edgecolor": "#ffdd44",
                    },
                )

            titulo_frame = f"Frame {indice_frame}"
            eixo.set_title(titulo_frame, color=COR_TEXTO_MPL, fontsize=9, pad=4)
            eixo.set_xlabel(
                classificacao,
                color=cor_score, fontsize=8, labelpad=3,
            )
            eixo.set_xticks([])
            eixo.set_yticks([])
            for borda in eixo.spines.values():
                borda.set_edgecolor(cor_score)
                borda.set_linewidth(2)

        # Oculta eixos extras da grade
        for posicao_extra in range(num_frames, num_linhas * num_colunas):
            linha = posicao_extra // num_colunas
            coluna = posicao_extra % num_colunas
            eixos[linha, coluna].axis("off")
            eixos[linha, coluna].set_facecolor(COR_FUNDO_MPL)

        figura.suptitle(
            f"Frames Mais Suspeitos (Top {num_frames})",
            color=COR_TEXTO_MPL, fontsize=14, fontweight="bold", y=1.01,
        )
        figura.tight_layout(pad=0.8)

        return figura

    @staticmethod
    def criar_resumo_video(
        timeline: list[EntradaTimeline],
        total_frames: int,
    ) -> go.Figure:
        """
        Cria painel de resumo estatistico do video com indicadores e tabela.

        Exibe uma grade de indicadores (go.Indicator) com:
        - Percentual de frames classificados como cada categoria.
        - Score medio, minimo e maximo.
        - Total de frames com rostos detectados.

        Args:
            timeline: Lista de entradas da timeline (pode estar vazia).
            total_frames: Numero total de frames analisados no video.

        Returns:
            Figura Plotly com grade de indicadores de resumo.
        """
        if not timeline:
            return _figura_plotly_vazia("Sem dados de timeline para gerar resumo")

        # Calculos do resumo
        scores = [e["score"] for e in timeline]
        score_medio = float(np.mean(scores))
        score_minimo = float(np.min(scores))
        score_maximo = float(np.max(scores))

        frames_com_rosto = sum(1 for e in timeline if e.get("tem_rosto", False))
        pct_rosto = (frames_com_rosto / len(timeline) * 100) if timeline else 0.0

        # Contagem por faixa de classificacao
        def _contar_faixa(limite_inf: float, limite_sup: float) -> int:
            return sum(1 for s in scores if limite_inf <= s < limite_sup)

        qtd_real = _contar_faixa(0.0, THRESHOLD_REAL)
        qtd_poss_real = _contar_faixa(THRESHOLD_REAL, THRESHOLD_POSS_REAL)
        qtd_poss_ia = _contar_faixa(THRESHOLD_POSS_REAL, THRESHOLD_POSS_IA)
        qtd_provavel_ia = _contar_faixa(THRESHOLD_POSS_IA, 1.01)

        n_frames = len(timeline)
        pct_real = qtd_real / n_frames * 100
        pct_poss_real = qtd_poss_real / n_frames * 100
        pct_poss_ia = qtd_poss_ia / n_frames * 100
        pct_provavel_ia = qtd_provavel_ia / n_frames * 100

        # Grade de 3 linhas x 3 colunas de indicadores
        figura = make_subplots(
            rows=3, cols=3,
            specs=[[{"type": "indicator"}] * 3] * 3,
        )

        def _adicionar_indicador(
            linha: int,
            coluna: int,
            valor: float,
            titulo: str,
            sufixo: str = "%",
            cor: str = "#00d4ff",
            referencia: float | None = None,
        ) -> None:
            """Insere um go.Indicator na posicao (linha, coluna) da grade."""
            delta_config: dict | None = None
            if referencia is not None:
                delta_config = {
                    "reference": referencia,
                    "valueformat": ".1f",
                    "suffix": sufixo,
                    "increasing": {"color": COR_VERMELHO},
                    "decreasing": {"color": COR_VERDE},
                }

            figura.add_trace(
                go.Indicator(
                    mode="number+delta" if referencia is not None else "number",
                    value=round(valor, 1),
                    number={"suffix": sufixo, "font": {"size": 28, "color": cor}},
                    delta=delta_config,
                    title={
                        "text": titulo,
                        "font": {"size": 12, "color": "#aaaaaa"},
                    },
                ),
                row=linha,
                col=coluna,
            )

        # Linha 1: score medio, minimo, maximo
        _adicionar_indicador(1, 1, score_medio * 100, "Score Medio", cor=_cor_para_score(score_medio))
        _adicionar_indicador(1, 2, score_minimo * 100, "Score Minimo", cor=COR_VERDE)
        _adicionar_indicador(1, 3, score_maximo * 100, "Score Maximo", cor=_cor_para_score(score_maximo))

        # Linha 2: percentuais de classificacao por faixa
        _adicionar_indicador(2, 1, pct_real, "Provavelmente Real", cor=COR_VERDE)
        _adicionar_indicador(2, 2, pct_poss_real + pct_poss_ia, "Zona de Incerteza", cor=COR_LARANJA)
        _adicionar_indicador(2, 3, pct_provavel_ia, "Provavelmente IA", cor=COR_VERMELHO)

        # Linha 3: frames totais, frames com rosto, indice temporal
        _adicionar_indicador(3, 1, float(n_frames), "Frames Analisados", sufixo="", cor="#00d4ff")
        _adicionar_indicador(3, 2, pct_rosto, "Frames com Rosto", cor="#ffdd44")
        _adicionar_indicador(
            3, 3,
            float(total_frames),
            "Total de Frames do Video",
            sufixo="",
            cor="#8888ff",
        )

        figura.update_layout(
            title={
                "text": "Resumo da Analise do Video",
                "font": {"size": 17},
                "x": 0.5,
                "xanchor": "center",
            },
            template=TEMPLATE_PLOTLY,
            height=420,
            margin={"t": 70, "b": 20, "l": 20, "r": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
        )

        return figura

