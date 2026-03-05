"""
Graficos interativos Plotly para o dashboard de deteccao de conteudo gerado por IA.

Todos os graficos sao retornados como go.Figure do Plotly, prontos para
uso via API REST ou frontend React.
"""

from __future__ import annotations

import logging

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import CONFIG_ENSEMBLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de apresentacao
# ---------------------------------------------------------------------------

# Mapeamento de IDs internos para nomes amigaveis ao usuario
NOMES_AMIGAVEIS_MODELOS: dict[str, str] = {
    "spatial_vit": "ViT Espacial",
    "clip_detector": "CLIP",
    "frequency": "Frequencia FFT",
    "frequency_analyzer": "Frequencia FFT",
    "pixel_stats": "Estatisticas de Pixels",
    "efficientnet_video": "EfficientNet Video",
}

# Thresholds de classificacao (extraidos de CONFIG_ENSEMBLE para reuso)
THRESHOLD_REAL: float = CONFIG_ENSEMBLE.thresholds["provavelmente_real"]
THRESHOLD_POSS_REAL: float = CONFIG_ENSEMBLE.thresholds["possivelmente_real"]
THRESHOLD_POSS_IA: float = CONFIG_ENSEMBLE.thresholds["possivelmente_ia"]

# Paleta de cores das faixas de classificacao
COR_VERDE: str = "#2ecc71"
COR_VERDE_CLARO: str = "#82e0aa"
COR_LARANJA: str = "#f39c12"
COR_VERMELHO: str = "#e74c3c"
COR_CINZA: str = "#808080"

# Tema base dos graficos
TEMPLATE_PLOTLY: str = "plotly_dark"


def _cor_para_score(score_normalizado: float) -> str:
    """
    Retorna a cor hex correspondente a um score normalizado [0, 1].

    Args:
        score_normalizado: Score entre 0.0 e 1.0.

    Returns:
        Cor hex da faixa de classificacao do score.
    """
    if score_normalizado < THRESHOLD_REAL:
        return COR_VERDE
    if score_normalizado < THRESHOLD_POSS_REAL:
        return COR_VERDE_CLARO
    if score_normalizado < THRESHOLD_POSS_IA:
        return COR_LARANJA
    return COR_VERMELHO


def _nome_amigavel(id_modelo: str, nomes_externos: dict[str, str] | None = None) -> str:
    """
    Resolve o nome de exibicao de um modelo.

    Prioridade: nomes_externos > NOMES_AMIGAVEIS_MODELOS > id_modelo.

    Args:
        id_modelo: Identificador interno do modelo.
        nomes_externos: Mapa opcional fornecido pelo chamador.

    Returns:
        Nome de exibicao para o modelo.
    """
    if nomes_externos and id_modelo in nomes_externos:
        return nomes_externos[id_modelo]
    return NOMES_AMIGAVEIS_MODELOS.get(id_modelo, id_modelo.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------


class GraficosDeteccao:
    """
    Fabrica de graficos Plotly para o dashboard de deteccao de IA.

    Todos os metodos retornam go.Figure prontos para uso via
    API REST ou frontend React.
    """

    @staticmethod
    def criar_gauge_confianca(
        score: float,
        classificacao: str,
        cor: str,
    ) -> go.Figure:
        """
        Cria um gauge (velocimetro) interativo mostrando o score de confianca.

        O ponteiro indica o score em percentual (0-100%). As faixas de cor
        do gauge refletem as quatro classificacoes possiveis: verde para
        provavel real, verde-claro para possivelmente real, laranja para
        possivelmente IA e vermelho para provavel IA.

        Args:
            score: Score final do ensemble entre 0.0 e 1.0.
            classificacao: Texto da classificacao (ex: "Provavelmente IA").
            cor: Cor hex principal do resultado (usada no delta indicator).

        Returns:
            Figura Plotly com o gauge de confianca.
        """
        score_pct: float = round(score * 100, 1)

        figura = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=score_pct,
                number={
                    "suffix": "%",
                    "font": {"size": 36, "color": cor},
                },
                delta={
                    "reference": 50,
                    "valueformat": ".1f",
                    "suffix": "%",
                    "increasing": {"color": COR_VERMELHO},
                    "decreasing": {"color": COR_VERDE},
                },
                title={
                    "text": (
                        f"<b>Score de Confianca</b><br>"
                        f"<span style='font-size:14px;color:{cor}'>"
                        f"{classificacao}</span>"
                    ),
                    "font": {"size": 18},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#aaaaaa",
                        "tickfont": {"size": 11},
                        "tickvals": [0, 25, 50, 75, 100],
                        "ticktext": ["0%", "25%", "50%", "75%", "100%"],
                    },
                    "bar": {"color": cor, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 2,
                    "bordercolor": "#444444",
                    "steps": [
                        {
                            "range": [0, THRESHOLD_REAL * 100],
                            "color": "rgba(46, 204, 113, 0.25)",
                        },
                        {
                            "range": [
                                THRESHOLD_REAL * 100,
                                THRESHOLD_POSS_REAL * 100,
                            ],
                            "color": "rgba(130, 224, 170, 0.25)",
                        },
                        {
                            "range": [
                                THRESHOLD_POSS_REAL * 100,
                                THRESHOLD_POSS_IA * 100,
                            ],
                            "color": "rgba(243, 156, 18, 0.25)",
                        },
                        {
                            "range": [THRESHOLD_POSS_IA * 100, 100],
                            "color": "rgba(231, 76, 60, 0.25)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": cor, "width": 4},
                        "thickness": 0.75,
                        "value": score_pct,
                    },
                },
            )
        )

        figura.update_layout(
            template=TEMPLATE_PLOTLY,
            height=300,
            margin={"t": 60, "b": 20, "l": 20, "r": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
        )

        return figura

    @staticmethod
    def criar_barras_modelos(
        scores_individuais: dict[str, float],
        nomes_modelos: dict[str, str] | None = None,
    ) -> go.Figure:
        """
        Cria um grafico de barras horizontais com o score de cada modelo.

        As barras sao coloridas do verde ao vermelho conforme o score,
        e exibem o percentual dentro da barra para leitura rapida.
        Modelos com scores ausentes sao ignorados graciosamente.

        Args:
            scores_individuais: Mapa {id_modelo: score} com valores em [0, 1].
            nomes_modelos: Mapa opcional de nomes de exibicao personalizados.

        Returns:
            Figura Plotly com barras horizontais por modelo.
        """
        if not scores_individuais:
            return _figura_vazia("Nenhum modelo disponivel")

        # Ordena por score decrescente para hierarquia visual clara
        itens_ordenados = sorted(
            scores_individuais.items(), key=lambda par: par[1], reverse=True
        )

        nomes_exibicao = [
            _nome_amigavel(id_modelo, nomes_modelos)
            for id_modelo, _ in itens_ordenados
        ]
        valores_pct = [round(score * 100, 1) for _, score in itens_ordenados]
        cores = [_cor_para_score(score) for _, score in itens_ordenados]
        rotulos = [f"{v:.1f}%" for v in valores_pct]

        figura = go.Figure(
            go.Bar(
                x=valores_pct,
                y=nomes_exibicao,
                orientation="h",
                marker={"color": cores, "line": {"color": "#333333", "width": 1}},
                text=rotulos,
                textposition="inside",
                textfont={"size": 13, "color": "#ffffff", "family": "monospace"},
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Score: %{x:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

        figura.update_layout(
            title={
                "text": "Score por Modelo",
                "font": {"size": 16},
                "x": 0.5,
                "xanchor": "center",
            },
            template=TEMPLATE_PLOTLY,
            xaxis={
                "title": "Score de IA (%)",
                "range": [0, 105],
                "ticksuffix": "%",
            },
            yaxis={"title": "Modelo", "autorange": "reversed"},
            height=max(250, len(scores_individuais) * 60 + 120),
            margin={"t": 60, "b": 50, "l": 160, "r": 30},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
            showlegend=False,
        )

        # Linhas de threshold como referencia visual
        for valor_threshold, nome_threshold in [
            (THRESHOLD_REAL * 100, "Real"),
            (THRESHOLD_POSS_REAL * 100, "Pos. Real"),
            (THRESHOLD_POSS_IA * 100, "Pos. IA"),
        ]:
            figura.add_vline(
                x=valor_threshold,
                line_dash="dot",
                line_color="#555555",
                line_width=1,
                annotation_text=nome_threshold,
                annotation_font_size=10,
                annotation_font_color="#888888",
                annotation_position="top",
            )

        return figura

    @staticmethod
    def criar_grafico_incerteza(
        score: float,
        intervalo: tuple[float, float],
        scores_individuais: dict[str, float],
    ) -> go.Figure:
        """
        Cria um grafico de barra de erro mostrando o score principal com
        intervalo de confianca e os scores individuais dos modelos como pontos.

        Args:
            score: Score final calibrado entre 0.0 e 1.0.
            intervalo: Tupla (limite_inferior, limite_superior) em [0, 1].
            scores_individuais: Mapa {id_modelo: score} dos modelos individuais.

        Returns:
            Figura Plotly com score central, barra de erro e pontos individuais.
        """
        score_pct = score * 100
        limite_inf_pct = intervalo[0] * 100
        limite_sup_pct = intervalo[1] * 100
        cor_principal = _cor_para_score(score)

        figura = go.Figure()

        # Pontos dos modelos individuais
        if scores_individuais:
            nomes_modelos = [_nome_amigavel(k) for k in scores_individuais]
            valores_modelos = [v * 100 for v in scores_individuais.values()]
            cores_modelos = [_cor_para_score(v) for v in scores_individuais.values()]

            figura.add_trace(
                go.Scatter(
                    x=valores_modelos,
                    y=nomes_modelos,
                    mode="markers",
                    name="Modelos individuais",
                    marker={
                        "size": 12,
                        "color": cores_modelos,
                        "symbol": "diamond",
                        "line": {"color": "#ffffff", "width": 1},
                    },
                    hovertemplate=(
                        "<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>"
                    ),
                )
            )

        # Score principal com barra de erro
        erro_negativo = score_pct - limite_inf_pct
        erro_positivo = limite_sup_pct - score_pct

        figura.add_trace(
            go.Scatter(
                x=[score_pct],
                y=["Score Final"],
                mode="markers",
                name="Score do ensemble",
                marker={
                    "size": 22,
                    "color": cor_principal,
                    "symbol": "star",
                    "line": {"color": "#ffffff", "width": 2},
                },
                error_x={
                    "type": "data",
                    "array": [erro_positivo],
                    "arrayminus": [erro_negativo],
                    "color": "#aaaaaa",
                    "thickness": 2,
                    "width": 10,
                },
                hovertemplate=(
                    "<b>Score Final</b><br>"
                    f"Score: {score_pct:.1f}%<br>"
                    f"IC 95%: [{limite_inf_pct:.1f}%, {limite_sup_pct:.1f}%]"
                    "<extra></extra>"
                ),
            )
        )

        figura.update_layout(
            title={
                "text": "Score Final com Intervalo de Confianca (95%)",
                "font": {"size": 16},
                "x": 0.5,
                "xanchor": "center",
            },
            template=TEMPLATE_PLOTLY,
            xaxis={
                "title": "Score de IA (%)",
                "range": [0, 105],
                "ticksuffix": "%",
            },
            yaxis={"title": ""},
            height=max(300, (len(scores_individuais) + 1) * 55 + 120),
            margin={"t": 60, "b": 50, "l": 160, "r": 30},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
        )

        return figura

    @staticmethod
    def criar_histograma_distribuicao(
        scores: list[float],
        titulo: str = "Distribuicao de Scores",
    ) -> go.Figure:
        """
        Cria um histograma da distribuicao de scores com regioes coloridas
        por faixa de classificacao e linhas verticais nos thresholds.

        Util para visualizar a distribuicao dos scores ao longo de multiplos
        frames de um video ou de um batch de imagens.

        Args:
            scores: Lista de scores entre 0.0 e 1.0.
            titulo: Titulo do grafico. Padrao: "Distribuicao de Scores".

        Returns:
            Figura Plotly com histograma e anotacoes de threshold.
        """
        if not scores:
            return _figura_vazia("Nenhum score disponivel para histograma")

        scores_pct = [s * 100 for s in scores]

        figura = go.Figure(
            go.Histogram(
                x=scores_pct,
                nbinsx=20,
                marker={
                    "color": scores_pct,
                    "colorscale": [
                        [0.0, COR_VERDE],
                        [THRESHOLD_REAL, COR_VERDE],
                        [THRESHOLD_REAL, COR_VERDE_CLARO],
                        [THRESHOLD_POSS_REAL, COR_VERDE_CLARO],
                        [THRESHOLD_POSS_REAL, COR_LARANJA],
                        [THRESHOLD_POSS_IA, COR_LARANJA],
                        [THRESHOLD_POSS_IA, COR_VERMELHO],
                        [1.0, COR_VERMELHO],
                    ],
                    "colorbar": {"title": "Score (%)"},
                    "line": {"color": "#333333", "width": 0.5},
                },
                opacity=0.85,
                hovertemplate="Score: %{x:.1f}%<br>Contagem: %{y}<extra></extra>",
            )
        )

        # Regioes de fundo com cores suaves
        regioes = [
            (0, THRESHOLD_REAL * 100, COR_VERDE, "Real"),
            (THRESHOLD_REAL * 100, THRESHOLD_POSS_REAL * 100, COR_VERDE_CLARO, "Pos. Real"),
            (THRESHOLD_POSS_REAL * 100, THRESHOLD_POSS_IA * 100, COR_LARANJA, "Pos. IA"),
            (THRESHOLD_POSS_IA * 100, 100, COR_VERMELHO, "IA"),
        ]
        for inicio, fim, cor, _ in regioes:
            figura.add_vrect(
                x0=inicio,
                x1=fim,
                fillcolor=cor,
                opacity=0.06,
                layer="below",
                line_width=0,
            )

        # Linhas de threshold
        for valor_thr, rotulo_thr in [
            (THRESHOLD_REAL * 100, "25%"),
            (THRESHOLD_POSS_REAL * 100, "50%"),
            (THRESHOLD_POSS_IA * 100, "75%"),
        ]:
            figura.add_vline(
                x=valor_thr,
                line_dash="dash",
                line_color="#888888",
                line_width=1.5,
                annotation_text=rotulo_thr,
                annotation_font_size=11,
                annotation_font_color="#aaaaaa",
                annotation_position="top right",
            )

        figura.update_layout(
            title={
                "text": titulo,
                "font": {"size": 16},
                "x": 0.5,
                "xanchor": "center",
            },
            template=TEMPLATE_PLOTLY,
            xaxis={"title": "Score de IA (%)", "ticksuffix": "%"},
            yaxis={"title": "Quantidade de Frames"},
            height=320,
            margin={"t": 60, "b": 50, "l": 60, "r": 30},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
            showlegend=False,
            bargap=0.05,
        )

        return figura

    @staticmethod
    def criar_grafico_concordancia(
        scores_individuais: dict[str, float],
        concordancia: float,
    ) -> go.Figure:
        """
        Cria um grafico radar (polar) mostrando a distribuicao dos scores
        por modelo e o indice de concordancia entre eles.

        O eixo radial representa o score de IA (0-100%). Cada eixo angular
        corresponde a um modelo. Um poligono uniforme indica alta concordancia.

        Args:
            scores_individuais: Mapa {id_modelo: score} dos modelos.
            concordancia: Indice de concordancia entre 0.0 e 1.0.

        Returns:
            Figura Plotly com grafico radar de concordancia.
        """
        if not scores_individuais:
            return _figura_vazia("Nenhum modelo disponivel para grafico de concordancia")

        nomes = [_nome_amigavel(k) for k in scores_individuais]
        valores = [round(v * 100, 1) for v in scores_individuais.values()]

        # Fecha o poligono repetindo o primeiro ponto
        nomes_fechado = nomes + [nomes[0]]
        valores_fechado = valores + [valores[0]]

        cor_concordancia = _cor_para_score(1.0 - concordancia)
        pct_concordancia = round(concordancia * 100, 0)

        figura = go.Figure(
            go.Scatterpolar(
                r=valores_fechado,
                theta=nomes_fechado,
                fill="toself",
                fillcolor=f"rgba(46,204,113,0.15)" if concordancia >= 0.75 else "rgba(231,76,60,0.15)",
                line={
                    "color": COR_VERDE if concordancia >= 0.75 else COR_VERMELHO,
                    "width": 2,
                },
                marker={"size": 8, "color": COR_VERDE if concordancia >= 0.75 else COR_VERMELHO},
                name="Score por modelo",
                hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
            )
        )

        figura.update_layout(
            title={
                "text": (
                    f"Concordancia entre Modelos: "
                    f"<span style='color:{cor_concordancia}'>"
                    f"{pct_concordancia:.0f}%</span>"
                ),
                "font": {"size": 16},
                "x": 0.5,
                "xanchor": "center",
            },
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [0, 100],
                    "ticksuffix": "%",
                    "tickfont": {"size": 10, "color": "#aaaaaa"},
                    "gridcolor": "#444444",
                    "linecolor": "#555555",
                },
                "angularaxis": {
                    "tickfont": {"size": 12, "color": "#cccccc"},
                    "gridcolor": "#333333",
                    "linecolor": "#555555",
                },
                "bgcolor": "rgba(0,0,0,0)",
            },
            template=TEMPLATE_PLOTLY,
            height=380,
            margin={"t": 80, "b": 40, "l": 60, "r": 60},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
            showlegend=False,
        )

        return figura


# ---------------------------------------------------------------------------
# Funcao auxiliar interna
# ---------------------------------------------------------------------------


def _figura_vazia(mensagem: str) -> go.Figure:
    """
    Retorna uma figura Plotly vazia com uma mensagem de aviso centralizada.

    Usada internamente como fallback quando os dados de entrada sao insuficientes
    para gerar o grafico solicitado.

    Args:
        mensagem: Texto a ser exibido no centro do grafico vazio.

    Returns:
        Figura Plotly minimalista com a mensagem de aviso.
    """
    figura = go.Figure()
    figura.add_annotation(
        text=mensagem,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#888888"},
    )
    figura.update_layout(
        template=TEMPLATE_PLOTLY,
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"},
        margin={"t": 30, "b": 30, "l": 30, "r": 30},
    )
    return figura
