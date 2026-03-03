"""
Modulos de visualizacao para o dashboard de deteccao de conteudo gerado por IA.

Exporta as tres classes principais de visualizacao:

- GraficosDeteccao: Graficos Plotly interativos (gauge, barras, radar, histograma).
- VisualizadorMapasCalor: Figuras Matplotlib (GradCAM, espectro FFT, noise print, RGB).
- TimelineVideo: Visualizacoes temporais para analise de video frame a frame.
"""

from visualization.charts import GraficosDeteccao
from visualization.heatmaps import VisualizadorMapasCalor
from visualization.video_timeline import TimelineVideo

__all__ = [
    "GraficosDeteccao",
    "VisualizadorMapasCalor",
    "TimelineVideo",
]
