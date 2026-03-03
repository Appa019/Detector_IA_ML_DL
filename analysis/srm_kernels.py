"""
Kernels SRM (Spatial Rich Model) para deteccao de manipulacao em imagens.

Implementa os 30 kernels de predicao de residuo baseados em
Fridrich & Kodovsky (2012) - "Rich Models for Steganalysis of Digital Images".

Categorias:
- 6 first-order edge (diferenca direcional simples)
- 6 second-order (diferenca de segunda ordem)
- 6 third-order (diferenca de terceira ordem)
- 6 SQUARE (kernels quadrados 3x3 centrados)
- 6 EDGE (kernels de borda diagonais e mistos)

Todos os kernels sao 5x5 e normalizados para soma zero (filtros passa-alta).
"""

from __future__ import annotations

import cv2
import numpy as np


def obter_kernels_srm() -> np.ndarray:
    """
    Retorna os 30 kernels SRM 5x5 do estado da arte.

    Returns:
        Array numpy com shape (30, 5, 5) contendo os 30 filtros de predicao
        de residuo normalizados (soma zero, float32).
    """
    kernels = np.zeros((30, 5, 5), dtype=np.float32)

    # === First-order edge (6 kernels) ===
    # Diferenca direcional simples em 6 direcoes

    # Horizontal
    kernels[0, 2, 1] = -1.0
    kernels[0, 2, 2] = 1.0

    # Vertical
    kernels[1, 1, 2] = -1.0
    kernels[1, 2, 2] = 1.0

    # Diagonal principal
    kernels[2, 1, 1] = -1.0
    kernels[2, 2, 2] = 1.0

    # Diagonal secundaria
    kernels[3, 1, 3] = -1.0
    kernels[3, 2, 2] = 1.0

    # Horizontal reverso
    kernels[4, 2, 3] = -1.0
    kernels[4, 2, 2] = 1.0

    # Vertical reverso
    kernels[5, 3, 2] = -1.0
    kernels[5, 2, 2] = 1.0

    # === Second-order (6 kernels) ===
    # Diferenca de segunda ordem — captura curvatura

    # Horizontal 2a ordem
    kernels[6, 2, 1] = 1.0
    kernels[6, 2, 2] = -2.0
    kernels[6, 2, 3] = 1.0

    # Vertical 2a ordem
    kernels[7, 1, 2] = 1.0
    kernels[7, 2, 2] = -2.0
    kernels[7, 3, 2] = 1.0

    # Diagonal principal 2a ordem
    kernels[8, 1, 1] = 1.0
    kernels[8, 2, 2] = -2.0
    kernels[8, 3, 3] = 1.0

    # Diagonal secundaria 2a ordem
    kernels[9, 1, 3] = 1.0
    kernels[9, 2, 2] = -2.0
    kernels[9, 3, 1] = 1.0

    # Cruz horizontal-vertical 2a ordem
    kernels[10, 1, 2] = 1.0
    kernels[10, 2, 2] = -2.0
    kernels[10, 2, 3] = 1.0

    # Cruz vertical-horizontal 2a ordem
    kernels[11, 2, 1] = 1.0
    kernels[11, 2, 2] = -2.0
    kernels[11, 3, 2] = 1.0

    # === Third-order (6 kernels) ===
    # Diferenca de terceira ordem — captura variacao de curvatura

    # Horizontal 3a ordem
    kernels[12, 2, 0] = -1.0
    kernels[12, 2, 1] = 3.0
    kernels[12, 2, 2] = -3.0
    kernels[12, 2, 3] = 1.0

    # Vertical 3a ordem
    kernels[13, 0, 2] = -1.0
    kernels[13, 1, 2] = 3.0
    kernels[13, 2, 2] = -3.0
    kernels[13, 3, 2] = 1.0

    # Diagonal principal 3a ordem
    kernels[14, 0, 0] = -1.0
    kernels[14, 1, 1] = 3.0
    kernels[14, 2, 2] = -3.0
    kernels[14, 3, 3] = 1.0

    # Diagonal secundaria 3a ordem
    kernels[15, 0, 4] = -1.0
    kernels[15, 1, 3] = 3.0
    kernels[15, 2, 2] = -3.0
    kernels[15, 3, 1] = 1.0

    # Horizontal reverso 3a ordem
    kernels[16, 2, 4] = -1.0
    kernels[16, 2, 3] = 3.0
    kernels[16, 2, 2] = -3.0
    kernels[16, 2, 1] = 1.0

    # Vertical reverso 3a ordem
    kernels[17, 4, 2] = -1.0
    kernels[17, 3, 2] = 3.0
    kernels[17, 2, 2] = -3.0
    kernels[17, 1, 2] = 1.0

    # === SQUARE (6 kernels) ===
    # Kernels quadrados 3x3 centrados no pixel 5x5

    # SQUARE 3x3 media (predicao pela media dos 8 vizinhos)
    kernels[18, 1, 1] = -1.0 / 8.0
    kernels[18, 1, 2] = -1.0 / 8.0
    kernels[18, 1, 3] = -1.0 / 8.0
    kernels[18, 2, 1] = -1.0 / 8.0
    kernels[18, 2, 2] = 1.0
    kernels[18, 2, 3] = -1.0 / 8.0
    kernels[18, 3, 1] = -1.0 / 8.0
    kernels[18, 3, 2] = -1.0 / 8.0
    kernels[18, 3, 3] = -1.0 / 8.0

    # SQUARE 5x5 media (predicao pela media dos 24 vizinhos)
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                kernels[19, i, j] = 1.0
            else:
                kernels[19, i, j] = -1.0 / 24.0

    # SQUARE cruz (predicao pelos 4 vizinhos diretos)
    kernels[20, 1, 2] = -1.0 / 4.0
    kernels[20, 2, 1] = -1.0 / 4.0
    kernels[20, 2, 2] = 1.0
    kernels[20, 2, 3] = -1.0 / 4.0
    kernels[20, 3, 2] = -1.0 / 4.0

    # SQUARE diagonal (predicao pelas 4 diagonais)
    kernels[21, 1, 1] = -1.0 / 4.0
    kernels[21, 1, 3] = -1.0 / 4.0
    kernels[21, 2, 2] = 1.0
    kernels[21, 3, 1] = -1.0 / 4.0
    kernels[21, 3, 3] = -1.0 / 4.0

    # SQUARE 5x5 cruz expandida (8 vizinhos a distancia 2)
    kernels[22, 0, 2] = -1.0 / 8.0
    kernels[22, 1, 2] = -1.0 / 8.0
    kernels[22, 2, 0] = -1.0 / 8.0
    kernels[22, 2, 1] = -1.0 / 8.0
    kernels[22, 2, 2] = 1.0
    kernels[22, 2, 3] = -1.0 / 8.0
    kernels[22, 2, 4] = -1.0 / 8.0
    kernels[22, 3, 2] = -1.0 / 8.0
    kernels[22, 4, 2] = -1.0 / 8.0

    # SQUARE 5x5 diagonal expandida (8 vizinhos diagonais a distancias 1 e 2)
    kernels[23, 0, 0] = -1.0 / 8.0
    kernels[23, 0, 4] = -1.0 / 8.0
    kernels[23, 1, 1] = -1.0 / 8.0
    kernels[23, 1, 3] = -1.0 / 8.0
    kernels[23, 2, 2] = 1.0
    kernels[23, 3, 1] = -1.0 / 8.0
    kernels[23, 3, 3] = -1.0 / 8.0
    kernels[23, 4, 0] = -1.0 / 8.0
    kernels[23, 4, 4] = -1.0 / 8.0

    # === EDGE (6 kernels) ===
    # Kernels de borda com orientacoes mistas

    # EDGE Laplaciano 3x3
    kernels[24, 1, 1] = 0.0
    kernels[24, 1, 2] = -1.0
    kernels[24, 1, 3] = 0.0
    kernels[24, 2, 1] = -1.0
    kernels[24, 2, 2] = 4.0
    kernels[24, 2, 3] = -1.0
    kernels[24, 3, 1] = 0.0
    kernels[24, 3, 2] = -1.0
    kernels[24, 3, 3] = 0.0

    # EDGE Laplaciano 3x3 com diagonais
    kernels[25, 1, 1] = -1.0
    kernels[25, 1, 2] = -1.0
    kernels[25, 1, 3] = -1.0
    kernels[25, 2, 1] = -1.0
    kernels[25, 2, 2] = 8.0
    kernels[25, 2, 3] = -1.0
    kernels[25, 3, 1] = -1.0
    kernels[25, 3, 2] = -1.0
    kernels[25, 3, 3] = -1.0

    # EDGE Sobel horizontal
    kernels[26, 1, 1] = -1.0
    kernels[26, 1, 2] = 0.0
    kernels[26, 1, 3] = 1.0
    kernels[26, 2, 1] = -2.0
    kernels[26, 2, 2] = 0.0
    kernels[26, 2, 3] = 2.0
    kernels[26, 3, 1] = -1.0
    kernels[26, 3, 2] = 0.0
    kernels[26, 3, 3] = 1.0

    # EDGE Sobel vertical
    kernels[27, 1, 1] = -1.0
    kernels[27, 1, 2] = -2.0
    kernels[27, 1, 3] = -1.0
    kernels[27, 2, 1] = 0.0
    kernels[27, 2, 2] = 0.0
    kernels[27, 2, 3] = 0.0
    kernels[27, 3, 1] = 1.0
    kernels[27, 3, 2] = 2.0
    kernels[27, 3, 3] = 1.0

    # EDGE Prewitt horizontal
    kernels[28, 1, 1] = -1.0
    kernels[28, 1, 2] = 0.0
    kernels[28, 1, 3] = 1.0
    kernels[28, 2, 1] = -1.0
    kernels[28, 2, 2] = 0.0
    kernels[28, 2, 3] = 1.0
    kernels[28, 3, 1] = -1.0
    kernels[28, 3, 2] = 0.0
    kernels[28, 3, 3] = 1.0

    # EDGE Prewitt vertical
    kernels[29, 1, 1] = -1.0
    kernels[29, 1, 2] = -1.0
    kernels[29, 1, 3] = -1.0
    kernels[29, 2, 1] = 0.0
    kernels[29, 2, 2] = 0.0
    kernels[29, 2, 3] = 0.0
    kernels[29, 3, 1] = 1.0
    kernels[29, 3, 2] = 1.0
    kernels[29, 3, 3] = 1.0

    return kernels


def aplicar_filtros_srm(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Aplica os 30 filtros SRM a uma imagem em escala de cinza.

    Cada kernel produz um mapa de residuos que captura artefatos
    de manipulacao invisiveis ao olho humano. CPU-only, sem VRAM.

    Args:
        imagem_cinza: Array numpy 2D (H, W) em uint8 ou float32.

    Returns:
        Array numpy com shape (30, H, W) contendo os 30 mapas de residuos.
    """
    if imagem_cinza.ndim != 2:
        raise ValueError(
            f"Imagem deve ser 2D (escala de cinza). Shape recebido: {imagem_cinza.shape}"
        )

    img_float = imagem_cinza.astype(np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0

    kernels = obter_kernels_srm()
    altura, largura = img_float.shape
    residuos = np.zeros((30, altura, largura), dtype=np.float32)

    for i in range(30):
        residuos[i] = cv2.filter2D(
            img_float,
            ddepth=cv2.CV_32F,
            kernel=kernels[i],
        )

    return residuos
