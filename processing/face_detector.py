"""
Detector facial baseado em MTCNN (Multi-task Cascaded Convolutional Networks).

Utiliza a biblioteca facenet-pytorch para deteccao de rostos em imagens.
Implementa fallback gracioso caso a biblioteca nao esteja instalada.

A deteccao facial e usada no pipeline de video para identificar rostos
e aplicar detectores especializados em deepfakes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Tenta importar facenet_pytorch; fallback se nao estiver instalado
try:
    from facenet_pytorch import MTCNN  # type: ignore[import]

    _FACENET_DISPONIVEL = True
    logger.debug("facenet_pytorch carregado com sucesso.")
except ImportError:
    _FACENET_DISPONIVEL = False
    logger.warning(
        "facenet_pytorch nao instalado. Deteccao facial sera desabilitada. "
        "Instale com: pip install facenet-pytorch"
    )


# Tipo anotado para o resultado de cada rosto detectado
TipoRosto = dict  # {"bbox": tuple[int,int,int,int], "confianca": float, "area": int}


class DetectorFacial:
    """
    Detecta e recorta rostos em imagens usando MTCNN.

    Realiza deteccao multi-escala com nao-maxima-supressao (NMS),
    retornando bounding boxes, confiancas e areas de cada rosto.

    Caso facenet-pytorch nao esteja instalado, os metodos retornam
    listas vazias e registram um aviso no log, sem lancar excecoes.
    """

    # Confianca minima padrao para aceitar uma deteccao como rosto
    CONFIANCA_MINIMA_PADRAO: float = 0.90

    # Margem extra (em pixels) adicionada ao corte de cada rosto
    MARGEM_CORTE_PADRAO: int = 20

    def __init__(
        self,
        dispositivo: str = "cpu",
        confianca_minima: float = CONFIANCA_MINIMA_PADRAO,
        tamanho_minimo_rosto: int = 20,
        margem_corte: int = MARGEM_CORTE_PADRAO,
    ) -> None:
        """
        Inicializa o detector MTCNN.

        Args:
            dispositivo: 'cuda' ou 'cpu'. Padrao: 'cpu'.
            confianca_minima: Threshold de confianca para aceitar deteccao.
                              Padrao: 0.90.
            tamanho_minimo_rosto: Menor tamanho de rosto (em pixels) a detectar.
                                  Padrao: 20.
            margem_corte: Pixels extras ao redor do bounding box no corte.
                          Padrao: 20.
        """
        self._dispositivo = dispositivo
        self._confianca_minima = confianca_minima
        self._tamanho_minimo_rosto = tamanho_minimo_rosto
        self._margem_corte = margem_corte
        self._mtcnn: Optional[object] = None

        if _FACENET_DISPONIVEL:
            self._inicializar_mtcnn()

    def detectar(self, imagem: Image.Image) -> list[TipoRosto]:
        """
        Detecta rostos em uma imagem e retorna metadados de cada deteccao.

        Args:
            imagem: Imagem PIL no modo RGB.

        Returns:
            Lista de dicionarios, um por rosto detectado:
            - 'bbox': tuple (x1, y1, x2, y2) com coordenadas inteiras
            - 'confianca': float entre 0.0 e 1.0
            - 'area': int com area do bounding box em pixels quadrados

            Lista vazia se nenhum rosto for encontrado ou se o MTCNN
            nao estiver disponivel.
        """
        if not _FACENET_DISPONIVEL or self._mtcnn is None:
            logger.debug(
                "MTCNN indisponivel; retornando lista vazia de deteccoes."
            )
            return []

        if imagem.mode != "RGB":
            imagem = imagem.convert("RGB")

        try:
            # MTCNN retorna (boxes, probs) quando keep_all=True
            boxes, probabilidades = self._mtcnn.detect(imagem)  # type: ignore[union-attr]

            if boxes is None or probabilidades is None:
                return []

            rostos_detectados: list[TipoRosto] = []
            for caixa, probabilidade in zip(boxes, probabilidades):
                if probabilidade is None or probabilidade < self._confianca_minima:
                    continue

                x1, y1, x2, y2 = (int(round(float(c))) for c in caixa)
                area = max(0, x2 - x1) * max(0, y2 - y1)

                rostos_detectados.append({
                    "bbox": (x1, y1, x2, y2),
                    "confianca": float(probabilidade),
                    "area": area,
                })

            logger.debug(
                "Detectados %d rosto(s) na imagem.", len(rostos_detectados)
            )
            return rostos_detectados

        except Exception as erro:
            logger.error(
                "Erro durante deteccao facial com MTCNN: %s", erro, exc_info=True
            )
            return []

    def detectar_e_recortar(self, imagem: Image.Image) -> list[Image.Image]:
        """
        Detecta rostos e retorna as regioes recortadas como imagens PIL.

        Aplica uma margem extra ao redor de cada bounding box para garantir
        que o rosto completo (incluindo queixo e testa) seja incluido no corte.

        Args:
            imagem: Imagem PIL no modo RGB.

        Returns:
            Lista de imagens PIL (modo RGB) com os rostos recortados.
            Lista vazia se nenhum rosto for detectado.
        """
        if imagem.mode != "RGB":
            imagem = imagem.convert("RGB")

        largura_img, altura_img = imagem.size
        deteccoes = self.detectar(imagem)

        imagens_rostos: list[Image.Image] = []
        for rosto in deteccoes:
            x1, y1, x2, y2 = rosto["bbox"]

            # Aplica margem com clamp nos limites da imagem
            x1_margem = max(0, x1 - self._margem_corte)
            y1_margem = max(0, y1 - self._margem_corte)
            x2_margem = min(largura_img, x2 + self._margem_corte)
            y2_margem = min(altura_img, y2 + self._margem_corte)

            # Descarta cortes degenerados (area zero)
            if x2_margem <= x1_margem or y2_margem <= y1_margem:
                logger.warning(
                    "Bounding box degenerado ignorado: (%d,%d,%d,%d)",
                    x1_margem, y1_margem, x2_margem, y2_margem,
                )
                continue

            rosto_recortado = imagem.crop(
                (x1_margem, y1_margem, x2_margem, y2_margem)
            )
            imagens_rostos.append(rosto_recortado)

        logger.debug(
            "Retornando %d imagem(ns) de rosto(s) recortada(s).",
            len(imagens_rostos),
        )
        return imagens_rostos

    # ------------------------------------------------------------------
    # Metodos auxiliares privados
    # ------------------------------------------------------------------

    def _inicializar_mtcnn(self) -> None:
        """Instancia o MTCNN com as configuracoes do detector."""
        try:
            self._mtcnn = MTCNN(
                keep_all=True,
                device=self._dispositivo,
                min_face_size=self._tamanho_minimo_rosto,
                # Tres estagios de confianca: P-Net, R-Net, O-Net
                thresholds=[0.6, 0.7, self._confianca_minima],
                post_process=False,
            )
            logger.info(
                "MTCNN inicializado (dispositivo=%s, confianca_min=%.2f).",
                self._dispositivo,
                self._confianca_minima,
            )
        except Exception as erro:
            logger.error(
                "Falha ao inicializar MTCNN: %s. "
                "Deteccao facial sera desabilitada.",
                erro,
                exc_info=True,
            )
            self._mtcnn = None

    @staticmethod
    def converter_para_numpy(imagens_rostos: list[Image.Image]) -> list[np.ndarray]:
        """
        Converte lista de imagens PIL de rostos para arrays numpy uint8.

        Metodo auxiliar para facilitar integracao com processadores que
        trabalham com numpy (AnalisadorPixels, GeradorGradCAM, etc.).

        Args:
            imagens_rostos: Lista de imagens PIL em RGB.

        Returns:
            Lista de arrays numpy (H, W, 3) uint8.
        """
        return [np.array(rosto, dtype=np.uint8) for rosto in imagens_rostos]
