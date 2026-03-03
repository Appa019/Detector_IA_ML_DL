"""
Extracao de frames e metadados de arquivos de video.

Utiliza OpenCV (cv2) para abrir e amostrar frames do video em
intervalos configuraveis, retornando os frames em formato RGB
compativel com os processadores de imagem do ensemble.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ProcessadorVideo:
    """
    Extrai frames e metadados de arquivos de video usando OpenCV.

    Permite amostrar frames em intervalos regulares para analise
    de conteudo gerado por IA sem processar todos os frames
    (o que seria computacionalmente inviavel para videos longos).
    """

    def extrair_frames(
        self,
        caminho_video: str | Path,
        intervalo: int = 10,
        max_frames: int = 100,
    ) -> list[tuple[np.ndarray, int]]:
        """
        Extrai frames do video em intervalos regulares.

        Args:
            caminho_video: Caminho para o arquivo de video.
            intervalo: Extrair um frame a cada N frames. Padrao: 10.
            max_frames: Numero maximo de frames a extrair. Padrao: 100.

        Returns:
            Lista de tuplas (frame_rgb, indice_frame), onde frame_rgb e
            um array numpy (H, W, 3) uint8 em RGB e indice_frame e a
            posicao do frame no video (0-indexed).

        Raises:
            FileNotFoundError: Se o arquivo de video nao existir.
            IOError: Se o video nao puder ser aberto pelo OpenCV.
        """
        caminho = Path(caminho_video)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo de video nao encontrado: {caminho}"
            )

        captura = cv2.VideoCapture(str(caminho))
        if not captura.isOpened():
            raise IOError(
                f"OpenCV nao conseguiu abrir o video: {caminho.name}"
            )

        frames_extraidos: list[tuple[np.ndarray, int]] = []

        try:
            frames_extraidos = list(
                self._iterar_frames(captura, intervalo, max_frames)
            )
        finally:
            captura.release()

        logger.info(
            "Extraidos %d frames de '%s' (intervalo=%d, max=%d)",
            len(frames_extraidos),
            caminho.name,
            intervalo,
            max_frames,
        )
        return frames_extraidos

    def obter_info_video(self, caminho_video: str | Path) -> dict:
        """
        Retorna informacoes tecnicas do arquivo de video.

        Args:
            caminho_video: Caminho para o arquivo de video.

        Returns:
            Dicionario com:
            - 'fps': float — quadros por segundo
            - 'total_frames': int — numero total de frames
            - 'duracao_segundos': float — duracao em segundos
            - 'largura': int — largura em pixels
            - 'altura': int — altura em pixels
            - 'resolucao': str — "largura x altura" para exibicao
            - 'codec': str — fourcc do codec (ex.: "mp4v", "avc1")

        Raises:
            FileNotFoundError: Se o arquivo nao existir.
            IOError: Se o video nao puder ser aberto.
        """
        caminho = Path(caminho_video)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo de video nao encontrado: {caminho}"
            )

        captura = cv2.VideoCapture(str(caminho))
        if not captura.isOpened():
            raise IOError(
                f"OpenCV nao conseguiu abrir o video: {caminho.name}"
            )

        try:
            fps = float(captura.get(cv2.CAP_PROP_FPS))
            total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
            largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Codec como string de 4 caracteres (fourcc)
            fourcc_int = int(captura.get(cv2.CAP_PROP_FOURCC))
            codec = self._fourcc_para_str(fourcc_int)

            duracao_segundos = (
                total_frames / fps if fps > 0 else 0.0
            )

        finally:
            captura.release()

        info = {
            "fps": fps,
            "total_frames": total_frames,
            "duracao_segundos": round(duracao_segundos, 2),
            "largura": largura,
            "altura": altura,
            "resolucao": f"{largura} x {altura}",
            "codec": codec,
        }

        logger.debug("Informacoes do video '%s': %s", caminho.name, info)
        return info

    # ------------------------------------------------------------------
    # Metodos auxiliares privados
    # ------------------------------------------------------------------

    @staticmethod
    def _iterar_frames(
        captura: cv2.VideoCapture,
        intervalo: int,
        max_frames: int,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """
        Gerador que le frames do objeto cv2.VideoCapture em intervalos.

        Usa leitura posicional (CAP_PROP_POS_FRAMES) para pular
        diretamente para o frame alvo sem decodificar todos os frames
        intermediarios, o que e significativamente mais rapido para
        videos longos.

        Args:
            captura: Objeto VideoCapture ja aberto.
            intervalo: Extrair um frame a cada N frames.
            max_frames: Limite maximo de frames a gerar.

        Yields:
            Tupla (frame_rgb, indice_frame).
        """
        total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        contador_extraidos = 0
        indice_frame = 0

        while indice_frame < total_frames and contador_extraidos < max_frames:
            # Posiciona captura no frame desejado
            captura.set(cv2.CAP_PROP_POS_FRAMES, float(indice_frame))
            leitura_ok, frame_bgr = captura.read()

            if not leitura_ok or frame_bgr is None:
                logger.debug(
                    "Falha ao ler frame %d; encerrando extracao.", indice_frame
                )
                break

            # OpenCV retorna BGR; converte para RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_rgb, indice_frame

            contador_extraidos += 1
            indice_frame += intervalo

    @staticmethod
    def _fourcc_para_str(fourcc_int: int) -> str:
        """Converte o inteiro fourcc retornado pelo OpenCV para string legivel."""
        try:
            caracteres = [
                chr((fourcc_int >> (8 * i)) & 0xFF)
                for i in range(4)
            ]
            codec_str = "".join(c for c in caracteres if c.isprintable())
            return codec_str if codec_str else "desconhecido"
        except Exception:
            return "desconhecido"
