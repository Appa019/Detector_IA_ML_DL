"""
Preprocessamento de imagens para os modelos de deteccao.

Encapsula o pipeline completo de carregamento, redimensionamento,
normalizacao e conversao para tensor PyTorch, compativel com os
modelos ViT e EfficientNet do ensemble.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)

# Estatisticas ImageNet — padrao para modelos pre-treinados
_MEDIA_IMAGENET: tuple[float, float, float] = (0.485, 0.456, 0.406)
_DESVIO_IMAGENET: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Tipos aceitos como fonte de imagem
FonteImagem = Union[str, Path, bytes, "io.IOBase"]  # UploadedFile tambem suportado


class ProcessadorImagem:
    """
    Pipeline de preprocessamento de imagens para inferencia nos detectores.

    Suporta carregamento a partir de caminho de arquivo (str/Path),
    bytes brutos ou objetos file-like (e.g., UploadFile do FastAPI).

    O metodo principal e preparar_para_modelo(), que executa o pipeline
    completo e retorna um tensor (1, 3, H, W) pronto para inferencia.
    """

    def carregar_imagem(self, fonte: FonteImagem) -> Image.Image:
        """
        Carrega uma imagem de diversas fontes e converte para RGB.

        Args:
            fonte: Pode ser:
                   - str ou Path: caminho para o arquivo de imagem
                   - bytes: dados binarios da imagem
                   - objeto file-like com metodo read() (UploadFile, BytesIO)

        Returns:
            Imagem PIL no modo RGB.

        Raises:
            ValueError: Se a fonte nao for de um tipo suportado.
            FileNotFoundError: Se o caminho fornecido nao existir.
            IOError: Se os dados nao puderem ser interpretados como imagem.
        """
        try:
            if isinstance(fonte, (str, Path)):
                caminho = Path(fonte)
                if not caminho.exists():
                    raise FileNotFoundError(
                        f"Arquivo de imagem nao encontrado: {caminho}"
                    )
                imagem = Image.open(caminho)

            elif isinstance(fonte, bytes):
                imagem = Image.open(io.BytesIO(fonte))

            elif hasattr(fonte, "read"):
                # Suporta BytesIO, SpooledTemporaryFile (FastAPI)
                dados = fonte.read()
                if isinstance(dados, str):
                    dados = dados.encode("utf-8")
                imagem = Image.open(io.BytesIO(dados))

            else:
                raise ValueError(
                    f"Tipo de fonte nao suportado: {type(fonte).__name__}. "
                    "Use str, Path, bytes ou file-like object."
                )

            # Garante modo RGB (descarta alpha, converte grayscale, etc.)
            if imagem.mode != "RGB":
                imagem = imagem.convert("RGB")

            logger.debug(
                "Imagem carregada: modo=%s, tamanho=%s", imagem.mode, imagem.size
            )
            return imagem

        except (FileNotFoundError, ValueError):
            raise
        except Exception as erro:
            raise IOError(
                f"Nao foi possivel carregar a imagem: {erro}"
            ) from erro

    def redimensionar(
        self,
        imagem: Image.Image,
        tamanho: int | tuple[int, int],
    ) -> Image.Image:
        """
        Redimensiona a imagem mantendo proporcao e aplica corte central.

        A menor dimensao e escalada para o tamanho alvo e depois aplica
        um center crop para obter uma imagem quadrada (ou do tamanho exato
        se tamanho for uma tupla).

        Args:
            imagem: Imagem PIL de entrada.
            tamanho: Tamanho alvo. Se int, produz imagem quadrada (H=W=tamanho).
                     Se tupla (H, W), produz imagem com essas dimensoes exatas.

        Returns:
            Imagem PIL redimensionada e recortada.
        """
        if isinstance(tamanho, int):
            tamanho_alvo_h = tamanho
            tamanho_alvo_w = tamanho
        else:
            tamanho_alvo_h, tamanho_alvo_w = tamanho

        largura_orig, altura_orig = imagem.size

        # Escala pela menor dimensao preservando proporcao
        escala = max(
            tamanho_alvo_w / largura_orig,
            tamanho_alvo_h / altura_orig,
        )
        nova_largura = int(round(largura_orig * escala))
        nova_altura = int(round(altura_orig * escala))

        imagem_redim = imagem.resize(
            (nova_largura, nova_altura), Image.BICUBIC
        )

        # Center crop
        margem_esq = (nova_largura - tamanho_alvo_w) // 2
        margem_topo = (nova_altura - tamanho_alvo_h) // 2
        imagem_cortada = imagem_redim.crop((
            margem_esq,
            margem_topo,
            margem_esq + tamanho_alvo_w,
            margem_topo + tamanho_alvo_h,
        ))

        return imagem_cortada

    def normalizar_tensor(
        self,
        imagem: Image.Image,
        media: tuple[float, float, float] = _MEDIA_IMAGENET,
        desvio: tuple[float, float, float] = _DESVIO_IMAGENET,
    ) -> torch.Tensor:
        """
        Converte imagem PIL para tensor normalizado.

        Aplica as transformacoes:
        1. PIL -> tensor float32 com valores em [0, 1]  (C, H, W)
        2. Normalizacao canal a canal: (pixel - media) / desvio

        Args:
            imagem: Imagem PIL em modo RGB.
            media: Media de normalizacao por canal (R, G, B).
            desvio: Desvio padrao de normalizacao por canal (R, G, B).

        Returns:
            Tensor float32 de shape (3, H, W) normalizado.
        """
        # to_tensor: divide por 255 e muda para (C, H, W)
        tensor = TF.to_tensor(imagem)
        tensor_normalizado = TF.normalize(tensor, list(media), list(desvio))
        return tensor_normalizado

    def preparar_para_modelo(
        self,
        imagem: Image.Image | FonteImagem,
        tamanho: int = 224,
        media: tuple[float, float, float] = _MEDIA_IMAGENET,
        desvio: tuple[float, float, float] = _DESVIO_IMAGENET,
    ) -> torch.Tensor:
        """
        Executa o pipeline completo de preprocessamento.

        Sequencia de operacoes:
        1. Carrega imagem se a entrada nao for PIL (opcional)
        2. Redimensiona com proporcao preservada + center crop
        3. Converte para tensor e normaliza com estatisticas ImageNet
        4. Adiciona dimensao de batch -> (1, 3, H, W)

        Args:
            imagem: Imagem PIL ja carregada OU qualquer fonte aceita
                    por carregar_imagem() (path, bytes, file-like).
            tamanho: Lado da imagem quadrada de saida. Padrao: 224.
            media: Tupla (R, G, B) de media para normalizacao.
            desvio: Tupla (R, G, B) de desvio para normalizacao.

        Returns:
            Tensor float32 de shape (1, 3, tamanho, tamanho),
            pronto para inferencia nos modelos do ensemble.
        """
        # Carrega se necessario
        if not isinstance(imagem, Image.Image):
            imagem = self.carregar_imagem(imagem)

        # Garante RGB
        if imagem.mode != "RGB":
            imagem = imagem.convert("RGB")

        # Redimensiona + center crop
        imagem_redim = self.redimensionar(imagem, tamanho)

        # Tensor (3, H, W) normalizado
        tensor = self.normalizar_tensor(imagem_redim, media, desvio)

        # Adiciona dimensao de batch: (1, 3, H, W)
        tensor_batch = tensor.unsqueeze(0)

        logger.debug(
            "Imagem preparada para modelo: shape=%s, dtype=%s",
            tuple(tensor_batch.shape),
            tensor_batch.dtype,
        )
        return tensor_batch

    @staticmethod
    def tensor_para_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Converte tensor (1, 3, H, W) ou (3, H, W) de volta para imagem numpy uint8.

        Inverte a normalizacao ImageNet e converte para [0, 255] uint8.
        Util para visualizacao de pre-processamento e GradCAM.

        Args:
            tensor: Tensor PyTorch normalizado.

        Returns:
            Array numpy (H, W, 3) uint8 em RGB.
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Move para CPU e converte para float32
        t = tensor.cpu().float()

        # Desfaz normalizacao ImageNet
        media = torch.tensor(_MEDIA_IMAGENET).view(3, 1, 1)
        desvio = torch.tensor(_DESVIO_IMAGENET).view(3, 1, 1)
        t = t * desvio + media

        # Clamp em [0, 1] e converte para uint8
        t = torch.clamp(t, 0.0, 1.0)
        imagem_numpy = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return imagem_numpy
