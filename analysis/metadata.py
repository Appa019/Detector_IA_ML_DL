"""
Analise de metadados EXIF para deteccao de conteudo gerado por IA.

Imagens fotograficas reais geralmente contem metadados ricos provenientes
do sensor da camera (modelo, configuracoes de exposicao, GPS, etc.).
Imagens geradas por IA tipicamente nao possuem EXIF ou contem assinaturas
de software especificas de ferramentas de geracao.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# Assinaturas de software conhecidas de ferramentas de geracao por IA.
# Comparacao case-insensitive na analise.
ASSINATURAS_SOFTWARE_IA: tuple[str, ...] = (
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dall·e",
    "adobe firefly",
    "firefly",
    "runway",
    "imagen",
    "bing image creator",
    "nightcafe",
    "dreamstudio",
    "novelai",
    "invoke ai",
    "invokeai",
    "automatic1111",
    "comfyui",
    "leonardo.ai",
    "leonardo ai",
    "canva ai",
    "adobe generative fill",
    "generative fill",
)

# Campos EXIF que indicam origem fotografica genuina
CAMPOS_CAMERA_EXIF: tuple[str, ...] = (
    "Make",
    "Model",
    "LensModel",
    "LensMake",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "FocalLength",
    "ShutterSpeedValue",
    "ApertureValue",
)

# Nivel de suspeita (0=nenhuma, 1=baixa, 2=media, 3=alta, 4=muito alta)
NIVEIS_SUSPEITA: dict[int, str] = {
    0: "nenhuma",
    1: "baixa",
    2: "media",
    3: "alta",
    4: "muito_alta",
}


class AnalisadorMetadados:
    """
    Extrai e analisa metadados EXIF de arquivos de imagem.

    Detecta indicadores de geracao por IA como:
    - Ausencia total de metadados EXIF (comum em imagens de IA)
    - Presenca de assinaturas de software de geracao de IA
    - Inconsistencias nos campos de camera (ex.: campo Software presente,
      mas Make/Model ausentes)
    """

    def extrair_metadados(self, caminho_arquivo: str | Path) -> dict[str, Any]:
        """
        Extrai todos os metadados EXIF de um arquivo de imagem.

        Os valores de tag numericos sao convertidos para nomes legíveis
        usando PIL.ExifTags.TAGS. Dados binarios (thumbnails, etc.) sao
        omitidos para manter o resultado serializavel.

        Args:
            caminho_arquivo: Caminho para o arquivo de imagem (str ou Path).

        Returns:
            Dicionario com metadados EXIF em formato legivel.
            Retorna dicionario vazio se o arquivo nao possuir EXIF ou
            se ocorrer erro na leitura.
        """
        caminho = Path(caminho_arquivo)

        if not caminho.exists():
            logger.warning(
                "Arquivo nao encontrado para extracao de metadados: %s", caminho
            )
            return {}

        try:
            with Image.open(caminho) as imagem:
                dados_exif_brutos = imagem._getexif()  # type: ignore[attr-defined]

            if not dados_exif_brutos:
                logger.debug(
                    "Nenhum dado EXIF encontrado em: %s", caminho.name
                )
                return {}

            metadados: dict[str, Any] = {}
            for tag_id, valor in dados_exif_brutos.items():
                nome_tag = TAGS.get(tag_id, f"Tag_{tag_id}")

                # Descarta dados binarios (thumbnails, previews, etc.)
                if isinstance(valor, bytes):
                    metadados[nome_tag] = f"<binario {len(valor)} bytes>"
                    continue

                # Converte IFDRational para float para serializacao
                try:
                    if hasattr(valor, "numerator") and hasattr(valor, "denominator"):
                        valor = float(valor)
                    elif isinstance(valor, tuple):
                        valor = tuple(
                            float(v) if hasattr(v, "numerator") else v
                            for v in valor
                        )
                except (TypeError, ZeroDivisionError):
                    valor = str(valor)

                metadados[nome_tag] = valor

            logger.debug(
                "Extraidos %d campos EXIF de: %s",
                len(metadados),
                caminho.name,
            )
            return metadados

        except AttributeError:
            # Formato sem suporte a EXIF (ex.: PNG sem chunk tEXt)
            logger.debug("Formato sem EXIF: %s", caminho.name)
            return {}
        except Exception as erro:
            logger.error(
                "Erro ao extrair metadados de %s: %s",
                caminho.name,
                erro,
                exc_info=True,
            )
            return {}

    def analisar_indicadores_ia(self, metadados: dict[str, Any]) -> dict[str, Any]:
        """
        Analisa metadados em busca de indicadores de geracao por IA.

        Verifica:
        1. Ausencia total de EXIF (forte indicador de IA)
        2. Software de geracao por IA no campo 'Software' ou 'Artist'
        3. Ausencia de informacoes de camera (Make/Model) com EXIF presente
        4. GPS sem informacao de camera (inconsistencia suspeita)

        Args:
            metadados: Dicionario retornado por extrair_metadados().

        Returns:
            Dicionario com:
            - 'nivel_suspeita': str descritivo ("nenhuma" a "muito_alta")
            - 'pontuacao_suspeita': int de 0 a 4
            - 'sem_exif': bool
            - 'software_ia_detectado': bool
            - 'software_identificado': str | None  (nome do software de IA)
            - 'campos_camera_presentes': bool
            - 'campos_camera_ausentes': list[str]
            - 'indicadores': list[str] com descricoes dos sinais encontrados
        """
        indicadores: list[str] = []
        pontuacao_suspeita = 0

        # --- Verificacao 1: ausencia total de EXIF ---
        sem_exif = len(metadados) == 0
        if sem_exif:
            indicadores.append(
                "Arquivo sem metadados EXIF — comum em imagens geradas por IA."
            )
            # Ausencia de EXIF e forte indicador, mas nao conclusiva
            pontuacao_suspeita += 2

        # --- Verificacao 2: assinatura de software de IA ---
        software_ia_detectado = False
        software_identificado: str | None = None

        campos_texto_possiveis = [
            str(metadados.get("Software", "")),
            str(metadados.get("Artist", "")),
            str(metadados.get("ImageDescription", "")),
            str(metadados.get("UserComment", "")),
            str(metadados.get("Copyright", "")),
        ]
        texto_combinado = " ".join(campos_texto_possiveis).lower()

        for assinatura in ASSINATURAS_SOFTWARE_IA:
            if assinatura.lower() in texto_combinado:
                software_ia_detectado = True
                software_identificado = assinatura
                indicadores.append(
                    f"Software de geracao por IA identificado: '{assinatura}'."
                )
                pontuacao_suspeita += 2
                break  # Uma assinatura ja e suficiente para pontuacao maxima

        # --- Verificacao 3: ausencia de informacoes de camera ---
        campos_camera_ausentes = [
            campo
            for campo in CAMPOS_CAMERA_EXIF
            if campo not in metadados
        ]
        campos_camera_presentes = len(campos_camera_ausentes) < len(CAMPOS_CAMERA_EXIF)

        if not sem_exif and len(campos_camera_ausentes) == len(CAMPOS_CAMERA_EXIF):
            # EXIF presente mas sem nenhum campo de camera -> suspeito
            indicadores.append(
                "Metadados EXIF presentes, mas sem nenhum campo de camera "
                "(Make, Model, ExposureTime, etc.)."
            )
            pontuacao_suspeita += 1
        elif not sem_exif and len(campos_camera_ausentes) > len(CAMPOS_CAMERA_EXIF) // 2:
            indicadores.append(
                f"Maioria dos campos de camera ausentes "
                f"({len(campos_camera_ausentes)}/{len(CAMPOS_CAMERA_EXIF)} campos)."
            )

        # --- Verificacao 4: inconsistencia GPS sem camera ---
        tem_gps = "GPSInfo" in metadados
        sem_make_model = (
            "Make" not in metadados and "Model" not in metadados
        )
        if tem_gps and sem_make_model and not sem_exif:
            indicadores.append(
                "GPS presente sem informacao de camera — inconsistencia incomum."
            )
            pontuacao_suspeita += 1

        # Clamp da pontuacao entre 0 e 4
        pontuacao_suspeita = min(pontuacao_suspeita, 4)
        nivel_suspeita = NIVEIS_SUSPEITA[pontuacao_suspeita]

        return {
            "nivel_suspeita": nivel_suspeita,
            "pontuacao_suspeita": pontuacao_suspeita,
            "sem_exif": sem_exif,
            "software_ia_detectado": software_ia_detectado,
            "software_identificado": software_identificado,
            "campos_camera_presentes": campos_camera_presentes,
            "campos_camera_ausentes": campos_camera_ausentes,
            "indicadores": indicadores,
        }
