"""
Download automatico dos pesos dos modelos do HuggingFace Hub.
"""

import logging
import sys
from pathlib import Path

from config.settings import DIRETORIO_MODELOS
from config.model_registry import REGISTRO_MODELOS

logger = logging.getLogger(__name__)


def baixar_modelo_huggingface(hub_id: str, diretorio_destino: Path) -> Path:
    """Baixa um modelo do HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    caminho = snapshot_download(
        repo_id=hub_id,
        local_dir=diretorio_destino / hub_id.replace("/", "_"),
        local_dir_use_symlinks=False,
    )
    logger.info(f"Modelo {hub_id} baixado em: {caminho}")
    return Path(caminho)


def baixar_efficientnet() -> Path:
    """Baixa pesos do EfficientNet-B4 pre-treinado."""
    import torch
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

    diretorio = DIRETORIO_MODELOS / "efficientnet_b4"
    diretorio.mkdir(parents=True, exist_ok=True)

    caminho_pesos = diretorio / "efficientnet_b4_imagenet.pth"

    if caminho_pesos.exists():
        logger.info("EfficientNet-B4 ja existe. Pulando download.")
        return caminho_pesos

    logger.info("Baixando EfficientNet-B4 pre-treinado (ImageNet)...")
    modelo = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    torch.save(modelo.state_dict(), caminho_pesos)
    logger.info(f"EfficientNet-B4 salvo em: {caminho_pesos}")
    return caminho_pesos


def baixar_todos_os_modelos():
    """Baixa todos os modelos registrados."""
    DIRETORIO_MODELOS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Download de Modelos para Deteccao de IA")
    print("=" * 60)

    for id_modelo, registro in REGISTRO_MODELOS.items():
        print(f"\n>> [{id_modelo}] {registro.nome_exibicao}")

        if registro.hub_id == "local":
            print("   Modelo local (sem download necessario)")
            continue

        if registro.hub_id == "efficientnet-b4":
            try:
                caminho = baixar_efficientnet()
                print(f"   OK: {caminho}")
            except Exception as e:
                print(f"   ERRO: {e}")
            continue

        try:
            caminho = baixar_modelo_huggingface(registro.hub_id, DIRETORIO_MODELOS)
            print(f"   OK: {caminho}")
        except Exception as e:
            print(f"   ERRO: {e}")

    print("\n" + "=" * 60)
    print("Download concluido!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    baixar_todos_os_modelos()
