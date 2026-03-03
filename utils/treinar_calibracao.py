"""
Script de treino e calibracao do ensemble de deteccao de IA.

Executa as seguintes etapas:
1. Carrega dataset de calibracao (imagens reais + geradas por IA)
2. Roda cada detector em cada imagem, coleta scores
3. Treina cabecas lineares do DINOv2 e CLIP (sondas)
4. Treina MLP do frequency_analyzer com features SRM expandidas
5. Treina meta-learner GradientBoosting sobre scores coletados
6. Treina calibracao isotonica no conjunto de validacao
7. Salva tudo em models_cache/

Uso:
    python utils/treinar_calibracao.py --dataset_dir <caminho> [--epochs 10] [--batch_size 32]

Estrutura esperada do dataset:
    dataset_dir/
        real/         # Imagens fotograficas reais
            *.jpg
            *.png
        ia/           # Imagens geradas por IA (SDXL, FLUX, MJ, DALL-E 3)
            *.jpg
            *.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DIRETORIO_MODELOS, CONFIG_APP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetCalibracao(Dataset):
    """Dataset para calibracao com imagens reais e geradas por IA."""

    def __init__(self, caminhos: list[Path], rotulos: list[int]) -> None:
        self.caminhos = caminhos
        self.rotulos = rotulos

    def __len__(self) -> int:
        return len(self.caminhos)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, Path]:
        imagem = Image.open(self.caminhos[idx]).convert("RGB")
        return imagem, self.rotulos[idx], self.caminhos[idx]


def carregar_dataset(diretorio: Path) -> tuple[list[Path], list[int]]:
    """
    Carrega caminhos de imagens e rotulos do diretorio.

    Args:
        diretorio: Caminho raiz com subpastas 'real/' e 'ia/'.

    Returns:
        Tupla (caminhos, rotulos) onde rotulo 0=real, 1=IA.
    """
    extensoes = set(CONFIG_APP.extensoes_imagem)
    caminhos = []
    rotulos = []

    dir_real = diretorio / "real"
    dir_ia = diretorio / "ia"

    if not dir_real.exists() or not dir_ia.exists():
        raise FileNotFoundError(
            f"Dataset deve ter subpastas 'real/' e 'ia/' em {diretorio}"
        )

    for arquivo in sorted(dir_real.iterdir()):
        if arquivo.suffix.lower() in extensoes:
            caminhos.append(arquivo)
            rotulos.append(0)  # Real

    for arquivo in sorted(dir_ia.iterdir()):
        if arquivo.suffix.lower() in extensoes:
            caminhos.append(arquivo)
            rotulos.append(1)  # IA

    logger.info(
        f"Dataset carregado: {sum(1 for r in rotulos if r == 0)} reais, "
        f"{sum(1 for r in rotulos if r == 1)} IA, total={len(caminhos)}"
    )

    return caminhos, rotulos


def treinar_cabeca_dinov2(
    caminhos_treino: list[Path],
    rotulos_treino: list[int],
    caminhos_val: list[Path],
    rotulos_val: list[int],
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> None:
    """Treina a cabeca linear do DINOv2 sobre embeddings congelados."""
    from transformers import AutoImageProcessor, AutoModel

    logger.info("=== Treinando cabeca DINOv2 ===")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processador = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    backbone = AutoModel.from_pretrained(
        "facebook/dinov2-small", torch_dtype=torch.float16
    ).to(dispositivo).eval()

    for param in backbone.parameters():
        param.requires_grad = False

    cabeca = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 2)).to(dispositivo)
    otimizador = torch.optim.Adam(cabeca.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        cabeca.train()
        perdas = []

        for i in range(0, len(caminhos_treino), batch_size):
            batch_caminhos = caminhos_treino[i:i + batch_size]
            batch_rotulos = rotulos_treino[i:i + batch_size]

            imagens = [Image.open(p).convert("RGB") for p in batch_caminhos]
            entradas = processador(images=imagens, return_tensors="pt")
            pixel_values = entradas["pixel_values"].to(dispositivo).half()
            alvos = torch.tensor(batch_rotulos, dtype=torch.long, device=dispositivo)

            with torch.no_grad():
                saidas = backbone(pixel_values=pixel_values)
                cls_tokens = saidas.last_hidden_state[:, 0, :].float()

            logits = cabeca(cls_tokens)
            perda = criterio(logits, alvos)

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())

        # Validacao
        cabeca.eval()
        corretos = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(caminhos_val), batch_size):
                batch_caminhos = caminhos_val[i:i + batch_size]
                batch_rotulos = rotulos_val[i:i + batch_size]

                imagens = [Image.open(p).convert("RGB") for p in batch_caminhos]
                entradas = processador(images=imagens, return_tensors="pt")
                pixel_values = entradas["pixel_values"].to(dispositivo).half()
                alvos = torch.tensor(batch_rotulos, dtype=torch.long, device=dispositivo)

                saidas = backbone(pixel_values=pixel_values)
                cls_tokens = saidas.last_hidden_state[:, 0, :].float()
                logits = cabeca(cls_tokens)
                predicoes = logits.argmax(dim=-1)
                corretos += (predicoes == alvos).sum().item()
                total += len(alvos)

        acuracia = corretos / total if total > 0 else 0.0
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Perda: {np.mean(perdas):.4f}, Acuracia val: {acuracia:.4f}"
        )

    # Salva cabeca
    caminho_salvar = DIRETORIO_MODELOS / "dinov2" / "cabeca_dinov2.pth"
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cabeca.state_dict(), caminho_salvar)
    logger.info(f"Cabeca DINOv2 salva em: {caminho_salvar}")

    del backbone, cabeca
    torch.cuda.empty_cache()


def treinar_sonda_clip(
    caminhos_treino: list[Path],
    rotulos_treino: list[int],
    caminhos_val: list[Path],
    rotulos_val: list[int],
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> None:
    """Treina a sonda linear do CLIP UniversalFakeDetect."""
    from transformers import CLIPModel, CLIPProcessor

    logger.info("=== Treinando sonda CLIP UniversalFakeDetect ===")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processador = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    backbone = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=torch.float16
    ).to(dispositivo).eval()

    for param in backbone.parameters():
        param.requires_grad = False

    sonda = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 2)).to(dispositivo)
    otimizador = torch.optim.Adam(sonda.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sonda.train()
        perdas = []

        for i in range(0, len(caminhos_treino), batch_size):
            batch_caminhos = caminhos_treino[i:i + batch_size]
            batch_rotulos = rotulos_treino[i:i + batch_size]

            imagens = [Image.open(p).convert("RGB") for p in batch_caminhos]
            entradas = processador(images=imagens, return_tensors="pt")
            pixel_values = entradas["pixel_values"].to(dispositivo).half()
            alvos = torch.tensor(batch_rotulos, dtype=torch.long, device=dispositivo)

            with torch.no_grad():
                features = backbone.get_image_features(pixel_values=pixel_values)
                features = nn.functional.normalize(features, p=2, dim=-1).float()

            logits = sonda(features)
            perda = criterio(logits, alvos)

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())

        # Validacao
        sonda.eval()
        corretos = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(caminhos_val), batch_size):
                batch_caminhos = caminhos_val[i:i + batch_size]
                batch_rotulos = rotulos_val[i:i + batch_size]

                imagens = [Image.open(p).convert("RGB") for p in batch_caminhos]
                entradas = processador(images=imagens, return_tensors="pt")
                pixel_values = entradas["pixel_values"].to(dispositivo).half()
                alvos = torch.tensor(batch_rotulos, dtype=torch.long, device=dispositivo)

                features = backbone.get_image_features(pixel_values=pixel_values)
                features = nn.functional.normalize(features, p=2, dim=-1).float()
                logits = sonda(features)
                predicoes = logits.argmax(dim=-1)
                corretos += (predicoes == alvos).sum().item()
                total += len(alvos)

        acuracia = corretos / total if total > 0 else 0.0
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Perda: {np.mean(perdas):.4f}, Acuracia val: {acuracia:.4f}"
        )

    # Salva sonda
    caminho_salvar = DIRETORIO_MODELOS / "clip_ufd" / "cabeca_clip_ufd.pth"
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sonda.state_dict(), caminho_salvar)
    logger.info(f"Sonda CLIP UFD salva em: {caminho_salvar}")

    del backbone, sonda
    torch.cuda.empty_cache()


def treinar_mlp_frequencia(
    caminhos_treino: list[Path],
    rotulos_treino: list[int],
    caminhos_val: list[Path],
    rotulos_val: list[int],
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> None:
    """Treina o MLP do frequency_analyzer com features FFT/DCT + SRM."""
    from models.frequency_analyzer import _extrair_features_espectrais, _DIM_FEATURES

    logger.info("=== Treinando MLP de frequencia (FFT/DCT + SRM) ===")

    # Extrai features de todas as imagens
    def extrair_todas(caminhos):
        features_lista = []
        for caminho in caminhos:
            imagem = Image.open(caminho).convert("L")
            imagem_np = np.array(imagem, dtype=np.float32)
            features = _extrair_features_espectrais(imagem_np)
            features_lista.append(features)
        return np.array(features_lista)

    logger.info("Extraindo features espectrais + SRM do treino...")
    X_treino = extrair_todas(caminhos_treino)
    y_treino = np.array(rotulos_treino)

    logger.info("Extraindo features espectrais + SRM da validacao...")
    X_val = extrair_todas(caminhos_val)
    y_val = np.array(rotulos_val)

    # Treina MLP
    from models.frequency_analyzer import _MLPClassificador
    mlp = _MLPClassificador(dim_entrada=_DIM_FEATURES, num_classes=2)
    otimizador = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        mlp.train()
        perdas = []

        indices = np.random.permutation(len(X_treino))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = torch.from_numpy(X_treino[batch_idx])
            y_batch = torch.from_numpy(y_treino[batch_idx]).long()

            logits = mlp(X_batch)
            perda = criterio(logits, y_batch)

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())

        # Validacao
        mlp.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val)
            y_val_tensor = torch.from_numpy(y_val).long()
            logits_val = mlp(X_val_tensor)
            predicoes = logits_val.argmax(dim=-1)
            acuracia = (predicoes == y_val_tensor).float().mean().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Perda: {np.mean(perdas):.4f}, Acuracia val: {acuracia:.4f}"
            )

    # Salva MLP
    caminho_salvar = DIRETORIO_MODELOS / "frequency_analyzer" / "mlp_frequencia.pth"
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mlp.state_dict(), caminho_salvar)
    logger.info(f"MLP de frequencia salvo em: {caminho_salvar}")


def coletar_scores_ensemble(
    caminhos: list[Path],
    rotulos: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roda todos os detectores em cada imagem e coleta scores.

    Returns:
        Tupla (features_meta, scores_brutos, rotulos) onde:
        - features_meta: array (N, 2*M+4) para o meta-learner
        - scores_brutos: array (N,) de scores medios brutos
        - rotulos: array (N,) de rotulos
    """
    from core.pipeline import PipelineDeteccao

    logger.info("=== Coletando scores do ensemble para meta-learner ===")

    pipeline = PipelineDeteccao()
    pipeline.inicializar()

    features_meta_lista = []
    scores_brutos = []

    for i, (caminho, rotulo) in enumerate(zip(caminhos, rotulos)):
        if (i + 1) % 50 == 0:
            logger.info(f"Processando imagem {i + 1}/{len(caminhos)}...")

        imagem = Image.open(caminho).convert("RGB")
        resultado = pipeline.analisar_imagem(imagem)

        # Extrai scores e confiancas dos resultados individuais
        ensemble_result = resultado["ensemble"]
        scores_ind = []
        confs_ind = []

        for det in ensemble_result.resultados_detalhados:
            scores_ind.append(det.score)
            confs_ind.append(det.confianca)

        scores_np = np.array(scores_ind, dtype=np.float64)
        confs_np = np.array(confs_ind, dtype=np.float64)

        pares = np.empty(len(scores_ind) * 2, dtype=np.float64)
        pares[0::2] = scores_np
        pares[1::2] = confs_np

        stats = np.array([
            float(scores_np.std()) if len(scores_np) > 1 else 0.0,
            float(scores_np.max()),
            float(scores_np.min()),
            float(scores_np.max() - scores_np.min()),
        ], dtype=np.float64)

        features_meta_lista.append(np.concatenate([pares, stats]))
        scores_brutos.append(ensemble_result.score_final)

    return (
        np.array(features_meta_lista),
        np.array(scores_brutos),
        np.array(rotulos),
    )


def treinar_meta_learner(
    features: np.ndarray,
    rotulos: np.ndarray,
) -> None:
    """Treina o meta-learner GradientBoosting."""
    from core.ensemble import MetaAprendizEnsemble

    logger.info("=== Treinando meta-learner GradientBoosting ===")
    meta = MetaAprendizEnsemble()
    metricas = meta.treinar(features, rotulos)
    logger.info(f"Meta-learner treinado. Metricas: {metricas}")


def treinar_calibracao_isotonica(
    scores_brutos: np.ndarray,
    rotulos: np.ndarray,
) -> None:
    """Treina calibracao isotonica."""
    from core.confidence import CalibradorConfianca

    logger.info("=== Treinando calibracao isotonica ===")
    calibrador = CalibradorConfianca()
    metricas = calibrador.treinar_calibracao(scores_brutos, rotulos)
    logger.info(f"Calibracao treinada. Metricas: {metricas}")


def main():
    parser = argparse.ArgumentParser(
        description="Treina cabecas, MLP, meta-learner e calibracao do ensemble"
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Diretorio raiz do dataset (com subpastas real/ e ia/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero de epochs para treino das cabecas (padrao: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamanho do batch (padrao: 32)",
    )
    parser.add_argument(
        "--etapas",
        nargs="+",
        default=["dinov2", "clip", "mlp", "meta", "calibracao"],
        help="Etapas a executar (padrao: todas)",
    )

    args = parser.parse_args()

    # Carrega dataset
    caminhos, rotulos = carregar_dataset(args.dataset_dir)

    # Split treino/validacao
    cam_treino, cam_val, rot_treino, rot_val = train_test_split(
        caminhos, rotulos, test_size=0.2, random_state=42, stratify=rotulos,
    )

    logger.info(
        f"Split: {len(cam_treino)} treino, {len(cam_val)} validacao"
    )

    inicio = time.time()

    # Etapa 1: Treina cabeca DINOv2
    if "dinov2" in args.etapas:
        treinar_cabeca_dinov2(
            cam_treino, rot_treino, cam_val, rot_val,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    # Etapa 2: Treina sonda CLIP
    if "clip" in args.etapas:
        treinar_sonda_clip(
            cam_treino, rot_treino, cam_val, rot_val,
            epochs=args.epochs, batch_size=max(1, args.batch_size // 2),
        )

    # Etapa 3: Treina MLP de frequencia
    if "mlp" in args.etapas:
        treinar_mlp_frequencia(
            cam_treino, rot_treino, cam_val, rot_val,
            epochs=30, batch_size=args.batch_size * 2,
        )

    # Etapa 4: Coleta scores e treina meta-learner + calibracao
    if "meta" in args.etapas or "calibracao" in args.etapas:
        features_meta, scores_brutos, rotulos_np = coletar_scores_ensemble(
            cam_val, rot_val,
        )

        if "meta" in args.etapas:
            treinar_meta_learner(features_meta, rotulos_np)

        if "calibracao" in args.etapas:
            treinar_calibracao_isotonica(scores_brutos, rotulos_np)

    tempo_total = time.time() - inicio
    logger.info(f"Treino completo em {tempo_total:.1f}s")


if __name__ == "__main__":
    main()
