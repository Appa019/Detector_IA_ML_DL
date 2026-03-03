"""
Analisador de Frequencia para deteccao de artefatos espectrais.

Imagens geradas por IA (especialmente GANs e modelos de difusao) deixam
assinaturas caracteristicas no dominio de frequencia:
- GANs: picos em frequencias especificas (grade espectral)
- Diffusion models: atenuacao de altas frequencias
- Upsampling: repeticao periodica no espectro

Este modulo extrai features FFT/DCT + SRM e as classifica com um MLP leve.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.fft import dctn
from scipy.stats import kurtosis, skew

from analysis.srm_kernels import aplicar_filtros_srm
from analysis.ela import AnalisadorELA
from analysis.wavelet import AnalisadorWavelet
from config.settings import DIRETORIO_MODELOS
from models.base import DetectorBase, ResultadoDeteccao
from utils.gpu_manager import gerenciador_gpu

logger = logging.getLogger(__name__)

# Caminho local dos pesos do MLP treinado
_CAMINHO_PESOS_MLP = DIRETORIO_MODELOS / "frequency_analyzer" / "mlp_frequencia.pth"

# Dimensao do vetor de features extraido por imagem (expandido para SRM + wavelet)
_DIM_FEATURES = 320

# Numero de bins da media azimutal do espectro FFT
_NUM_BINS_AZIMUTAL = 64

# Numero de kernels SRM
_NUM_KERNELS_SRM = 30

# Numero de estatisticas por kernel SRM (mean, std, skew, kurtosis)
_STATS_POR_KERNEL = 4


class _MLPClassificador(nn.Module):
    """
    MLP leve para classificar features espectrais + SRM como real/IA.

    Arquitetura: [256] -> Linear(512) -> BN -> ReLU -> Dropout ->
                         Linear(256) -> BN -> ReLU -> Dropout ->
                         Linear(2) -> Logits
    """

    def __init__(self, dim_entrada: int = _DIM_FEATURES, num_classes: int = 2) -> None:
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dim_entrada, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.rede(x)


def _calcular_media_azimutal(magnitude: np.ndarray, num_bins: int = _NUM_BINS_AZIMUTAL) -> np.ndarray:
    """
    Calcula a media azimutal (radial average) do espectro de magnitude.

    Agrupa pixels do espectro por distancia ao centro e calcula a media
    em cada anel, produzindo um perfil 1D que captura distribuicao de energia
    por frequencia radial.

    Args:
        magnitude: Array 2D do espectro de magnitude FFT (escala log).
        num_bins: Numero de aneis radiais (resolucao do perfil).

    Returns:
        Array 1D com a media de magnitude em cada anel radial.
    """
    altura, largura = magnitude.shape
    cy, cx = altura // 2, largura // 2

    # Grade de distancias radiais (em pixels) de cada ponto ate o centro
    ys, xs = np.ogrid[:altura, :largura]
    distancias = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    # Raio maximo utilizavel (ate a borda mais proxima)
    raio_max = min(cy, cx)
    bins_radiais = np.linspace(0, raio_max, num_bins + 1)

    media_por_bin = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mascara = (distancias >= bins_radiais[i]) & (distancias < bins_radiais[i + 1])
        if mascara.any():
            media_por_bin[i] = float(magnitude[mascara].mean())

    return media_por_bin


def _extrair_features_srm(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Extrai features estatisticas dos 30 residuos SRM.

    Para cada kernel SRM, calcula 4 estatisticas (mean, std, skew, kurtosis)
    sobre o mapa de residuos, totalizando 120 features.

    Args:
        imagem_cinza: Array 2D em escala de cinza.

    Returns:
        Array 1D com 120 features SRM (float32).
    """
    residuos = aplicar_filtros_srm(imagem_cinza)
    features_srm = np.zeros(_NUM_KERNELS_SRM * _STATS_POR_KERNEL, dtype=np.float32)

    for i in range(_NUM_KERNELS_SRM):
        residuo_flat = residuos[i].ravel()
        offset = i * _STATS_POR_KERNEL
        features_srm[offset] = float(residuo_flat.mean())
        features_srm[offset + 1] = float(residuo_flat.std())
        features_srm[offset + 2] = float(skew(residuo_flat))
        features_srm[offset + 3] = float(kurtosis(residuo_flat))

    return features_srm


def _extrair_features_wavelet(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Extrai features estatisticas dos coeficientes wavelet.

    Energia por nivel, kurtosis e skewness das sub-bandas de detalhe.

    Args:
        imagem_cinza: Array 2D em escala de cinza.

    Returns:
        Array 1D com features wavelet (float32).
    """
    try:
        analisador = AnalisadorWavelet()
        features_dict = analisador.extrair_features_wavelet(imagem_cinza)
        # Converte dict ordenado para array
        valores = np.array(list(features_dict.values()), dtype=np.float32)
        return valores
    except Exception:
        # Fallback: retorna zeros se wavelet falhar
        return np.zeros(45, dtype=np.float32)


def _extrair_features_espectrais(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Extrai vetor de features espectrais + SRM de uma imagem em escala de cinza.

    Pipeline:
    1. FFT 2D -> espectro de magnitude (escala log)
    2. DCT 2D -> coeficientes de cosseno
    3. Media azimutal do espectro FFT (64 features)
    4. Estatisticas do espectro FFT: media, desvio, skewness, kurtose (4 features)
    5. Estatisticas dos coeficientes DCT: media, desvio, skewness, kurtose (4 features)
    6. Razao de energia em altas frequencias vs total FFT (1 feature)
    7. Razao de energia em altas frequencias vs total DCT (1 feature)
    8. Features SRM: 30 kernels x 4 stats = 120 features

    Total: 64 + 4 + 4 + 1 + 1 + 120 = 194 features (preenchido com zeros ate _DIM_FEATURES).

    Args:
        imagem_cinza: Array 2D uint8 ou float32 em escala de cinza.

    Returns:
        Vetor de features com dimensao _DIM_FEATURES (float32).
    """
    # Normaliza para [0.0, 1.0] em float64 para FFT precisa
    img_float = imagem_cinza.astype(np.float64)
    if img_float.max() > 1.0:
        img_float /= 255.0

    # --- FFT 2D ---
    espectro_fft = np.fft.fft2(img_float)
    espectro_deslocado = np.fft.fftshift(espectro_fft)
    magnitude_log = np.log1p(np.abs(espectro_deslocado)).astype(np.float32)

    # --- DCT 2D ---
    coeficientes_dct = dctn(img_float, type=2, norm="ortho").astype(np.float32)

    # --- Feature 1-64: Media azimutal do espectro FFT ---
    features_azimutais = _calcular_media_azimutal(magnitude_log, num_bins=_NUM_BINS_AZIMUTAL)

    # --- Features 65-68: Estatisticas do espectro FFT ---
    fft_flat = magnitude_log.ravel()
    stats_fft = np.array([
        float(fft_flat.mean()),
        float(fft_flat.std()),
        float(skew(fft_flat)),
        float(kurtosis(fft_flat)),
    ], dtype=np.float32)

    # --- Features 69-72: Estatisticas dos coeficientes DCT ---
    dct_flat = coeficientes_dct.ravel()
    stats_dct = np.array([
        float(dct_flat.mean()),
        float(dct_flat.std()),
        float(skew(dct_flat)),
        float(kurtosis(dct_flat)),
    ], dtype=np.float32)

    # --- Feature 73: Razao energia alta-frequencia FFT ---
    altura, largura = magnitude_log.shape
    cy, cx = altura // 2, largura // 2
    raio_baixa = min(cy, cx) // 4  # Quartil interno = baixas frequencias
    ys, xs = np.ogrid[:altura, :largura]
    distancias = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    mascara_alta_freq_fft = distancias > raio_baixa

    energia_total_fft = float(magnitude_log.sum()) + 1e-8
    energia_alta_fft = float(magnitude_log[mascara_alta_freq_fft].sum())
    razao_alta_fft = np.array([energia_alta_fft / energia_total_fft], dtype=np.float32)

    # --- Feature 74: Razao energia alta-frequencia DCT ---
    limiar_baixa_dct = max(1, len(dct_flat) // 4)
    energia_total_dct = float(np.abs(dct_flat).sum()) + 1e-8
    energia_alta_dct = float(np.abs(dct_flat[limiar_baixa_dct:]).sum())
    razao_alta_dct = np.array([energia_alta_dct / energia_total_dct], dtype=np.float32)

    # --- Features 75-194: Features SRM (30 kernels x 4 stats) ---
    features_srm = _extrair_features_srm(imagem_cinza)

    # --- Features 195+: Features Wavelet ---
    features_wavelet = _extrair_features_wavelet(imagem_cinza)

    # --- Concatenacao e normalizacao ---
    features_brutas = np.concatenate([
        features_azimutais,  # 64
        stats_fft,           # 4
        stats_dct,           # 4
        razao_alta_fft,      # 1
        razao_alta_dct,      # 1
        features_srm,        # 120
        features_wavelet,    # ~45
    ])  # Total: ~239

    # Preenche com zeros ate _DIM_FEATURES (320)
    features_padded = np.zeros(_DIM_FEATURES, dtype=np.float32)
    n = min(len(features_brutas), _DIM_FEATURES)
    features_padded[:n] = features_brutas[:n]

    # Normaliza por z-score (robusto a NaN e inf)
    features_padded = np.nan_to_num(features_padded, nan=0.0, posinf=1.0, neginf=-1.0)
    desvio = features_padded.std()
    if desvio > 1e-8:
        features_padded = (features_padded - features_padded.mean()) / desvio

    return features_padded


def _score_heuristico_frequencia(
    features: np.ndarray,
    imagem_cinza: np.ndarray | None = None,
) -> float:
    """
    Heuristica de score quando o MLP ainda nao foi treinado.

    Combina razao de alta frequencia FFT/DCT, estatisticas SRM,
    score wavelet e score ELA para uma estimativa robusta.

    Args:
        features: Vetor de features com dimensao _DIM_FEATURES.
        imagem_cinza: Array 2D em escala de cinza (para ELA/wavelet).

    Returns:
        Score entre 0.0 (provavel real) e 1.0 (provavel IA).
    """
    # --- Score espectral (FFT/DCT) ---
    indice_razao_fft = _NUM_BINS_AZIMUTAL  # = 64
    indice_razao_dct = _NUM_BINS_AZIMUTAL + 4 + 4  # = 72

    if indice_razao_fft < len(features):
        razao_fft = float(features[indice_razao_fft])
    else:
        razao_fft = 0.0

    if indice_razao_dct < len(features):
        razao_dct = float(features[indice_razao_dct])
    else:
        razao_dct = 0.0

    razao_media = (razao_fft + razao_dct) / 2.0
    score_espectral = float(1.0 / (1.0 + np.exp(razao_media)))

    # --- Score SRM ---
    inicio_srm = _NUM_BINS_AZIMUTAL + 4 + 4 + 1 + 1  # = 74
    indices_std_srm = [inicio_srm + i * _STATS_POR_KERNEL + 1 for i in range(_NUM_KERNELS_SRM)]
    indices_validos = [idx for idx in indices_std_srm if idx < len(features)]

    if indices_validos:
        std_media_srm = float(np.mean([features[idx] for idx in indices_validos]))
        score_srm = float(1.0 / (1.0 + np.exp(std_media_srm * 2.0)))
    else:
        score_srm = 0.5

    # --- Scores ELA e Wavelet (quando imagem disponivel) ---
    score_wavelet = 0.5
    score_ela = 0.5

    if imagem_cinza is not None:
        try:
            # Prepara imagem RGB para ELA (precisa de 3 canais)
            imagem_rgb = np.stack([imagem_cinza] * 3, axis=-1).astype(np.uint8)
            analisador_ela = AnalisadorELA()
            score_ela = analisador_ela.calcular_score_ela(imagem_rgb)
        except Exception:
            score_ela = 0.5

        try:
            analisador_wavelet = AnalisadorWavelet()
            score_wavelet = analisador_wavelet.calcular_score_wavelet(imagem_cinza)
        except Exception:
            score_wavelet = 0.5

    # Combina os 4 sinais com pesos calibrados
    score_ia = (
        0.30 * score_espectral
        + 0.25 * score_srm
        + 0.25 * score_wavelet
        + 0.20 * score_ela
    )

    return max(0.0, min(1.0, score_ia))


class AnalisadorFrequencia(DetectorBase):
    """
    Detector baseado em analise espectral (FFT/DCT) + SRM + MLP classificador.

    Detecta artefatos no dominio de frequencia tipicos de:
    - GANs: picos periodicos no espectro de magnitude (grade espectral)
    - Modelos de difusao: atenuacao caracteristica de altas frequencias
    - Upsampling artificial: repeticoes periodicas no espectro

    Integra 30 kernels SRM (Spatial Rich Model) para capturar padroes
    de residuo de manipulacao invisiveis ao olho humano.

    Quando os pesos do MLP nao estao disponiveis (modelo ainda nao treinado),
    utiliza uma heuristica baseada em FFT/DCT + estatisticas SRM.

    Atributos:
        _mlp: Rede MLP classificadora de features espectrais + SRM.
        _mlp_treinado: Indica se o MLP foi carregado com pesos pre-treinados.
        _dispositivo_mlp: Dispositivo para inferencia do MLP (CPU ou GPU).
    """

    def __init__(self) -> None:
        super().__init__(
            id_modelo="frequency_analyzer",
            nome_modelo="Analisador de Frequencia (FFT/DCT + SRM + MLP)",
        )
        self._mlp: Optional[_MLPClassificador] = None
        self._mlp_treinado: bool = False
        self._dispositivo_mlp: torch.device = torch.device("cpu")

    def carregar(self, dispositivo: str = "cuda") -> None:
        """
        Carrega ou inicializa o MLP classificador de features espectrais + SRM.

        Tenta carregar pesos pre-treinados de `_CAMINHO_PESOS_MLP`.
        Se o arquivo nao existir, cria um MLP sem treinamento (pesos aleatorios).
        Neste caso, o metodo `detectar` usara a heuristica de frequencia.

        O MLP e pequeno e roda na CPU para minimizar uso de VRAM.

        Args:
            dispositivo: Dispositivo preferido (ignorado para o MLP, que usa CPU).
        """
        logger.info(f"Carregando {self.nome_modelo}...")
        inicio = time.time()

        self._dispositivo_mlp = torch.device("cpu")

        self._mlp = _MLPClassificador(dim_entrada=_DIM_FEATURES, num_classes=2)

        caminho_pesos = Path(_CAMINHO_PESOS_MLP)

        if caminho_pesos.exists():
            try:
                estado_salvo = torch.load(caminho_pesos, map_location="cpu", weights_only=True)
                self._mlp.load_state_dict(estado_salvo)
                self._mlp_treinado = True
                logger.info(f"Pesos do MLP carregados de: {caminho_pesos}")
            except Exception as excecao:
                logger.warning(
                    f"Falha ao carregar pesos do MLP ({caminho_pesos}): {excecao}. "
                    "Usando heuristica de frequencia como fallback."
                )
                self._mlp_treinado = False
        else:
            self._mlp_treinado = False
            logger.warning(
                f"Pesos do MLP nao encontrados em '{caminho_pesos}'. "
                "Usando heuristica de frequencia ate treinamento ser realizado."
            )

        self._mlp.eval()
        self._carregado = True

        tempo_carregamento = time.time() - inicio
        logger.info(
            f"{self.nome_modelo} {'(treinado)' if self._mlp_treinado else '(sem treino - heuristica)'} "
            f"carregado em {tempo_carregamento:.3f}s."
        )

    def descarregar(self) -> None:
        """Remove o MLP da memoria."""
        if self._mlp is not None:
            del self._mlp
            self._mlp = None

        self._mlp_treinado = False
        self._carregado = False
        logger.info(f"{self.nome_modelo} descarregado da memoria.")

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """
        Analisa o espectro de frequencia e residuos SRM da imagem.

        Pipeline:
        1. Converte para escala de cinza
        2. Extrai features FFT/DCT (74 features) + SRM (120 features)
        3. Classifica com MLP (se treinado) ou heuristica combinada

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score 0.0 (real) a 1.0 (gerado por IA),
            incluindo features espectrais nos metadados.
        """
        if not self._carregado or self._mlp is None:
            logger.error(f"{self.nome_modelo} nao esta carregado. Retornando score incerto.")
            return ResultadoDeteccao(
                score=0.5,
                confianca=0.0,
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados={"erro": "Modelo nao carregado"},
            )

        inicio = time.time()

        try:
            # --- Pre-processamento ---
            imagem_cinza = np.array(imagem.convert("L"), dtype=np.float32)

            # Garante dimensoes minimas para FFT significativa
            altura, largura = imagem_cinza.shape
            if altura < 32 or largura < 32:
                logger.warning(
                    f"Imagem muito pequena ({largura}x{altura}). "
                    "Redimensionando para 64x64 para analise espectral."
                )
                imagem_resized = imagem.convert("L").resize((64, 64), Image.LANCZOS)
                imagem_cinza = np.array(imagem_resized, dtype=np.float32)

            # --- Extracao de features espectrais + SRM ---
            features = _extrair_features_espectrais(imagem_cinza)

            # --- Classificacao ---
            if self._mlp_treinado:
                score_ia, confianca = self._classificar_com_mlp(features)
                metodo = "MLP treinado (FFT/DCT + SRM)"
            else:
                score_ia = _score_heuristico_frequencia(features, imagem_cinza)
                confianca = 0.45  # Confianca melhorada com ELA + wavelet
                metodo = "heuristica de frequencia + SRM + wavelet + ELA (MLP sem treino)"

            tempo_ms = (time.time() - inicio) * 1000.0

            metadados = {
                "metodo": metodo,
                "mlp_treinado": self._mlp_treinado,
                "dimensao_features": int(len(features)),
                "num_features_fft_dct": 74,
                "num_features_srm": _NUM_KERNELS_SRM * _STATS_POR_KERNEL,
                "razao_alta_frequencia_fft": float(features[_NUM_BINS_AZIMUTAL]),
                "razao_alta_frequencia_dct": float(features[_NUM_BINS_AZIMUTAL + 8]),
                "tamanho_imagem_cinza": f"{imagem_cinza.shape[1]}x{imagem_cinza.shape[0]}",
            }

            logger.debug(
                f"{self.nome_modelo}: score_ia={score_ia:.4f}, "
                f"confianca={confianca:.4f}, metodo='{metodo}', tempo={tempo_ms:.1f}ms"
            )

            return ResultadoDeteccao(
                score=score_ia,
                confianca=confianca,
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados=metadados,
                tempo_inferencia_ms=tempo_ms,
            )

        except Exception as excecao:
            tempo_ms = (time.time() - inicio) * 1000.0
            logger.error(
                f"Erro durante analise espectral em {self.nome_modelo}: {excecao}", exc_info=True
            )
            return ResultadoDeteccao(
                score=0.5,
                confianca=0.0,
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados={"erro": str(excecao)},
                tempo_inferencia_ms=tempo_ms,
            )

    def _classificar_com_mlp(self, features: np.ndarray) -> tuple[float, float]:
        """
        Classifica o vetor de features com o MLP treinado.

        Args:
            features: Vetor de features com dimensao _DIM_FEATURES.

        Returns:
            Tupla (score_ia, confianca), ambos entre 0.0 e 1.0.
        """
        tensor_features = torch.from_numpy(features).unsqueeze(0)  # [1, DIM]
        tensor_features = tensor_features.to(self._dispositivo_mlp)

        with torch.no_grad():
            logits = self._mlp(tensor_features)
            probabilidades = torch.softmax(logits, dim=-1).squeeze()

        # Indice 1 = classe IA/Fake (convencao de treinamento)
        score_ia = float(probabilidades[1].item()) if len(probabilidades) > 1 else 0.5
        confianca = float(probabilidades.max().item())

        return max(0.0, min(1.0, score_ia)), confianca
