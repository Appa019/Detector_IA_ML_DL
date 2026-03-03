"""
Detector ViT Espacial baseado em 'prithivMLmods/Deep-Fake-Detector-v2-Model'.

Analisa padroes espaciais de texturas e artefatos visuais tipicos de imagens
geradas por IA (GANs, modelos de difusao, etc.).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from models.base import DetectorBase, ResultadoDeteccao
from utils.gpu_manager import gerenciador_gpu

logger = logging.getLogger(__name__)

# ID do modelo no HuggingFace Hub
_HUB_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

# Rotulos esperados na saida do modelo (Real -> 0.0, Fake -> 1.0)
_ROTULO_FALSO = "Fake"
_ROTULO_REAL = "Real"


class DetectorViTEspacial(DetectorBase):
    """
    Detector baseado em Vision Transformer (ViT) para analise espacial.

    Utiliza o modelo 'Deep-Fake-Detector-v2-Model' do HuggingFace para
    classificar imagens como reais ou geradas por IA, explorando padroes
    de textura e artefatos em nivel de pixel e patch.

    Atributos:
        _processador: AutoImageProcessor para pre-processamento das imagens.
        _dispositivo: Dispositivo PyTorch (cuda ou cpu) em uso.
        _ultimo_features: Ultimo tensor de features intermediarias (GradCAM).
        _gancho_features: Hook para capturar features da ultima camada.
    """

    def __init__(self) -> None:
        super().__init__(id_modelo="spatial_vit", nome_modelo="ViT Espacial (Deep Fake Detector v2)")
        self._processador: Optional[AutoImageProcessor] = None
        self._dispositivo: Optional[torch.device] = None
        self._ultimo_features: Optional[torch.Tensor] = None
        self._gancho_features: Optional[torch.utils.hooks.RemovableHook] = None

    def carregar(self, dispositivo: str = "cuda") -> None:
        """
        Carrega o processador e modelo ViT do HuggingFace Hub em FP16.

        Args:
            dispositivo: Dispositivo PyTorch alvo ('cuda' ou 'cpu').

        Raises:
            RuntimeError: Se o download ou carregamento do modelo falhar.
        """
        logger.info(f"Carregando {self.nome_modelo} de '{_HUB_ID}'...")
        inicio = time.time()

        self._dispositivo = gerenciador_gpu.dispositivo

        try:
            self._processador = AutoImageProcessor.from_pretrained(_HUB_ID)

            modelo_raw = AutoModelForImageClassification.from_pretrained(
                _HUB_ID,
                torch_dtype=torch.float16,
                ignore_mismatched_sizes=True,
            )

            self._modelo = gerenciador_gpu.mover_para_gpu(modelo_raw, fp16=True)

            # Registra hook na ultima camada do encoder para capturar features
            self._registrar_gancho_features()

            self._carregado = True
            tempo_carregamento = time.time() - inicio
            logger.info(
                f"{self.nome_modelo} carregado em {tempo_carregamento:.2f}s. "
                f"Dispositivo: {self._dispositivo}. "
                f"VRAM usada: {gerenciador_gpu.obter_vram_usada():.0f}MB"
            )

        except Exception as excecao:
            logger.error(f"Falha ao carregar {self.nome_modelo}: {excecao}")
            self._carregado = False
            raise RuntimeError(f"Nao foi possivel carregar o modelo ViT: {excecao}") from excecao

    def _registrar_gancho_features(self) -> None:
        """Registra forward hook para capturar features intermediarias (GradCAM)."""
        # Remove gancho anterior se existir
        if self._gancho_features is not None:
            self._gancho_features.remove()
            self._gancho_features = None

        # Tenta capturar a saida do ultimo bloco do encoder ViT
        try:
            ultima_camada = self._modelo.vit.encoder.layer[-1]

            def _capturar_features(
                modulo: torch.nn.Module,
                entrada: tuple,
                saida: torch.Tensor,
            ) -> None:
                # saida do bloco: (hidden_states, ...) — armazena o hidden_state
                self._ultimo_features = saida[0].detach()

            self._gancho_features = ultima_camada.register_forward_hook(_capturar_features)

        except AttributeError:
            # Arquitetura inesperada — desativa captura silenciosamente
            logger.warning(
                f"Nao foi possivel registrar hook de features em {self.nome_modelo}. "
                "GradCAM indisponivel para este modelo."
            )

    def descarregar(self) -> None:
        """Remove o modelo da GPU e libera toda VRAM associada."""
        if self._gancho_features is not None:
            self._gancho_features.remove()
            self._gancho_features = None

        if self._modelo is not None:
            del self._modelo
            self._modelo = None

        if self._processador is not None:
            del self._processador
            self._processador = None

        self._ultimo_features = None
        self._carregado = False
        gerenciador_gpu.limpar_vram()
        logger.info(f"{self.nome_modelo} descarregado da VRAM.")

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """
        Executa inferencia ViT na imagem e retorna score de IA.

        O modelo classifica a imagem em 'Real' ou 'Fake'. O score retornado
        e a probabilidade da classe 'Fake' (IA), normalizada entre 0.0 e 1.0.

        Args:
            imagem: Imagem PIL no formato RGB.

        Returns:
            ResultadoDeteccao com score 0.0 (real) a 1.0 (gerado por IA),
            confianca, metadados e features intermediarias para GradCAM.
        """
        if not self._carregado or self._modelo is None or self._processador is None:
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
            # Garante formato RGB
            imagem_rgb = imagem.convert("RGB")

            # Pre-processamento com o processador oficial do modelo
            entradas = self._processador(images=imagem_rgb, return_tensors="pt")
            entradas = {chave: tensor.to(self._dispositivo).half()
                        for chave, tensor in entradas.items()}

            # Inferencia com precisao automatica FP16
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    saidas = self._modelo(**entradas)

            # Extrai probabilidades via softmax
            logits = saidas.logits.float()  # Converte para FP32 para softmax estavel
            probabilidades = torch.softmax(logits, dim=-1).squeeze()

            # Mapeia indices de classe para rotulos do modelo
            mapa_rotulos = self._modelo.config.id2label
            score_ia = self._extrair_score_falso(probabilidades, mapa_rotulos)
            confianca = float(probabilidades.max().item())

            tempo_ms = (time.time() - inicio) * 1000.0

            # Coleta features intermediarias capturadas pelo hook
            features_capturadas = (
                self._ultimo_features.cpu() if self._ultimo_features is not None else None
            )

            metadados = {
                "hub_id": _HUB_ID,
                "rotulos": {str(idx): rotulo for idx, rotulo in mapa_rotulos.items()},
                "probabilidades": {
                    rotulo: float(probabilidades[idx].item())
                    for idx, rotulo in mapa_rotulos.items()
                },
                "dispositivo": str(self._dispositivo),
                "tem_features_gradcam": features_capturadas is not None,
            }

            logger.debug(
                f"{self.nome_modelo}: score_ia={score_ia:.4f}, "
                f"confianca={confianca:.4f}, tempo={tempo_ms:.1f}ms"
            )

            return ResultadoDeteccao(
                score=score_ia,
                confianca=confianca,
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados=metadados,
                mapa_calor=features_capturadas.numpy() if features_capturadas is not None else None,
                tempo_inferencia_ms=tempo_ms,
            )

        except Exception as excecao:
            tempo_ms = (time.time() - inicio) * 1000.0
            logger.error(f"Erro durante inferencia em {self.nome_modelo}: {excecao}", exc_info=True)
            return ResultadoDeteccao(
                score=0.5,
                confianca=0.0,
                id_modelo=self.id_modelo,
                nome_modelo=self.nome_modelo,
                metadados={"erro": str(excecao)},
                tempo_inferencia_ms=tempo_ms,
            )

    def _extrair_score_falso(
        self,
        probabilidades: torch.Tensor,
        mapa_rotulos: dict[int, str],
    ) -> float:
        """
        Extrai a probabilidade da classe 'Fake' do tensor de saida.

        Busca por rotulos que contenham 'fake', 'artificial', 'generated' ou 'ai'
        (case-insensitive) para robustez a variacoes de nomenclatura do modelo.

        Args:
            probabilidades: Tensor 1D com probabilidades de cada classe.
            mapa_rotulos: Dicionario {indice: nome_rotulo} do modelo.

        Returns:
            Score entre 0.0 (real) e 1.0 (IA).
        """
        _termos_falso = {"fake", "artificial", "generated", "ai", "deepfake", "synthetic"}
        _termos_real = {"real", "authentic", "genuine", "human"}

        score_falso = 0.0

        for indice, rotulo in mapa_rotulos.items():
            rotulo_lower = rotulo.lower()
            if any(termo in rotulo_lower for termo in _termos_falso):
                score_falso += float(probabilidades[indice].item())

        # Se nao encontrou rotulo 'falso', assume que indice 1 = Fake (convencao comum)
        if score_falso == 0.0:
            logger.warning(
                f"Nao foi possivel identificar rotulo 'Fake' no mapa: {mapa_rotulos}. "
                "Usando indice 1 como fallback."
            )
            if len(probabilidades) > 1:
                score_falso = float(probabilidades[1].item())
            else:
                score_falso = float(probabilidades[0].item())

        # Garante que o score esta no intervalo [0.0, 1.0]
        return max(0.0, min(1.0, score_falso))
