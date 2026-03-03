"""
Gerenciamento de VRAM e carregamento sequencial de modelos.
Otimizado para RTX 2070 SUPER (8GB VRAM).
"""

import gc
import logging
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# VRAM total disponivel (RTX 2070 SUPER)
VRAM_TOTAL_MB = 8192

# Margem de seguranca (manter livre para sistema e buffers)
MARGEM_SEGURANCA_MB = 1024


class GerenciadorGPU:
    """Gerencia VRAM e carregamento sequencial de modelos na GPU."""

    def __init__(self, dispositivo: Optional[str] = None):
        if dispositivo:
            self.dispositivo = torch.device(dispositivo)
        elif torch.cuda.is_available():
            self.dispositivo = torch.device("cuda")
        else:
            self.dispositivo = torch.device("cpu")
            logger.warning("CUDA nao disponivel. Usando CPU.")

        self._modelo_atual: Optional[str] = None

    @property
    def cuda_disponivel(self) -> bool:
        return self.dispositivo.type == "cuda"

    def obter_vram_livre(self) -> float:
        """Retorna VRAM livre em MB."""
        if not self.cuda_disponivel:
            return 0.0
        vram_livre = torch.cuda.mem_get_info(self.dispositivo)[0] / (1024 ** 2)
        return vram_livre

    def obter_vram_usada(self) -> float:
        """Retorna VRAM usada em MB."""
        if not self.cuda_disponivel:
            return 0.0
        return torch.cuda.memory_allocated(self.dispositivo) / (1024 ** 2)

    def obter_vram_total(self) -> float:
        """Retorna VRAM total em MB."""
        if not self.cuda_disponivel:
            return 0.0
        return torch.cuda.mem_get_info(self.dispositivo)[1] / (1024 ** 2)

    def limpar_vram(self):
        """Libera toda VRAM nao utilizada."""
        if self.cuda_disponivel:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(
                f"VRAM liberada. Uso atual: {self.obter_vram_usada():.0f}MB / "
                f"{self.obter_vram_total():.0f}MB"
            )

    def verificar_espaco(self, vram_necessaria_mb: int) -> bool:
        """Verifica se ha VRAM suficiente para carregar um modelo."""
        if not self.cuda_disponivel:
            return True  # CPU nao tem limite de VRAM
        vram_livre = self.obter_vram_livre()
        return vram_livre >= (vram_necessaria_mb + MARGEM_SEGURANCA_MB)

    @contextmanager
    def contexto_modelo(self, nome_modelo: str, vram_necessaria_mb: int = 0):
        """
        Context manager para carregar/descarregar modelos sequencialmente.

        Uso:
            with gpu_manager.contexto_modelo("spatial_vit", 250):
                modelo = carregar_modelo()
                resultado = modelo.inferir(imagem)
            # VRAM automaticamente liberada aqui
        """
        # Limpa VRAM antes de carregar novo modelo
        if self._modelo_atual and self._modelo_atual != nome_modelo:
            logger.info(f"Descarregando modelo anterior: {self._modelo_atual}")
            self.limpar_vram()

        # Verifica espaco
        if vram_necessaria_mb > 0 and not self.verificar_espaco(vram_necessaria_mb):
            logger.warning(
                f"VRAM possivelmente insuficiente para {nome_modelo}. "
                f"Necessario: {vram_necessaria_mb}MB, "
                f"Livre: {self.obter_vram_livre():.0f}MB. "
                f"Tentando mesmo assim..."
            )

        self._modelo_atual = nome_modelo
        logger.info(
            f"Carregando modelo: {nome_modelo} "
            f"(VRAM livre: {self.obter_vram_livre():.0f}MB)"
        )

        try:
            yield self.dispositivo
        finally:
            # Limpa VRAM apos uso
            self._modelo_atual = None
            self.limpar_vram()
            logger.info(
                f"Modelo {nome_modelo} descarregado. "
                f"VRAM livre: {self.obter_vram_livre():.0f}MB"
            )

    def mover_para_gpu(self, modelo: torch.nn.Module, fp16: bool = True) -> torch.nn.Module:
        """Move modelo para GPU com FP16 opcional."""
        if fp16 and self.cuda_disponivel:
            modelo = modelo.half()
        modelo = modelo.to(self.dispositivo)
        modelo.eval()
        return modelo

    def obter_info(self) -> dict:
        """Retorna informacoes da GPU para exibicao no dashboard."""
        if not self.cuda_disponivel:
            return {
                "disponivel": False,
                "dispositivo": "CPU",
                "nome": "N/A",
                "vram_total_mb": 0,
                "vram_usada_mb": 0,
                "vram_livre_mb": 0,
            }

        return {
            "disponivel": True,
            "dispositivo": str(self.dispositivo),
            "nome": torch.cuda.get_device_name(self.dispositivo),
            "vram_total_mb": round(self.obter_vram_total()),
            "vram_usada_mb": round(self.obter_vram_usada()),
            "vram_livre_mb": round(self.obter_vram_livre()),
        }


# Instancia global do gerenciador
gerenciador_gpu = GerenciadorGPU()
