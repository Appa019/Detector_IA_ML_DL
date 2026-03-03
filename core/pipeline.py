"""
Pipeline principal de deteccao de conteudo gerado por IA.
Orquestra o carregamento sequencial de modelos e a agregacao do ensemble.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image

from config.settings import CONFIG_ENSEMBLE, CONFIG_VIDEO, CONFIG_APP
from config.model_registry import REGISTRO_MODELOS, obter_modelos_por_tipo
from core.ensemble import AgregadorEnsemble, ResultadoEnsemble
from core.confidence import CalibradorConfianca
from models.base import DetectorBase, ResultadoDeteccao
from utils.gpu_manager import gerenciador_gpu
from analysis.ela import AnalisadorELA
from analysis.frequency import AnalisadorEspectral
from analysis.metadata import AnalisadorMetadados
from analysis.pixel_stats import AnalisadorPixels
from analysis.wavelet import AnalisadorWavelet

logger = logging.getLogger(__name__)


class PipelineDeteccao:
    """
    Pipeline principal que orquestra a deteccao de conteudo gerado por IA.

    Carrega modelos sequencialmente para economizar VRAM,
    coleta scores e agrega via ensemble.
    """

    def __init__(self):
        self.detectores: Dict[str, DetectorBase] = {}
        self.ensemble = AgregadorEnsemble()
        self.calibrador = CalibradorConfianca()
        self.analisador_espectral = AnalisadorEspectral()
        self.analisador_pixels = AnalisadorPixels()
        self.analisador_metadados = AnalisadorMetadados()
        self.analisador_ela = AnalisadorELA()
        self.analisador_wavelet = AnalisadorWavelet()
        self._inicializado = False

    def registrar_detector(self, detector: DetectorBase):
        """Registra um detector no pipeline."""
        self.detectores[detector.id_modelo] = detector
        logger.info(f"Detector registrado: {detector.id_modelo}")

    def inicializar(self):
        """Inicializa todos os detectores registrados."""
        from models.spatial_vit import DetectorViTEspacial
        from models.sdxl_detector import DetectorSDXL
        from models.ai_image_detector import DetectorAIImage
        from models.siglip_detector import DetectorSigLIP
        from models.frequency_analyzer import AnalisadorFrequencia
        from models.efficientnet_detector import DetectorEfficientNet

        detectores_padrao = [
            DetectorViTEspacial(),
            DetectorSDXL(),
            DetectorAIImage(),
            DetectorSigLIP(),
            AnalisadorFrequencia(),
            DetectorEfficientNet(),
        ]

        for detector in detectores_padrao:
            registro = REGISTRO_MODELOS.get(detector.id_modelo)
            if registro and registro.habilitado:
                self.registrar_detector(detector)

        self._inicializado = True
        logger.info(
            f"Pipeline inicializada com {len(self.detectores)} detectores: "
            f"{list(self.detectores.keys())}"
        )

    def analisar_imagem(
        self,
        imagem: Image.Image,
        modelos_habilitados: Optional[List[str]] = None,
        callback_progresso=None,
        caminho_arquivo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analisa uma imagem para detectar se foi gerada por IA.

        Args:
            imagem: Imagem PIL em RGB.
            modelos_habilitados: Lista de IDs de modelos a usar (None = todos).
            callback_progresso: Funcao callback(modelo_atual, progresso_pct).
            caminho_arquivo: Caminho opcional do arquivo para extracao de metadados EXIF.

        Returns:
            Dicionario com resultado do ensemble, scores individuais,
            metadados, visualizacoes e analises forenses.
        """
        if not self._inicializado:
            self.inicializar()

        inicio = time.time()
        resultados: List[ResultadoDeteccao] = []
        visualizacoes = {}

        # Filtra detectores por tipo imagem
        detectores_imagem = {
            k: v for k, v in self.detectores.items()
            if REGISTRO_MODELOS.get(k, None) is None
            or REGISTRO_MODELOS[k].tipo in ("imagem", "ambos")
        }

        if modelos_habilitados:
            detectores_imagem = {
                k: v for k, v in detectores_imagem.items()
                if k in modelos_habilitados
            }

        total = len(detectores_imagem)

        # Executa cada detector sequencialmente
        for i, (id_modelo, detector) in enumerate(detectores_imagem.items()):
            registro = REGISTRO_MODELOS.get(id_modelo)
            vram = registro.vram_fp16_mb if registro else 0

            if callback_progresso:
                callback_progresso({
                    "evento": "inicio_modelo",
                    "modelo_id": id_modelo,
                    "modelo_nome": detector.nome_modelo,
                    "indice": i,
                    "total": total,
                })

            try:
                with gerenciador_gpu.contexto_modelo(id_modelo, vram):
                    detector.carregar(str(gerenciador_gpu.dispositivo))
                    resultado = detector.detectar(imagem)
                    detector.descarregar()

                # Coleta mapa de calor se disponivel
                if resultado.mapa_calor is not None:
                    visualizacoes[f"heatmap_{id_modelo}"] = resultado.mapa_calor

                resultados.append(resultado)

                if callback_progresso:
                    callback_progresso({
                        "evento": "fim_modelo",
                        "modelo_id": id_modelo,
                        "modelo_nome": detector.nome_modelo,
                        "score": resultado.score,
                        "tempo_ms": resultado.tempo_inferencia_ms,
                        "indice": i,
                        "total": total,
                    })

                logger.info(
                    f"[{id_modelo}] Score: {resultado.score:.3f} "
                    f"({resultado.tempo_inferencia_ms:.0f}ms)"
                )

            except Exception as e:
                logger.error(f"Erro no detector {id_modelo}: {e}")
                if callback_progresso:
                    callback_progresso({
                        "evento": "erro_modelo",
                        "modelo_id": id_modelo,
                        "modelo_nome": detector.nome_modelo,
                        "erro": str(e),
                        "indice": i,
                        "total": total,
                    })
                continue

        # Agrega resultados do ensemble
        resultado_ensemble = self.ensemble.agregar(resultados)

        # Calibra score final
        score_calibrado = self.calibrador.calibrar(resultado_ensemble.score_final)

        # Calcula intervalo de confianca
        intervalo = self.calibrador.calcular_intervalo_confianca(
            score_calibrado, resultado_ensemble.incerteza
        )

        # Concordancia entre modelos
        concordancia = self.calibrador.calcular_concordancia(
            resultado_ensemble.scores_individuais
        )

        # --- Analises forenses ---
        imagem_np = np.array(imagem)
        analise_forense = {}

        if callback_progresso:
            callback_progresso({"evento": "inicio_forense", "etapa": "pixels"})

        try:
            # Analise de pixels
            score_pixels = self.analisador_pixels.calcular_score_pixels(imagem_np)
            histograma_rgb = self.analisador_pixels.calcular_histograma_rgb(imagem_np)
            noise_print = self.analisador_pixels.calcular_noise_print(imagem_np)
            stats_locais = self.analisador_pixels.calcular_estatisticas_locais(imagem_np)

            analise_forense["score_pixels"] = score_pixels
            analise_forense["histograma_rgb"] = {
                canal: valores.tolist() for canal, valores in histograma_rgb.items()
            }
            analise_forense["noise_print"] = noise_print
            analise_forense["uniformidade"] = stats_locais.get("uniformidade_local", 0.0)
        except Exception as e:
            logger.error(f"Erro na analise de pixels: {e}")

        if callback_progresso:
            callback_progresso({"evento": "inicio_forense", "etapa": "ela"})

        try:
            # Analise ELA
            score_ela = self.analisador_ela.calcular_score_ela(imagem_np)
            mapa_ela = self.analisador_ela.calcular_ela(imagem_np, qualidade_jpeg=90)
            analise_forense["score_ela"] = score_ela
            analise_forense["mapa_ela"] = mapa_ela
        except Exception as e:
            logger.error(f"Erro na analise ELA: {e}")

        if callback_progresso:
            callback_progresso({"evento": "inicio_forense", "etapa": "wavelet"})

        try:
            # Analise wavelet
            score_wavelet = self.analisador_wavelet.calcular_score_wavelet(imagem_np)
            features_wavelet = self.analisador_wavelet.extrair_features_wavelet(imagem_np)
            analise_forense["score_wavelet"] = score_wavelet
            analise_forense["features_wavelet"] = features_wavelet
        except Exception as e:
            logger.error(f"Erro na analise wavelet: {e}")

        if callback_progresso:
            callback_progresso({"evento": "inicio_forense", "etapa": "consistencia_ruido"})

        try:
            # Mapa de inconsistencia de ruido
            mapa_inconsistencia = self.analisador_pixels.calcular_mapa_inconsistencia(imagem_np)
            analise_forense["mapa_inconsistencia_ruido"] = mapa_inconsistencia
        except Exception as e:
            logger.error(f"Erro no mapa de inconsistencia de ruido: {e}")

        if callback_progresso:
            callback_progresso({"evento": "inicio_forense", "etapa": "espectro"})

        try:
            # Analise espectral (FFT)
            espectro_fft = self.analisador_espectral.calcular_fft_2d(imagem_np)
            features_freq = self.analisador_espectral.extrair_features_frequencia(imagem_np)

            analise_forense["espectro_fft"] = espectro_fft
            analise_forense["features_frequencia"] = features_freq
        except Exception as e:
            logger.error(f"Erro na analise espectral: {e}")

        if callback_progresso:
            callback_progresso({"evento": "inicio_forense", "etapa": "metadados"})

        try:
            # Metadados EXIF (requer caminho do arquivo)
            if caminho_arquivo:
                metadados = self.analisador_metadados.extrair_metadados(caminho_arquivo)
                indicadores_ia = self.analisador_metadados.analisar_indicadores_ia(metadados)
                analise_forense["metadados"] = metadados
                analise_forense["indicadores_ia"] = indicadores_ia
        except Exception as e:
            logger.error(f"Erro na analise de metadados: {e}")

        tempo_total = (time.time() - inicio) * 1000

        if callback_progresso:
            callback_progresso({"evento": "concluido"})

        return {
            "ensemble": resultado_ensemble,
            "score_calibrado": score_calibrado,
            "intervalo_confianca": intervalo,
            "concordancia": concordancia,
            "visualizacoes": visualizacoes,
            "analise_forense": analise_forense,
            "tempo_total_ms": tempo_total,
            "tipo": "imagem",
        }

    def analisar_video(
        self,
        caminho_video: str,
        callback_progresso=None,
    ) -> Dict[str, Any]:
        """
        Analisa um video para detectar deepfakes.

        Args:
            caminho_video: Caminho para o arquivo de video.
            callback_progresso: Funcao callback(etapa, progresso_pct).

        Returns:
            Dicionario com resultado agregado, timeline e frames suspeitos.
        """
        from processing.video_processor import ProcessadorVideo
        from processing.face_detector import DetectorFacial

        if not self._inicializado:
            self.inicializar()

        inicio = time.time()
        processador_video = ProcessadorVideo()
        detector_facial = DetectorFacial()

        # Extrai frames
        if callback_progresso:
            callback_progresso("Extraindo frames", 0)

        frames = processador_video.extrair_frames(
            caminho_video,
            intervalo=CONFIG_VIDEO.intervalo_frames,
            max_frames=CONFIG_VIDEO.max_frames,
        )

        if not frames:
            return {
                "ensemble": ResultadoEnsemble(
                    score_final=0.5,
                    classificacao="Indeterminado",
                    cor="#808080",
                    incerteza=1.0,
                ),
                "erro": "Nenhum frame extraido do video",
                "tipo": "video",
            }

        # Analisa cada frame
        resultados_frames = []
        scores_timeline = []

        for i, (frame, indice_frame) in enumerate(frames):
            if callback_progresso:
                progresso = ((i + 1) / len(frames)) * 100
                callback_progresso(f"Frame {i+1}/{len(frames)}", progresso)

            # Detecta rostos no frame
            imagem_pil = Image.fromarray(frame)
            rostos = detector_facial.detectar(imagem_pil)

            if rostos:
                # Analisa o rosto principal (maior area)
                rosto_principal = max(rostos, key=lambda r: r["area"])
                imagem_rosto = imagem_pil.crop(rosto_principal["bbox"])
            else:
                # Sem rosto - analisa frame inteiro
                imagem_rosto = imagem_pil

            # Executa pipeline de imagem neste frame
            resultado_frame = self.analisar_imagem(
                imagem_rosto,
                modelos_habilitados=["spatial_vit", "efficientnet_video", "frequency_analyzer"],
            )

            resultados_frames.append(resultado_frame["ensemble"])
            scores_timeline.append({
                "indice_frame": indice_frame,
                "score": resultado_frame["ensemble"].score_final,
                "classificacao": resultado_frame["ensemble"].classificacao,
                "tem_rosto": len(rostos) > 0,
                "num_rostos": len(rostos),
            })

        # Agrega temporalmente
        resultado_final = self.ensemble.agregar_temporal(resultados_frames)

        # Calibra
        score_calibrado = self.calibrador.calibrar(resultado_final.score_final)
        intervalo = self.calibrador.calcular_intervalo_confianca(
            score_calibrado, resultado_final.incerteza
        )

        # Frames mais suspeitos
        scores_timeline_sorted = sorted(
            scores_timeline, key=lambda x: x["score"], reverse=True
        )
        frames_suspeitos = scores_timeline_sorted[:5]

        tempo_total = (time.time() - inicio) * 1000

        return {
            "ensemble": resultado_final,
            "score_calibrado": score_calibrado,
            "intervalo_confianca": intervalo,
            "timeline": scores_timeline,
            "frames_suspeitos": frames_suspeitos,
            "total_frames": len(frames),
            "tempo_total_ms": tempo_total,
            "tipo": "video",
        }


# Instancia global do pipeline
pipeline = PipelineDeteccao()
