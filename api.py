"""
API REST para o Detector de Conteudo Gerado por IA.
Backend FastAPI que expoe os endpoints de analise.
"""

import asyncio
import base64
import json
import logging
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CONFIG_APP, CONFIG_ENSEMBLE
from config.model_registry import REGISTRO_MODELOS
from utils.gpu_manager import gerenciador_gpu
from PIL import Image
import numpy as np
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

def _numpy_para_base64_png(array: np.ndarray) -> str:
    """Converte um array numpy (imagem/heatmap) para string base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(array, cmap="inferno" if array.ndim == 2 else None)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _noise_print_para_base64(noise: np.ndarray) -> str:
    """Converte noise print normalizado [-1,1] para base64 PNG com colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(noise, cmap="seismic", vmin=-1, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


app = FastAPI(
    title="Detector de Conteudo Gerado por IA",
    description="API para analise de imagens e videos usando ensemble de modelos de Deep Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline global (lazy init)
_pipeline = None


def obter_pipeline():
    global _pipeline
    if _pipeline is None:
        from core.pipeline import PipelineDeteccao
        _pipeline = PipelineDeteccao()
        _pipeline.inicializar()
    return _pipeline


# --- Modelos Pydantic ---

class InfoGPU(BaseModel):
    disponivel: bool
    dispositivo: str
    nome: str
    vram_total_mb: int
    vram_usada_mb: int
    vram_livre_mb: int


class InfoModelo(BaseModel):
    id: str
    nome_exibicao: str
    arquitetura: str
    vram_fp16_mb: int
    papel: str
    tipo: str
    habilitado: bool


class ScoreIndividual(BaseModel):
    modelo_id: str
    modelo_nome: str
    score: float
    tempo_ms: float


class ResultadoAnalise(BaseModel):
    score_final: float
    score_calibrado: float
    classificacao: str
    cor: str
    incerteza: float
    intervalo_confianca: List[float]
    concordancia: float
    scores_individuais: List[ScoreIndividual]
    tempo_total_ms: float
    tipo: str
    # Visualizacoes forenses (base64 PNG)
    heatmap_gradcam: Optional[str] = None
    espectro_fft: Optional[str] = None
    noise_print: Optional[str] = None
    # Histograma RGB (3 arrays de 256 valores)
    histograma_rgb: Optional[dict] = None
    # Analise de frequencia
    features_frequencia: Optional[dict] = None
    # Estatisticas de pixels
    score_pixels: Optional[float] = None
    uniformidade: Optional[float] = None
    # Novas analises forenses
    score_ela: Optional[float] = None
    score_wavelet: Optional[float] = None
    mapa_ela: Optional[str] = None
    mapa_inconsistencia_ruido: Optional[str] = None
    # Metadados EXIF e indicadores de IA
    metadados: Optional[dict] = None
    indicadores_ia: Optional[dict] = None


class ResultadoVideo(ResultadoAnalise):
    timeline: list
    frames_suspeitos: list
    total_frames: int


class StatusAPI(BaseModel):
    status: str
    versao: str
    gpu: InfoGPU
    modelos_disponiveis: int


# --- Endpoints ---

@app.get("/api/status", response_model=StatusAPI)
async def obter_status():
    """Retorna status da API e informacoes do hardware."""
    info_gpu = gerenciador_gpu.obter_info()
    return StatusAPI(
        status="online",
        versao="1.0.0",
        gpu=InfoGPU(**info_gpu),
        modelos_disponiveis=len([m for m in REGISTRO_MODELOS.values() if m.habilitado]),
    )


@app.get("/api/modelos", response_model=List[InfoModelo])
async def listar_modelos():
    """Lista todos os modelos disponiveis no ensemble."""
    return [
        InfoModelo(
            id=m.id,
            nome_exibicao=m.nome_exibicao,
            arquitetura=m.arquitetura,
            vram_fp16_mb=m.vram_fp16_mb,
            papel=m.papel,
            tipo=m.tipo,
            habilitado=m.habilitado,
        )
        for m in REGISTRO_MODELOS.values()
    ]


@app.post("/api/analisar/imagem", response_model=ResultadoAnalise)
async def analisar_imagem(
    arquivo: UploadFile = File(...),
    modelos: Optional[str] = Query(None, description="IDs dos modelos separados por virgula"),
):
    """Analisa uma imagem para detectar conteudo gerado por IA."""
    # Valida extensao
    extensao = Path(arquivo.filename).suffix.lower()
    if extensao not in CONFIG_APP.extensoes_imagem:
        raise HTTPException(
            status_code=400,
            detail=f"Formato nao suportado: {extensao}. Use: {', '.join(CONFIG_APP.extensoes_imagem)}"
        )

    try:
        conteudo = await arquivo.read()
        imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler imagem: {str(e)}")

    # Salva temporariamente para extracao de metadados EXIF
    caminho_temp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extensao, delete=False) as tmp:
            tmp.write(conteudo)
            caminho_temp = tmp.name
    except Exception:
        logger.warning("Nao foi possivel salvar arquivo temporario para metadados")

    # Parse modelos habilitados
    modelos_habilitados = None
    if modelos:
        modelos_habilitados = [m.strip() for m in modelos.split(",")]

    try:
        pipeline = obter_pipeline()
        resultado = pipeline.analisar_imagem(
            imagem,
            modelos_habilitados=modelos_habilitados,
            caminho_arquivo=caminho_temp,
        )
    except Exception as e:
        logger.error(f"Erro na analise: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro na analise: {str(e)}")
    finally:
        if caminho_temp:
            Path(caminho_temp).unlink(missing_ok=True)

    # Monta resposta
    scores_individuais = []
    for res in resultado["ensemble"].resultados_detalhados:
        scores_individuais.append(ScoreIndividual(
            modelo_id=res.id_modelo,
            modelo_nome=res.nome_modelo,
            score=round(res.score, 4),
            tempo_ms=round(res.tempo_inferencia_ms, 1),
        ))

    # Serializa dados forenses
    forense = resultado.get("analise_forense", {})
    heatmap_gradcam = None
    espectro_fft = None
    noise_print_b64 = None
    histograma_rgb = None
    features_frequencia = None
    score_pixels = None
    uniformidade = None
    metadados = None
    indicadores_ia = None

    try:
        # GradCAM: pega o primeiro heatmap disponivel
        for chave, mapa in resultado.get("visualizacoes", {}).items():
            if mapa is not None and isinstance(mapa, np.ndarray) and mapa.any():
                heatmap_gradcam = _numpy_para_base64_png(mapa)
                break

        # Espectro FFT
        if "espectro_fft" in forense and forense["espectro_fft"] is not None:
            espectro_fft = _numpy_para_base64_png(forense["espectro_fft"])

        # Noise print
        if "noise_print" in forense and forense["noise_print"] is not None:
            noise_print_b64 = _noise_print_para_base64(forense["noise_print"])

        # Histograma RGB
        histograma_rgb = forense.get("histograma_rgb")

        # Features de frequencia
        features_frequencia = forense.get("features_frequencia")

        # Score de pixels e uniformidade
        score_pixels = forense.get("score_pixels")
        uniformidade = forense.get("uniformidade")

        # ELA
        score_ela = forense.get("score_ela")
        mapa_ela_b64 = None
        if "mapa_ela" in forense and forense["mapa_ela"] is not None:
            mapa_ela_b64 = _numpy_para_base64_png(forense["mapa_ela"])

        # Wavelet
        score_wavelet = forense.get("score_wavelet")

        # Inconsistencia de ruido
        mapa_inconsistencia_b64 = None
        if "mapa_inconsistencia_ruido" in forense and forense["mapa_inconsistencia_ruido"] is not None:
            mapa_inconsistencia_b64 = _numpy_para_base64_png(forense["mapa_inconsistencia_ruido"])

        # Metadados EXIF e indicadores
        raw_metadados = forense.get("metadados")
        if raw_metadados:
            # Garante serializacao JSON (converte valores nao serializaveis para string)
            metadados = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                         for k, v in raw_metadados.items()}
        indicadores_ia = forense.get("indicadores_ia")
    except Exception as e:
        logger.error(f"Erro ao serializar dados forenses: {e}", exc_info=True)

    return ResultadoAnalise(
        score_final=round(resultado["ensemble"].score_final, 4),
        score_calibrado=round(resultado["score_calibrado"], 4),
        classificacao=resultado["ensemble"].classificacao,
        cor=resultado["ensemble"].cor,
        incerteza=round(resultado["ensemble"].incerteza, 4),
        intervalo_confianca=[round(v, 4) for v in resultado["intervalo_confianca"]],
        concordancia=round(resultado["concordancia"], 4),
        scores_individuais=scores_individuais,
        tempo_total_ms=round(resultado["tempo_total_ms"], 1),
        tipo="imagem",
        heatmap_gradcam=heatmap_gradcam,
        espectro_fft=espectro_fft,
        noise_print=noise_print_b64,
        histograma_rgb=histograma_rgb,
        features_frequencia=features_frequencia,
        score_pixels=score_pixels,
        uniformidade=uniformidade,
        score_ela=score_ela,
        score_wavelet=score_wavelet,
        mapa_ela=mapa_ela_b64,
        mapa_inconsistencia_ruido=mapa_inconsistencia_b64,
        metadados=metadados,
        indicadores_ia=indicadores_ia,
    )


def _serializar_resultado_imagem(resultado: dict) -> dict:
    """Converte resultado do pipeline em dict serializavel JSON."""
    scores_individuais = []
    for res in resultado["ensemble"].resultados_detalhados:
        scores_individuais.append({
            "modelo_id": res.id_modelo,
            "modelo_nome": res.nome_modelo,
            "score": round(res.score, 4),
            "tempo_ms": round(res.tempo_inferencia_ms, 1),
        })

    forense = resultado.get("analise_forense", {})
    heatmap_gradcam = None
    espectro_fft = None
    noise_print_b64 = None

    try:
        for _chave, mapa in resultado.get("visualizacoes", {}).items():
            if mapa is not None and isinstance(mapa, np.ndarray) and mapa.any():
                heatmap_gradcam = _numpy_para_base64_png(mapa)
                break
        if "espectro_fft" in forense and forense["espectro_fft"] is not None:
            espectro_fft = _numpy_para_base64_png(forense["espectro_fft"])
        if "noise_print" in forense and forense["noise_print"] is not None:
            noise_print_b64 = _noise_print_para_base64(forense["noise_print"])
    except Exception as e:
        logger.error(f"Erro ao serializar visualizacoes: {e}")

    raw_metadados = forense.get("metadados")
    metadados = None
    if raw_metadados:
        metadados = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                     for k, v in raw_metadados.items()}

    # ELA
    score_ela = forense.get("score_ela")
    mapa_ela_b64 = None
    try:
        if "mapa_ela" in forense and forense["mapa_ela"] is not None:
            mapa_ela_b64 = _numpy_para_base64_png(forense["mapa_ela"])
    except Exception as e:
        logger.error(f"Erro ao serializar mapa ELA: {e}")

    # Wavelet
    score_wavelet = forense.get("score_wavelet")

    # Inconsistencia de ruido
    mapa_inconsistencia_b64 = None
    try:
        if "mapa_inconsistencia_ruido" in forense and forense["mapa_inconsistencia_ruido"] is not None:
            mapa_inconsistencia_b64 = _numpy_para_base64_png(forense["mapa_inconsistencia_ruido"])
    except Exception as e:
        logger.error(f"Erro ao serializar mapa inconsistencia: {e}")

    return {
        "score_final": round(resultado["ensemble"].score_final, 4),
        "score_calibrado": round(resultado["score_calibrado"], 4),
        "classificacao": resultado["ensemble"].classificacao,
        "cor": resultado["ensemble"].cor,
        "incerteza": round(resultado["ensemble"].incerteza, 4),
        "intervalo_confianca": [round(v, 4) for v in resultado["intervalo_confianca"]],
        "concordancia": round(resultado["concordancia"], 4),
        "scores_individuais": scores_individuais,
        "tempo_total_ms": round(resultado["tempo_total_ms"], 1),
        "tipo": "imagem",
        "heatmap_gradcam": heatmap_gradcam,
        "espectro_fft": espectro_fft,
        "noise_print": noise_print_b64,
        "histograma_rgb": forense.get("histograma_rgb"),
        "features_frequencia": forense.get("features_frequencia"),
        "score_pixels": forense.get("score_pixels"),
        "uniformidade": forense.get("uniformidade"),
        "score_ela": score_ela,
        "score_wavelet": score_wavelet,
        "mapa_ela": mapa_ela_b64,
        "mapa_inconsistencia_ruido": mapa_inconsistencia_b64,
        "metadados": metadados,
        "indicadores_ia": forense.get("indicadores_ia"),
    }


@app.post("/api/analisar/imagem/stream")
async def analisar_imagem_stream(
    arquivo: UploadFile = File(...),
    modelos: Optional[str] = Query(None, description="IDs dos modelos separados por virgula"),
):
    """Analisa imagem com streaming de progresso via SSE."""
    extensao = Path(arquivo.filename).suffix.lower()
    if extensao not in CONFIG_APP.extensoes_imagem:
        raise HTTPException(
            status_code=400,
            detail=f"Formato nao suportado: {extensao}. Use: {', '.join(CONFIG_APP.extensoes_imagem)}"
        )

    try:
        conteudo = await arquivo.read()
        imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler imagem: {str(e)}")

    caminho_temp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extensao, delete=False) as tmp:
            tmp.write(conteudo)
            caminho_temp = tmp.name
    except Exception:
        logger.warning("Nao foi possivel salvar arquivo temporario para metadados")

    modelos_habilitados = None
    if modelos:
        modelos_habilitados = [m.strip() for m in modelos.split(",")]

    fila_eventos = queue.Queue()

    def callback_progresso(evento):
        fila_eventos.put(evento)

    def executar():
        try:
            pipeline = obter_pipeline()
            resultado = pipeline.analisar_imagem(
                imagem,
                modelos_habilitados=modelos_habilitados,
                callback_progresso=callback_progresso,
                caminho_arquivo=caminho_temp,
            )
            fila_eventos.put({"_resultado": resultado})
        except Exception as e:
            logger.error(f"Erro na analise streaming: {e}", exc_info=True)
            fila_eventos.put({"_erro": str(e)})
        finally:
            fila_eventos.put(None)  # Sentinel
            if caminho_temp:
                Path(caminho_temp).unlink(missing_ok=True)

    thread = threading.Thread(target=executar, daemon=True)
    thread.start()

    async def gerar_sse():
        while True:
            await asyncio.sleep(0.05)
            try:
                evento = fila_eventos.get_nowait()
            except queue.Empty:
                continue

            if evento is None:
                break

            if "_resultado" in evento:
                dados = _serializar_resultado_imagem(evento["_resultado"])
                yield f"data: {json.dumps({'tipo': 'resultado', 'dados': dados})}\n\n"
            elif "_erro" in evento:
                yield f"data: {json.dumps({'tipo': 'erro', 'mensagem': evento['_erro']})}\n\n"
            else:
                yield f"data: {json.dumps({'tipo': 'progresso', **evento})}\n\n"

    return StreamingResponse(
        gerar_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/analisar/video", response_model=ResultadoVideo)
async def analisar_video(
    arquivo: UploadFile = File(...),
):
    """Analisa um video para detectar deepfakes."""
    extensao = Path(arquivo.filename).suffix.lower()
    if extensao not in CONFIG_APP.extensoes_video:
        raise HTTPException(
            status_code=400,
            detail=f"Formato nao suportado: {extensao}. Use: {', '.join(CONFIG_APP.extensoes_video)}"
        )

    # Salva video temporariamente
    try:
        conteudo = await arquivo.read()
        with tempfile.NamedTemporaryFile(suffix=extensao, delete=False) as tmp:
            tmp.write(conteudo)
            caminho_temp = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler video: {str(e)}")

    try:
        pipeline = obter_pipeline()
        resultado = pipeline.analisar_video(caminho_temp)
    except Exception as e:
        logger.error(f"Erro na analise de video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro na analise: {str(e)}")
    finally:
        Path(caminho_temp).unlink(missing_ok=True)

    scores_individuais = []
    if resultado["ensemble"].resultados_detalhados:
        for res in resultado["ensemble"].resultados_detalhados:
            scores_individuais.append(ScoreIndividual(
                modelo_id=res.id_modelo,
                modelo_nome=res.nome_modelo,
                score=round(res.score, 4),
                tempo_ms=round(res.tempo_inferencia_ms, 1),
            ))

    return ResultadoVideo(
        score_final=round(resultado["ensemble"].score_final, 4),
        score_calibrado=round(resultado.get("score_calibrado", resultado["ensemble"].score_final), 4),
        classificacao=resultado["ensemble"].classificacao,
        cor=resultado["ensemble"].cor,
        incerteza=round(resultado["ensemble"].incerteza, 4),
        intervalo_confianca=[round(v, 4) for v in resultado.get("intervalo_confianca", [0.0, 1.0])],
        concordancia=round(resultado.get("concordancia", 0.0), 4),
        scores_individuais=scores_individuais,
        tempo_total_ms=round(resultado["tempo_total_ms"], 1),
        tipo="video",
        timeline=resultado.get("timeline", []),
        frames_suspeitos=resultado.get("frames_suspeitos", []),
        total_frames=resultado.get("total_frames", 0),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
