# Detector de Conteudo Gerado por IA

## Sobre o Projeto
App de deteccao de imagens e videos gerados por IA usando ensemble de modelos DL/ML.
Fornece score de confianca (0-100%) com visualizacao estatistica detalhada.

## Hardware Alvo
- GPU: NVIDIA RTX 2070 SUPER (8GB VRAM)
- Todos os modelos rodam em FP16 (half precision)
- Modelos carregados sequencialmente com torch.cuda.empty_cache() entre eles

## Idioma
- Codigo, comentarios, docstrings e UI: Portugues brasileiro
- Nomes de variaveis e funcoes: snake_case em portugues (ex: calcular_score, processar_imagem)
- Nomes de classes: PascalCase em portugues (ex: DetectorEspacial, AnalisadorFrequencia)

## Arquitetura
- Backend: FastAPI (api.py na raiz) + Python 3.12 com .venv
- Frontend principal: React + Vite (frontend/) — app de analise
- Landing page: React + Vite (landing/) — pagina de apresentacao
- DL: PyTorch + HuggingFace Transformers
- Graficos: Recharts (frontend) + Matplotlib (heatmaps backend)
- Video: OpenCV + facenet-pytorch (MTCNN)

## Comandos
```bash
# Backend (porta 8001 — 8000 ocupada por outro projeto no host)
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8001

# Frontend principal
cd frontend && npm run dev        # http://localhost:5176 (proxy -> 8001)

# Landing page
cd landing && npm run dev

# Baixar modelos
.venv/bin/python utils/download_models.py

# Testes
.venv/bin/pytest tests/
```

## Modelos do Ensemble (config/model_registry.py)
| ID | Nome | VRAM FP16 | Tipo | Habilitado |
|----|------|-----------|------|------------|
| spatial_vit | ViT Espacial (Deep Fake Detector v2) | 250MB | imagem | sim |
| sdxl_detector | SDXL Detector (Organika) | 1740MB | imagem | sim |
| ai_image_detector | AI Image Detector (ViT) | 280MB | imagem | sim |
| siglip_detector | SigLIP AI vs Human | 180MB | imagem | sim |
| frequency_analyzer | Analisador FFT/DCT | 50MB | imagem | sim |
| efficientnet_video | EfficientNet-B4 Video | 150MB | video | sim |
| clip_detector | CLIP UniversalFakeDetect | 1700MB | imagem | nao |
| dinov2_detector | DINOv2-Small | 350MB | imagem | nao |
| siglip2_detector | SigLIP2 3-Classes | 180MB | imagem | nao |

## Convencoes
- Cada modelo implementa a classe abstrata `DetectorBase` (models/base.py)
- Pipeline orquestrada por core/pipeline.py — instancia global `pipeline`
- Score final calculado por core/ensemble.py com weighted average ou meta-learner
- VRAM gerenciada por utils/gpu_manager.py — instancia global `gerenciador_gpu`
- Preparado para fine-tuning futuro: DetectorBase tem metodo treinar()

## Analises Forenses (analysis/)
- frequency.py — FFT/DCT espectral
- ela.py — Error Level Analysis
- wavelet.py — decomposicao wavelet (PyWavelets)
- pixel_stats.py — histograma RGB, noise print, mapa de inconsistencia
- gradcam.py — GradCAM para heatmaps
- metadata.py — EXIF e indicadores de software IA
- srm_kernels.py — kernels SRM para deteccao de manipulacao

## Estrutura de Pastas
- api.py — entry point FastAPI (rotas /api/*)
- config/ — configuracoes e registro de modelos
- core/ — pipeline.py, ensemble.py, confidence.py
- models/ — implementacao dos detectores (um arquivo por modelo)
- analysis/ — modulos de analise forense
- processing/ — preprocessamento de imagens e videos
- visualization/ — heatmaps, video_timeline
- utils/ — gpu_manager, download_models
- frontend/ — app React principal (Vite + Tailwind + Recharts)
- landing/ — landing page React (Vite + Tailwind)
- tests/ — testes unitarios e integracao

## Gotchas
- Porta 8000 ocupada por outro projeto no host — backend roda em 8001
- frontend/vite.config.ts tem proxy `/api` apontando para localhost:8001
- Modelos sao carregados e descarregados sequencialmente (nao em paralelo) para economizar VRAM
- Usar sempre `.venv/bin/python` e `.venv/bin/uvicorn` — nao o python do sistema
- Meta-learner do ensemble so ativa se `models_cache/ensemble/meta_learner.pkl` existir; caso contrario usa media ponderada como fallback
