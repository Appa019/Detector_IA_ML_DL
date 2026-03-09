# Detector de Conteudo Gerado por IA
**Link da Landing:https://detector-ia-landing.vercel.app/**

Sistema de deteccao de imagens e videos gerados por inteligencia artificial, utilizando um ensemble de modelos de Deep Learning e Machine Learning. Fornece um score de confianca (0-100%) com analise forense detalhada e visualizacoes interativas.

## Download e Inicio Rapido

```bash
# 1. Clone o repositorio
git clone https://github.com/Appa019/Detector_IA_ML_DL.git
cd Detector_IA_ML_DL

# 2. Crie o ambiente virtual e instale as dependencias Python
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

# 3. Baixe os modelos do HuggingFace (~3-5 GB no total)
python utils/download_models.py

# 4. Instale as dependencias do frontend
cd frontend && npm install && cd ..

# 5. Inicie o backend e o frontend (em terminais separados)
uvicorn api:app --host 0.0.0.0 --port 8000
# em outro terminal:
cd frontend && npm run dev
```

Acesse `http://localhost:5173` no navegador. A API estara em `http://localhost:8000/docs`.

## Funcionalidades

- **Deteccao de imagens** geradas por IA (Midjourney, Stable Diffusion, DALL-E, etc.)
- **Deteccao de deepfakes** em videos com timeline frame-a-frame
- **Ensemble de modelos** com agregacao inteligente (meta-learner ou media ponderada)
- **Analise forense completa**: GradCAM, FFT/DCT, ELA, wavelets, noise print, SRM kernels
- **Calibracao de confianca** via Isotonic Regression ou Temperature Scaling
- **Extracao de metadados EXIF** com indicadores automaticos de geracao por IA
- **Interface moderna**: frontend React + API REST (FastAPI)
- **Otimizado para GPU**: modelos em FP16 com carregamento sequencial para economizar VRAM

## Arquitetura

```
┌──────────────────────────────────────────────────┐
│                   Interface                       │
│              React + FastAPI                      │
├──────────────────────────────────────────────────┤
│               Pipeline (core/)                    │
│  Orquestra deteccao sequencial + analise forense  │
├──────────────────────────────────────────────────┤
│            Ensemble (core/ensemble.py)            │
│    Meta-learner (GBM)  |  Media ponderada         │
├──────────────────────────────────────────────────┤
│              Modelos (models/)                    │
│  ViT  |  SDXL Det  |  AI Image  |  SigLIP  |    │
│  FFT/DCT  |  EfficientNet-B4 (video)             │
├──────────────────────────────────────────────────┤
│          Analise Forense (analysis/)              │
│  ELA | FFT | GradCAM | Wavelets | Pixels | EXIF  │
├──────────────────────────────────────────────────┤
│           GPU Manager (utils/)                    │
│     VRAM tracking  |  FP16  |  Cache cleanup      │
└──────────────────────────────────────────────────┘
```

## Modelos do Ensemble

| Modelo | Arquitetura | VRAM (FP16) | Papel |
|--------|------------|-------------|-------|
| Deep Fake Detector v2 | ViT-base-patch16-224 | ~250 MB | Analise espacial de texturas e artefatos |
| SDXL Detector (Organika) | Fine-tuned SDXL classifier | ~1.7 GB | Especializado em modelos de difusao |
| AI Image Detector | ViT (Vision Transformer) | ~280 MB | Treinado em Midjourney/SD/DALL-E |
| SigLIP AI vs Human | SiglipForImageClassification | ~180 MB | Detector binario IA vs Humano |
| Analisador de Frequencia | FFT/DCT + MLP | ~50 MB | Artefatos de frequencia de GANs/difusao |
| EfficientNet-B4 | EfficientNet-B4 | ~150 MB | Deepfakes em video (face swap) |

Modelos adicionais disponiveis (desabilitados por padrao): CLIP ViT-L/14, DINOv2-Small, SigLIP2 3-Classes.

## Analises Forenses

- **GradCAM**: Mapas de calor mostrando regioes que influenciaram a decisao
- **Espectro FFT/DCT**: Detecta padroes de frequencia tipicos de geradores
- **ELA (Error Level Analysis)**: Identifica inconsistencias de compressao JPEG
- **Wavelets**: Analise multi-escala de texturas e ruido
- **Noise Print**: Visualiza padroes de ruido da imagem
- **SRM Kernels**: Filtros de Steganalysis Rich Model para deteccao de manipulacao
- **Estatisticas de Pixels**: Uniformidade, histograma RGB, inconsistencias locais
- **Metadados EXIF**: Extrai e analisa metadados com indicadores automaticos de IA

## Requisitos

### Hardware
- **GPU**: NVIDIA com suporte CUDA (testado em RTX 2070 SUPER 8GB)
- **RAM**: 16 GB recomendado
- **Disco**: ~5 GB para cache dos modelos

### Software
- Python 3.10+
- CUDA 11.8+ e cuDNN
- Node.js 18+ (para o frontend React)

## Instalacao

### 1. Clone o repositorio

```bash
git clone https://github.com/Appa019/Detector_IA_ML_DL.git
cd Detector_IA_ML_DL
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as dependencias Python

```bash
pip install -r requirements.txt
```

### 4. Baixe os modelos

```bash
python utils/download_models.py
```

Os modelos serao baixados do HuggingFace Hub e salvos em `models_cache/`.

### 5. (Opcional) Configure o frontend React

```bash
cd frontend
npm install
npm run build
cd ..
```

## Uso

### API REST (FastAPI)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Documentacao interativa disponivel em `http://localhost:8000/docs`.

### Frontend React + API

Em um terminal, inicie a API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Em outro terminal, inicie o frontend:
```bash
cd frontend
npm run dev
```

## Endpoints da API

| Metodo | Rota | Descricao |
|--------|------|-----------|
| `GET` | `/api/status` | Status da API, GPU e modelos |
| `GET` | `/api/modelos` | Lista modelos do ensemble |
| `POST` | `/api/analisar/imagem` | Analisa uma imagem (multipart) |
| `POST` | `/api/analisar/imagem/stream` | Analise com progresso via SSE |
| `POST` | `/api/analisar/video` | Analisa um video (multipart) |

### Exemplo com cURL

```bash
curl -X POST http://localhost:8000/api/analisar/imagem \
  -F "arquivo=@foto.jpg"
```

## Estrutura do Projeto

```
├── api.py                    # API REST (FastAPI)
├── requirements.txt          # Dependencias Python
├── config/
│   ├── settings.py           # Configuracoes globais
│   └── model_registry.py     # Registro de modelos do ensemble
├── core/
│   ├── pipeline.py           # Pipeline principal de deteccao
│   ├── ensemble.py           # Agregacao de scores (meta-learner + fallback)
│   └── confidence.py         # Calibracao e intervalos de confianca
├── models/
│   ├── base.py               # Classe abstrata DetectorBase
│   ├── spatial_vit.py        # ViT Deep Fake Detector v2
│   ├── sdxl_detector.py      # SDXL Detector (Organika)
│   ├── ai_image_detector.py  # AI Image Detector (ViT)
│   ├── siglip_detector.py    # SigLIP AI vs Human
│   ├── frequency_analyzer.py # Analisador FFT/DCT
│   ├── efficientnet_detector.py # EfficientNet-B4 (video)
│   ├── clip_detector.py      # CLIP UniversalFakeDetect
│   ├── dinov2_detector.py    # DINOv2-Small
│   └── siglip2_detector.py   # SigLIP2 3-Classes
├── analysis/
│   ├── ela.py                # Error Level Analysis
│   ├── frequency.py          # Analise espectral FFT/DCT
│   ├── gradcam.py            # GradCAM para mapas de calor
│   ├── metadata.py           # Extracao EXIF e indicadores de IA
│   ├── pixel_stats.py        # Estatisticas de pixels e noise print
│   ├── srm_kernels.py        # Steganalysis Rich Model kernels
│   └── wavelet.py            # Analise wavelet multi-escala
├── processing/
│   ├── image_processor.py    # Preprocessamento de imagens
│   ├── video_processor.py    # Extracao de frames de video
│   └── face_detector.py      # Deteccao facial (MTCNN)
├── visualization/
│   ├── charts.py             # Graficos Plotly interativos
│   ├── heatmaps.py           # Heatmaps Matplotlib
│   └── video_timeline.py     # Timeline de analise de video
├── utils/
│   ├── gpu_manager.py        # Gerenciamento de VRAM
│   ├── download_models.py    # Download de modelos do HuggingFace
│   └── treinar_calibracao.py # Script de treino do calibrador
├── frontend/                 # Frontend React + TypeScript
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts            # Cliente da API
│   │   └── components/       # Componentes React
│   ├── package.json
│   └── vite.config.ts
└── tests/
    ├── test_ensemble.py
    ├── test_frequency.py
    ├── test_models.py
    └── test_pipeline.py
```

## Calibracao e Fine-tuning

O sistema suporta calibracao com dados proprios para melhorar a precisao:

```bash
python utils/treinar_calibracao.py
```

Isso treina:
- **IsotonicRegression** para calibracao de scores
- **Meta-learner (GradientBoosting)** para stacking do ensemble

Os artefatos sao salvos em `models_cache/calibracao/` e `models_cache/ensemble/`.

## Testes

```bash
pytest tests/
```

## Tecnologias

- **Backend**: Python, PyTorch, HuggingFace Transformers, FastAPI
- **Frontend**: React, TypeScript, Tailwind CSS, Recharts, Vite
- **Visualizacao**: Plotly, Matplotlib
- **Video**: OpenCV, facenet-pytorch (MTCNN)
- **ML**: scikit-learn (calibracao e meta-learner)
- **Analise de sinais**: NumPy, SciPy, PyWavelets

## Licenca

Este projeto e de uso academico e educacional.
