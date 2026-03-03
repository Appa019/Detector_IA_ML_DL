/**
 * Cliente de API — Detector de Conteúdo Gerado por IA
 * Conecta ao backend FastAPI em /api (proxy Vite -> http://localhost:8000)
 */

import axios from 'axios'

const cliente = axios.create({
  baseURL: '/api',
  timeout: 600_000, // 10 minutos para análise com download de modelos
})

// ============================================================
// TIPOS — espelham os modelos Pydantic do backend
// ============================================================

export interface InfoGPU {
  disponivel: boolean
  dispositivo: string
  nome: string
  vram_total_mb: number
  vram_usada_mb: number
  vram_livre_mb: number
}

export interface StatusAPI {
  status: string
  versao: string
  gpu: InfoGPU
  modelos_disponiveis: number
}

export interface InfoModelo {
  id: string
  nome_exibicao: string
  arquitetura: string
  vram_fp16_mb: number
  papel: string
  tipo: string
  habilitado: boolean
}

export interface ScoreIndividual {
  modelo_id: string
  modelo_nome: string
  score: number
  tempo_ms: number
}

export interface HistogramaRGB {
  vermelho: number[]
  verde: number[]
  azul: number[]
}

export interface FeaturesFrequencia {
  media_espectro: number
  desvio_espectro: number
  assimetria_espectro: number
  curtose_espectro: number
  razao_hf_lf: number
  inclinacao_espectral: number
  media_perfil_alta_freq: number
  media_perfil_baixa_freq: number
  energia_lf_dct: number
  energia_total_dct: number
  razao_energia_dct: number
}

export interface IndicadoresIA {
  nivel_suspeita: string
  pontuacao_suspeita: number
  sem_exif: boolean
  software_ia_detectado: boolean
  software_identificado: string | null
  campos_camera_presentes: boolean
  campos_camera_ausentes: string[]
  indicadores: string[]
}

export interface ResultadoAnalise {
  score_final: number
  score_calibrado: number
  classificacao: string
  cor: string
  incerteza: number
  intervalo_confianca: [number, number]
  concordancia: number
  scores_individuais: ScoreIndividual[]
  tempo_total_ms: number
  tipo: 'imagem' | 'video'
  // Visualizacoes forenses (base64 PNG)
  heatmap_gradcam?: string | null
  espectro_fft?: string | null
  noise_print?: string | null
  // Histograma RGB
  histograma_rgb?: HistogramaRGB | null
  // Analise de frequencia
  features_frequencia?: FeaturesFrequencia | null
  // Estatisticas de pixels
  score_pixels?: number | null
  uniformidade?: number | null
  // Novas analises forenses
  score_ela?: number | null
  score_wavelet?: number | null
  mapa_ela?: string | null
  mapa_inconsistencia_ruido?: string | null
  // Metadados EXIF e indicadores
  metadados?: Record<string, unknown> | null
  indicadores_ia?: IndicadoresIA | null
}

export interface FrameTimeline {
  frame: number
  score: number
  tempo_s?: number
}

export interface ResultadoVideo extends ResultadoAnalise {
  timeline: FrameTimeline[]
  frames_suspeitos: number[]
  total_frames: number
}

// ============================================================
// FUNÇÕES DA API
// ============================================================

/**
 * Verifica status da API e informações de hardware.
 */
export async function obterStatus(): Promise<StatusAPI> {
  const { data } = await cliente.get<StatusAPI>('/status')
  return data
}

/**
 * Lista todos os modelos disponíveis no ensemble.
 */
export async function listarModelos(): Promise<InfoModelo[]> {
  const { data } = await cliente.get<InfoModelo[]>('/modelos')
  return data
}

/**
 * Analisa uma imagem para detectar conteúdo gerado por IA.
 * @param arquivo - Arquivo de imagem (JPG, PNG, WebP)
 * @param modelos - IDs dos modelos a usar (opcional — usa todos se omitido)
 * @param onProgresso - Callback de progresso do upload (0-100)
 */
export async function analisarImagem(
  arquivo: File,
  modelos?: string[],
  onProgresso?: (pct: number) => void,
): Promise<ResultadoAnalise> {
  const form = new FormData()
  form.append('arquivo', arquivo)

  const params: Record<string, string> = {}
  if (modelos && modelos.length > 0) {
    params.modelos = modelos.join(',')
  }

  const { data } = await cliente.post<ResultadoAnalise>('/analisar/imagem', form, {
    params,
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (ev) => {
      if (onProgresso && ev.total) {
        onProgresso(Math.round((ev.loaded / ev.total) * 100))
      }
    },
  })

  return data
}

/**
 * Analisa um vídeo para detectar deepfakes frame a frame.
 * @param arquivo - Arquivo de vídeo (MP4, AVI, MOV, MKV)
 * @param onProgresso - Callback de progresso do upload (0-100)
 */
export async function analisarVideo(
  arquivo: File,
  onProgresso?: (pct: number) => void,
): Promise<ResultadoVideo> {
  const form = new FormData()
  form.append('arquivo', arquivo)

  const { data } = await cliente.post<ResultadoVideo>('/analisar/video', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (ev) => {
      if (onProgresso && ev.total) {
        onProgresso(Math.round((ev.loaded / ev.total) * 100))
      }
    },
  })

  return data
}

// ============================================================
// STREAMING — Análise com progresso em tempo real via SSE
// ============================================================

export interface EventoProgresso {
  evento: 'inicio_modelo' | 'fim_modelo' | 'erro_modelo' | 'inicio_forense' | 'concluido'
  modelo_id?: string
  modelo_nome?: string
  score?: number
  tempo_ms?: number
  indice?: number
  total?: number
  etapa?: string
  erro?: string
}

/**
 * Analisa uma imagem com streaming de progresso via SSE.
 * Envia eventos em tempo real conforme cada modelo é processado.
 */
export async function analisarImagemStream(
  arquivo: File,
  modelos?: string[],
  onProgresso?: (evento: EventoProgresso) => void,
): Promise<ResultadoAnalise> {
  const form = new FormData()
  form.append('arquivo', arquivo)

  const params = new URLSearchParams()
  if (modelos && modelos.length > 0) {
    params.set('modelos', modelos.join(','))
  }

  const url = `/api/analisar/imagem/stream${params.toString() ? '?' + params : ''}`

  const response = await fetch(url, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    const texto = await response.text()
    let mensagem = `Erro HTTP ${response.status}`
    try {
      const json = JSON.parse(texto)
      mensagem = json.detail || mensagem
    } catch { /* ignora */ }
    throw new Error(mensagem)
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let resultado: ResultadoAnalise | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const linhas = buffer.split('\n')
    buffer = linhas.pop()!

    for (const linha of linhas) {
      if (!linha.startsWith('data: ')) continue
      try {
        const data = JSON.parse(linha.slice(6))
        if (data.tipo === 'progresso') {
          const { tipo: _, ...evento } = data
          onProgresso?.(evento as EventoProgresso)
        } else if (data.tipo === 'resultado') {
          resultado = data.dados
        } else if (data.tipo === 'erro') {
          throw new Error(data.mensagem)
        }
      } catch (e) {
        if (e instanceof Error && e.message !== 'Unexpected end of JSON input') {
          throw e
        }
      }
    }
  }

  if (!resultado) throw new Error('Análise encerrada sem resultado')
  return resultado
}
