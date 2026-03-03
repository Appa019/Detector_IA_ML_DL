/**
 * App — Shell principal do Detector de Conteúdo Gerado por IA
 * Layout: Navbar fixa + Sidebar + Área de conteúdo principal
 */

import { useState, useEffect, useCallback } from 'react'
import { ScanSearch, Wifi, WifiOff, RefreshCw, AlertCircle } from 'lucide-react'
import type { StatusAPI, InfoModelo, ResultadoAnalise, ResultadoVideo } from './api'
import { obterStatus, listarModelos, analisarImagemStream, analisarVideo } from './api'
import { Sidebar } from './components/Sidebar'
import { UploadZone, type TipoArquivo } from './components/UploadZone'
import { ResultPanel } from './components/ResultPanel'
import { ResultCarousel } from './components/ResultCarousel'
import { AnalysisProgress, useEtapasAnalise } from './components/AnalysisProgress'

type EstadoAPI = 'verificando' | 'online' | 'offline'

interface ItemResultado {
  resultado: ResultadoAnalise | ResultadoVideo
  nomeArquivo: string
  arquivoOriginal: File | null
}

export default function App() {
  /* ─── Estado da API ─── */
  const [estadoAPI, setEstadoAPI] = useState<EstadoAPI>('verificando')
  const [statusAPI, setStatusAPI] = useState<StatusAPI | null>(null)
  const [modelos, setModelos] = useState<InfoModelo[]>([])
  const [modelosAtivos, setModelosAtivos] = useState<Set<string>>(new Set())
  const [carregandoConfig, setCarregandoConfig] = useState(true)

  /* ─── Estado da análise ─── */
  const [analisando, setAnalisando] = useState(false)
  const [resultados, setResultados] = useState<ItemResultado[]>([])
  const [indiceAtivo, setIndiceAtivo] = useState(0)
  const [erroAnalise, setErroAnalise] = useState<string | null>(null)
  const [nomeArquivo, setNomeArquivo] = useState<string | undefined>()
  const [progressoLote, setProgressoLote] = useState<{ atual: number; total: number } | null>(null)
  const etapasAnalise = useEtapasAnalise()

  /* ─── Inicialização ─── */
  const inicializar = useCallback(async () => {
    setCarregandoConfig(true)
    setEstadoAPI('verificando')

    try {
      const [status, lista] = await Promise.all([obterStatus(), listarModelos()])
      setStatusAPI(status)
      setEstadoAPI('online')
      setModelos(lista)

      const ativos = new Set(lista.filter((m) => m.habilitado).map((m) => m.id))
      setModelosAtivos(ativos)
    } catch {
      setEstadoAPI('offline')
    } finally {
      setCarregandoConfig(false)
    }
  }, [])

  useEffect(() => {
    inicializar()
  }, [inicializar])

  /* ─── Toggle de modelo ─── */
  const handleToggleModelo = useCallback((id: string) => {
    setModelosAtivos((prev) => {
      const novo = new Set(prev)
      if (novo.has(id)) {
        if (novo.size <= 1) return prev
        novo.delete(id)
      } else {
        novo.add(id)
      }
      return novo
    })
  }, [])

  /* ─── Análise em lote ─── */
  const handleAnalisar = useCallback(async (arquivos: File[], tipos: TipoArquivo[]) => {
    if (arquivos.length === 0) return

    setAnalisando(true)
    setResultados([])
    setIndiceAtivo(0)
    setErroAnalise(null)
    setProgressoLote({ atual: 0, total: arquivos.length })

    for (let i = 0; i < arquivos.length; i++) {
      const arquivo = arquivos[i]
      const tipo = tipos[i]
      if (!tipo) continue

      setProgressoLote({ atual: i + 1, total: arquivos.length })
      setNomeArquivo(arquivo.name)
      etapasAnalise.iniciar()

      try {
        let resultado: ResultadoAnalise | ResultadoVideo

        if (tipo === 'imagem') {
          const ids = Array.from(modelosAtivos)
          resultado = await analisarImagemStream(arquivo, ids, etapasAnalise.processarEvento)
        } else {
          resultado = await analisarVideo(arquivo)
        }

        const item: ItemResultado = {
          resultado,
          nomeArquivo: arquivo.name,
          arquivoOriginal: tipo === 'imagem' ? arquivo : null,
        }

        setResultados((prev) => {
          const novos = [...prev, item]
          // Auto-navega para o resultado mais recente
          setIndiceAtivo(novos.length - 1)
          return novos
        })
      } catch (err: any) {
        const msg = err?.response?.data?.detail || err?.message || 'Erro desconhecido na análise.'
        setErroAnalise(`Erro em "${arquivo.name}": ${msg}`)
        // Continua com os próximos arquivos mesmo se um falhar
      } finally {
        etapasAnalise.parar()
      }
    }

    setAnalisando(false)
    setProgressoLote(null)
  }, [modelosAtivos, etapasAnalise])

  /* ─── Render ─── */
  const temResultados = resultados.length > 0

  return (
    <div className="flex flex-col h-screen bg-[#09090b] text-[#fafafa] overflow-hidden">

      {/* ── Navbar ── */}
      <header className="h-12 flex-shrink-0 flex items-center justify-between px-4 border-b border-[#27272a] bg-[#09090b] z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded bg-[#1e1b4b] border border-[#4338ca] flex items-center justify-center">
            <ScanSearch size={13} className="text-[#818cf8]" />
          </div>
          <span className="text-sm font-semibold text-[#fafafa] tracking-tight">
            Detector de IA
          </span>
          <span className="text-[10px] text-[#3f3f46] font-mono">v1.0</span>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-2">
          {estadoAPI === 'verificando' && (
            <div className="flex items-center gap-1.5 text-[11px] text-[#71717a]">
              <RefreshCw size={11} className="animate-spin-smooth" />
              Conectando...
            </div>
          )}
          {estadoAPI === 'online' && (
            <div className="flex items-center gap-1.5 text-[11px] text-[#a1a1aa]">
              <span className="relative flex w-1.5 h-1.5">
                <span className="absolute inline-flex h-full w-full rounded-full bg-[#22c55e] opacity-75 animate-ping" />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#22c55e]" />
              </span>
              <Wifi size={11} className="text-[#22c55e]" />
              <span>Online</span>
              {statusAPI && (
                <span className="text-[#52525b]">
                  &mdash; {statusAPI.modelos_disponiveis} modelos
                </span>
              )}
            </div>
          )}
          {estadoAPI === 'offline' && (
            <div className="flex items-center gap-1.5 text-[11px] text-[#ef4444]">
              <WifiOff size={11} />
              Offline
              <button
                onClick={inicializar}
                className="ml-1 text-[#71717a] hover:text-[#a1a1aa] transition-colors"
                aria-label="Reconectar"
              >
                <RefreshCw size={11} />
              </button>
            </div>
          )}
        </div>
      </header>

      {/* ── Corpo principal ── */}
      <div className="flex flex-1 min-h-0">

        {/* Sidebar (oculta em mobile) */}
        <div className="hidden md:block">
          <Sidebar
            gpu={statusAPI?.gpu ?? null}
            modelos={modelos}
            modelosAtivos={modelosAtivos}
            onToggleModelo={handleToggleModelo}
            carregando={carregandoConfig}
          />
        </div>

        {/* Área de conteúdo */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">

            {/* Alerta de API offline */}
            {estadoAPI === 'offline' && (
              <div className="flex items-start gap-3 p-4 rounded-lg border border-[#7f1d1d] bg-[#7f1d1d22] text-[#ef4444] animate-fade-in">
                <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold">API indisponível</p>
                  <p className="text-xs mt-1 text-[#fca5a5]">
                    Não foi possível conectar ao servidor em{' '}
                    <span className="font-mono">localhost:8000</span>.
                    Verifique se o backend está em execução.
                  </p>
                </div>
                <button
                  onClick={inicializar}
                  className="ml-auto flex-shrink-0 flex items-center gap-1.5 text-xs text-[#ef4444] hover:text-[#fca5a5] transition-colors"
                >
                  <RefreshCw size={12} />
                  Tentar novamente
                </button>
              </div>
            )}

            {/* Seção de upload */}
            <section>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h1 className="text-base font-semibold text-[#fafafa]">Análise de Conteúdo</h1>
                  <p className="text-xs text-[#71717a] mt-0.5">
                    Detecta imagens e vídeos gerados ou manipulados por IA
                  </p>
                </div>

                {/* Modelos ativos (pill) */}
                {modelosAtivos.size > 0 && (
                  <div className="hidden sm:flex items-center gap-1.5 text-[10px] text-[#52525b]">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#6366f1]" />
                    {modelosAtivos.size} modelo{modelosAtivos.size !== 1 ? 's' : ''} ativo{modelosAtivos.size !== 1 ? 's' : ''}
                  </div>
                )}
              </div>

              <UploadZone
                onAnalisar={handleAnalisar}
                carregando={analisando}
              />
            </section>

            {/* Divisor */}
            {(analisando || temResultados || erroAnalise) && (
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-[#27272a]" />
                <span className="text-[10px] text-[#3f3f46] uppercase tracking-widest">
                  {analisando && progressoLote
                    ? `Resultado — Imagem ${progressoLote.atual} de ${progressoLote.total}`
                    : 'Resultado'
                  }
                </span>
                <div className="flex-1 h-px bg-[#27272a]" />
              </div>
            )}

            {/* Progresso detalhado durante análise */}
            {analisando && (
              <AnalysisProgress
                etapas={etapasAnalise.etapas}
                nomeArquivo={nomeArquivo}
                tempoDecorrido={etapasAnalise.tempoDecorrido}
              />
            )}

            {/* Erro de análise */}
            {erroAnalise && !analisando && (
              <div className="flex items-start gap-3 p-4 rounded-lg border border-[#7f1d1d] bg-[#7f1d1d22] text-[#ef4444] animate-fade-in">
                <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold">Erro na análise</p>
                  <p className="text-xs mt-1 text-[#fca5a5]">{erroAnalise}</p>
                </div>
              </div>
            )}

            {/* Resultados parciais já prontos (visíveis durante análise) */}
            {analisando && resultados.length > 0 && (
              <div className="opacity-80">
                {resultados.length === 1 ? (
                  <ResultPanel
                    resultado={resultados[0].resultado}
                    nomeArquivo={resultados[0].nomeArquivo}
                    arquivoOriginal={resultados[0].arquivoOriginal}
                  />
                ) : (
                  <ResultCarousel
                    resultados={resultados}
                    indiceAtivo={indiceAtivo}
                    onMudarIndice={setIndiceAtivo}
                  />
                )}
              </div>
            )}

            {/* Resultados finais */}
            {!analisando && resultados.length === 1 && (
              <ResultPanel
                resultado={resultados[0].resultado}
                nomeArquivo={resultados[0].nomeArquivo}
                arquivoOriginal={resultados[0].arquivoOriginal}
              />
            )}

            {!analisando && resultados.length > 1 && (
              <ResultCarousel
                resultados={resultados}
                indiceAtivo={indiceAtivo}
                onMudarIndice={setIndiceAtivo}
              />
            )}

            {/* Estado vazio */}
            {!analisando && resultados.length === 0 && !erroAnalise && estadoAPI === 'online' && (
              <div className="text-center py-16">
                <div className="w-12 h-12 rounded-lg border border-[#27272a] bg-[#18181b] flex items-center justify-center mx-auto mb-4">
                  <ScanSearch size={20} className="text-[#3f3f46]" />
                </div>
                <p className="text-sm text-[#52525b]">Nenhuma análise realizada</p>
                <p className="text-xs text-[#3f3f46] mt-1">
                  Envie uma imagem ou vídeo para começar
                </p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
