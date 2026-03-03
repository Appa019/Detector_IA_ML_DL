/**
 * AnalysisProgress — Stepper visual de progresso da analise
 * Mostra cada modelo/etapa em tempo real via SSE
 */

import { useState, useEffect, useRef } from 'react'
import {
  Cpu, CheckCircle2, Loader2, Circle, AlertTriangle,
  Microscope, Waves, FileSearch, Sparkles, Fingerprint,
} from 'lucide-react'
import clsx from 'clsx'
import type { EventoProgresso } from '../api'

export type StatusEtapa = 'pendente' | 'processando' | 'concluido' | 'erro'

export interface EtapaAnalise {
  id: string
  nome: string
  tipo: 'modelo' | 'forense'
  status: StatusEtapa
  score?: number
  tempo_ms?: number
  erro?: string
}

interface AnalysisProgressProps {
  etapas: EtapaAnalise[]
  nomeArquivo?: string
  tempoDecorrido: number
}

function IconeEtapa({ status, tipo }: { status: StatusEtapa; tipo: 'modelo' | 'forense' }) {
  if (status === 'concluido') return <CheckCircle2 size={15} className="text-[#22c55e]" />
  if (status === 'processando') return <Loader2 size={15} className="text-[#818cf8] animate-spin" />
  if (status === 'erro') return <AlertTriangle size={15} className="text-[#f59e0b]" />
  return <Circle size={15} className="text-[#3f3f46]" />
}

function IconeTipoForense({ etapa }: { etapa: string }) {
  if (etapa === 'pixels') return <Microscope size={11} className="text-[#52525b]" />
  if (etapa === 'ela') return <Sparkles size={11} className="text-[#52525b]" />
  if (etapa === 'wavelet') return <Waves size={11} className="text-[#52525b]" />
  if (etapa === 'consistencia_ruido') return <Fingerprint size={11} className="text-[#52525b]" />
  if (etapa === 'espectro') return <Waves size={11} className="text-[#52525b]" />
  if (etapa === 'metadados') return <FileSearch size={11} className="text-[#52525b]" />
  return <Sparkles size={11} className="text-[#52525b]" />
}

function formatarTempo(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function AnalysisProgress({ etapas, nomeArquivo, tempoDecorrido }: AnalysisProgressProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  const etapasModelo = etapas.filter(e => e.tipo === 'modelo')
  const etapasForense = etapas.filter(e => e.tipo === 'forense')

  const totalEtapas = etapas.length
  const concluidas = etapas.filter(e => e.status === 'concluido').length
  const pct = totalEtapas > 0 ? Math.round((concluidas / totalEtapas) * 100) : 0

  const etapaAtual = etapas.find(e => e.status === 'processando')

  // Auto-scroll para etapa ativa
  useEffect(() => {
    if (!containerRef.current) return
    const ativo = containerRef.current.querySelector('[data-ativo="true"]')
    if (ativo) {
      ativo.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [etapaAtual?.id])

  return (
    <div className="rounded-lg border border-[#27272a] bg-[#18181b] overflow-hidden animate-fade-in">

      {/* Header */}
      <div className="px-4 py-3 border-b border-[#27272a] flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="relative w-5 h-5">
            <Cpu size={15} className="text-[#818cf8] animate-pulse" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[#fafafa]">Analisando</p>
            {nomeArquivo && (
              <p className="text-[10px] text-[#52525b] font-mono truncate max-w-[240px]">
                {nomeArquivo}
              </p>
            )}
          </div>
        </div>
        <div className="text-right">
          <p className="font-mono text-xs font-bold text-[#818cf8]">{pct}%</p>
          <p className="text-[10px] text-[#52525b] font-mono">
            {formatarTempo(tempoDecorrido)}
          </p>
        </div>
      </div>

      {/* Barra de progresso */}
      <div className="h-1 bg-[#27272a]">
        <div
          className="h-full bg-gradient-to-r from-[#6366f1] to-[#818cf8] transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Etapas */}
      <div ref={containerRef} className="p-4 space-y-4 max-h-[420px] overflow-y-auto">

        {/* Modelos de deteccao */}
        {etapasModelo.length > 0 && (
          <div>
            <p className="text-[10px] text-[#52525b] uppercase tracking-widest font-semibold mb-2.5">
              Modelos de Deteccao
            </p>
            <div className="space-y-1">
              {etapasModelo.map((etapa) => (
                <div
                  key={etapa.id}
                  data-ativo={etapa.status === 'processando'}
                  className={clsx(
                    'flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-300',
                    etapa.status === 'processando' && 'bg-[#1e1b4b]/40 ring-1 ring-[#4338ca]/30',
                    etapa.status === 'concluido' && 'bg-[#14532d]/10',
                    etapa.status === 'erro' && 'bg-[#78350f]/10',
                    etapa.status === 'pendente' && 'opacity-40',
                  )}
                >
                  <IconeEtapa status={etapa.status} tipo="modelo" />

                  <span className={clsx(
                    'flex-1 text-xs font-medium truncate',
                    etapa.status === 'processando' ? 'text-[#c7d2fe]' :
                    etapa.status === 'concluido' ? 'text-[#a1a1aa]' :
                    etapa.status === 'erro' ? 'text-[#fbbf24]' :
                    'text-[#52525b]'
                  )}>
                    {etapa.nome}
                  </span>

                  {/* Score + tempo (quando concluido) */}
                  {etapa.status === 'concluido' && etapa.score != null && (
                    <div className="flex items-center gap-2.5 flex-shrink-0">
                      <span className={clsx(
                        'font-mono text-[11px] font-bold',
                        etapa.score < 0.25 ? 'text-[#22c55e]' :
                        etapa.score < 0.50 ? 'text-[#14b8a6]' :
                        etapa.score < 0.75 ? 'text-[#f59e0b]' :
                        'text-[#ef4444]'
                      )}>
                        {(etapa.score * 100).toFixed(1)}%
                      </span>
                      {etapa.tempo_ms != null && (
                        <span className="font-mono text-[10px] text-[#3f3f46]">
                          {formatarTempo(etapa.tempo_ms)}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Spinner quando processando */}
                  {etapa.status === 'processando' && (
                    <span className="text-[10px] text-[#818cf8] font-mono animate-pulse flex-shrink-0">
                      processando...
                    </span>
                  )}

                  {/* Erro */}
                  {etapa.status === 'erro' && etapa.erro && (
                    <span className="text-[10px] text-[#fbbf24] truncate max-w-[140px] flex-shrink-0" title={etapa.erro}>
                      erro
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Analise forense */}
        {etapasForense.length > 0 && (
          <div>
            <p className="text-[10px] text-[#52525b] uppercase tracking-widest font-semibold mb-2.5">
              Analise Forense
            </p>
            <div className="space-y-1">
              {etapasForense.map((etapa) => (
                <div
                  key={etapa.id}
                  data-ativo={etapa.status === 'processando'}
                  className={clsx(
                    'flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-300',
                    etapa.status === 'processando' && 'bg-[#1e1b4b]/40 ring-1 ring-[#4338ca]/30',
                    etapa.status === 'concluido' && 'bg-[#14532d]/10',
                    etapa.status === 'pendente' && 'opacity-40',
                  )}
                >
                  <IconeEtapa status={etapa.status} tipo="forense" />

                  <IconeTipoForense etapa={etapa.id} />

                  <span className={clsx(
                    'flex-1 text-xs font-medium',
                    etapa.status === 'processando' ? 'text-[#c7d2fe]' :
                    etapa.status === 'concluido' ? 'text-[#a1a1aa]' :
                    'text-[#52525b]'
                  )}>
                    {etapa.nome}
                  </span>

                  {etapa.status === 'processando' && (
                    <span className="text-[10px] text-[#818cf8] font-mono animate-pulse flex-shrink-0">
                      processando...
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer - etapa atual */}
      {etapaAtual && (
        <div className="px-4 py-2.5 border-t border-[#27272a] bg-[#09090b]">
          <p className="text-[10px] text-[#71717a] truncate">
            <span className="text-[#818cf8]">&#9654;</span>{' '}
            {etapaAtual.nome}
          </p>
        </div>
      )}
    </div>
  )
}


// ============================================================
// Hook para gerenciar estado das etapas
// ============================================================

export function useEtapasAnalise() {
  const [etapas, setEtapas] = useState<EtapaAnalise[]>([])
  const [tempoDecorrido, setTempoDecorrido] = useState(0)
  const inicioRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>()

  const iniciar = () => {
    setEtapas([])
    setTempoDecorrido(0)
    inicioRef.current = Date.now()
    timerRef.current = setInterval(() => {
      setTempoDecorrido(Date.now() - inicioRef.current)
    }, 100)
  }

  const parar = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = undefined
    }
  }

  const processarEvento = (evento: EventoProgresso) => {
    setEtapas(prev => {
      const novas = [...prev]

      if (evento.evento === 'inicio_modelo') {
        // Marca anterior como concluido se ainda processando (fallback)
        const anterior = novas.find(e => e.status === 'processando' && e.tipo === 'modelo')
        if (anterior) anterior.status = 'concluido'

        // Adiciona ou atualiza modelo
        const existente = novas.find(e => e.id === evento.modelo_id)
        if (existente) {
          existente.status = 'processando'
        } else {
          novas.push({
            id: evento.modelo_id!,
            nome: evento.modelo_nome!,
            tipo: 'modelo',
            status: 'processando',
          })
        }
      }

      else if (evento.evento === 'fim_modelo') {
        const etapa = novas.find(e => e.id === evento.modelo_id)
        if (etapa) {
          etapa.status = 'concluido'
          etapa.score = evento.score
          etapa.tempo_ms = evento.tempo_ms
        }
      }

      else if (evento.evento === 'erro_modelo') {
        const etapa = novas.find(e => e.id === evento.modelo_id)
        if (etapa) {
          etapa.status = 'erro'
          etapa.erro = evento.erro
        } else {
          novas.push({
            id: evento.modelo_id!,
            nome: evento.modelo_nome!,
            tipo: 'modelo',
            status: 'erro',
            erro: evento.erro,
          })
        }
      }

      else if (evento.evento === 'inicio_forense') {
        // Marca anterior forense como concluido
        const anterior = novas.find(e => e.status === 'processando' && e.tipo === 'forense')
        if (anterior) anterior.status = 'concluido'

        const nomeForense: Record<string, string> = {
          pixels: 'Estatisticas de Pixels',
          ela: 'Error Level Analysis (ELA)',
          wavelet: 'Analise Wavelet',
          consistencia_ruido: 'Consistencia de Ruido',
          espectro: 'Analise Espectral (FFT/DCT)',
          metadados: 'Metadados EXIF',
        }

        const id = evento.etapa!
        const existente = novas.find(e => e.id === id)
        if (existente) {
          existente.status = 'processando'
        } else {
          novas.push({
            id,
            nome: nomeForense[id] || id,
            tipo: 'forense',
            status: 'processando',
          })
        }
      }

      else if (evento.evento === 'concluido') {
        // Marca tudo que sobrou como concluido
        novas.forEach(e => {
          if (e.status === 'processando') e.status = 'concluido'
        })
      }

      return novas
    })
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  return { etapas, tempoDecorrido, iniciar, parar, processarEvento }
}
