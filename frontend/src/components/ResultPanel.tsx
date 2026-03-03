/**
 * ResultPanel — Painel principal de resultados da analise
 * Combina Gauge + ModelScores + tabs (Resultado | Forense | Estatisticas)
 */

import { useState } from 'react'
import { Clock, FileImage, Film, CheckCircle2, AlertTriangle, XCircle, BarChart3, Shield, PieChart } from 'lucide-react'
import clsx from 'clsx'
import type { ResultadoAnalise, ResultadoVideo } from '../api'
import { GaugeChart } from './GaugeChart'
import { ModelScores } from './ModelScores'
import { StatsPanel } from './StatsPanel'
import { VideoTimeline } from './VideoTimeline'
import { ForensicPanel } from './ForensicPanel'
import { ImagePreview } from './ImagePreview'

interface ResultPanelProps {
  resultado: ResultadoAnalise | ResultadoVideo
  nomeArquivo?: string
  arquivoOriginal?: File | null
}

type AbaResultado = 'resultado' | 'forense' | 'estatisticas'

function corDoScore(v: number) {
  if (v < 0.25) return { texto: '#22c55e', fundo: '#14532d22', borda: '#22c55e44' }
  if (v < 0.50) return { texto: '#14b8a6', fundo: '#134e4a22', borda: '#14b8a644' }
  if (v < 0.75) return { texto: '#f59e0b', fundo: '#78350f22', borda: '#f59e0b44' }
  return { texto: '#ef4444', fundo: '#7f1d1d22', borda: '#ef444444' }
}

function IconeScore({ score }: { score: number }) {
  if (score < 0.25) return <CheckCircle2 size={14} className="text-[#22c55e]" />
  if (score < 0.50) return <CheckCircle2 size={14} className="text-[#14b8a6]" />
  if (score < 0.75) return <AlertTriangle size={14} className="text-[#f59e0b]" />
  return <XCircle size={14} className="text-[#ef4444]" />
}

// Verifica se tem dados forenses
function temDadosForenses(r: ResultadoAnalise): boolean {
  return !!(
    r.heatmap_gradcam ||
    r.espectro_fft ||
    r.noise_print ||
    r.histograma_rgb ||
    r.features_frequencia ||
    r.metadados ||
    r.indicadores_ia ||
    r.score_pixels != null ||
    r.score_ela != null ||
    r.score_wavelet != null ||
    r.mapa_ela ||
    r.mapa_inconsistencia_ruido
  )
}

// Skeleton de carregamento
export function ResultPanelSkeleton() {
  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header skeleton */}
      <div className="rounded-lg border border-[#27272a] bg-[#18181b] p-4">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-24 h-4 bg-[#27272a] rounded animate-skeleton" />
          <div className="w-16 h-4 bg-[#27272a] rounded animate-skeleton" />
        </div>
        <div className="flex gap-6 items-center">
          <div className="w-[220px] h-[220px] rounded-full bg-[#27272a] animate-skeleton" />
          <div className="flex-1 space-y-3">
            {[80, 65, 90, 55, 70].map((w, i) => (
              <div key={i} className="h-10 bg-[#27272a] rounded animate-skeleton" style={{ width: `${w}%` }} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export function ResultPanel({ resultado, nomeArquivo, arquivoOriginal }: ResultPanelProps) {
  const { score_calibrado, classificacao, tipo, tempo_total_ms } = resultado
  const cores = corDoScore(score_calibrado)
  const ehVideo = tipo === 'video'
  const resultadoVideo = ehVideo ? (resultado as ResultadoVideo) : null
  const [abaAtiva, setAbaAtiva] = useState<AbaResultado>('resultado')

  const temForense = temDadosForenses(resultado)

  const abas: { id: AbaResultado; label: string; icone: typeof PieChart; habilitada: boolean }[] = [
    { id: 'resultado', label: 'Resultado', icone: PieChart, habilitada: true },
    { id: 'forense', label: 'Forense', icone: Shield, habilitada: temForense },
    { id: 'estatisticas', label: 'Estatisticas', icone: BarChart3, habilitada: true },
  ]

  return (
    <div className="space-y-4 animate-fade-in-up">

      {/* Cabecalho do resultado */}
      <div
        className="rounded-lg border p-4"
        style={{ backgroundColor: cores.fundo, borderColor: cores.borda }}
      >
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <IconeScore score={score_calibrado} />
            <span className="text-sm font-semibold" style={{ color: cores.texto }}>
              {classificacao}
            </span>
          </div>

          <div className="flex items-center gap-4 text-xs text-[#71717a]">
            {/* Tipo do arquivo */}
            <span className="flex items-center gap-1">
              {ehVideo ? <Film size={11} /> : <FileImage size={11} />}
              {ehVideo ? 'Video' : 'Imagem'}
            </span>

            {/* Tempo de analise */}
            <span className="flex items-center gap-1">
              <Clock size={11} />
              <span className="font-mono">
                {tempo_total_ms < 1000
                  ? `${tempo_total_ms.toFixed(0)}ms`
                  : `${(tempo_total_ms / 1000).toFixed(2)}s`}
              </span>
            </span>

            {/* Nome do arquivo */}
            {nomeArquivo && (
              <span className="font-mono text-[10px] truncate max-w-[180px] text-[#52525b]">
                {nomeArquivo}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Preview da imagem original */}
      {arquivoOriginal && !ehVideo && (
        <div className="rounded-lg border border-[#27272a] bg-[#18181b] p-4">
          <ImagePreview
            arquivo={arquivoOriginal}
            classificacao={classificacao}
            cor={cores.texto}
          />
        </div>
      )}

      {/* Conteudo principal: Gauge + Scores lado a lado */}
      <div className="grid grid-cols-1 lg:grid-cols-[240px_1fr] gap-4">

        {/* Gauge */}
        <div className="rounded-lg border border-[#27272a] bg-[#18181b] p-4 flex flex-col items-center justify-center">
          <GaugeChart
            valor={score_calibrado}
            classificacao={classificacao}
            tamanho={220}
          />

          {/* Score bruto vs calibrado */}
          <div className="mt-3 flex gap-4 text-center w-full">
            <div className="flex-1 rounded border border-[#27272a] bg-[#09090b] p-2">
              <p className="text-[9px] text-[#52525b] uppercase tracking-wider mb-0.5">Bruto</p>
              <p className="font-mono text-sm font-bold text-[#a1a1aa]">
                {(resultado.score_final * 100).toFixed(1)}%
              </p>
            </div>
            <div className="flex-1 rounded border border-[#27272a] bg-[#09090b] p-2">
              <p className="text-[9px] text-[#52525b] uppercase tracking-wider mb-0.5">Calibrado</p>
              <p className="font-mono text-sm font-bold" style={{ color: cores.texto }}>
                {(score_calibrado * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        {/* Scores por modelo */}
        <div className="rounded-lg border border-[#27272a] bg-[#18181b] p-4">
          <ModelScores scores={resultado.scores_individuais} />
        </div>
      </div>

      {/* Tabs: Resultado | Forense | Estatisticas */}
      <div className="rounded-lg border border-[#27272a] bg-[#18181b] overflow-hidden">
        {/* Tab bar */}
        <div className="flex border-b border-[#27272a]">
          {abas.map(({ id, label, icone: Icone, habilitada }) => (
            <button
              key={id}
              onClick={() => habilitada && setAbaAtiva(id)}
              disabled={!habilitada}
              className={clsx(
                'flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors relative',
                abaAtiva === id
                  ? 'text-[#fafafa]'
                  : habilitada
                    ? 'text-[#71717a] hover:text-[#a1a1aa]'
                    : 'text-[#3f3f46] cursor-not-allowed'
              )}
            >
              <Icone size={13} />
              {label}
              {id === 'forense' && temForense && (
                <span className="w-1.5 h-1.5 rounded-full bg-[#6366f1]" />
              )}
              {/* Indicador de aba ativa */}
              {abaAtiva === id && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#6366f1]" />
              )}
            </button>
          ))}
        </div>

        {/* Conteudo da aba */}
        <div className="p-4">
          {abaAtiva === 'resultado' && (
            <div className="space-y-4">
              {/* Timeline (apenas para video) */}
              {ehVideo && resultadoVideo && resultadoVideo.timeline.length > 0 && (
                <VideoTimeline resultado={resultadoVideo} />
              )}

              {/* Resumo rapido quando nao e video */}
              {!ehVideo && (
                <div className="space-y-3">
                  <h3 className="text-xs font-semibold uppercase tracking-widest text-[#71717a]">
                    Resumo da Analise
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3 text-center">
                      <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">Score Final</p>
                      <p className="font-mono text-lg font-bold" style={{ color: cores.texto }}>
                        {(score_calibrado * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3 text-center">
                      <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">Concordancia</p>
                      <p className="font-mono text-lg font-bold text-[#a1a1aa]">
                        {Math.round(resultado.concordancia * 100)}%
                      </p>
                    </div>
                    <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3 text-center">
                      <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">Modelos</p>
                      <p className="font-mono text-lg font-bold text-[#a1a1aa]">
                        {resultado.scores_individuais.length}
                      </p>
                    </div>
                    <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3 text-center">
                      <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">Tempo</p>
                      <p className="font-mono text-lg font-bold text-[#a1a1aa]">
                        {tempo_total_ms < 1000
                          ? `${tempo_total_ms.toFixed(0)}ms`
                          : `${(tempo_total_ms / 1000).toFixed(1)}s`}
                      </p>
                    </div>
                  </div>

                  {/* Indicadores rapidos de forense */}
                  {(resultado.score_pixels != null || resultado.score_ela != null || resultado.score_wavelet != null) && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {resultado.score_pixels != null && (
                        <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3">
                          <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">Pixels</p>
                          <p className="font-mono text-sm font-bold" style={{
                            color: resultado.score_pixels > 0.6 ? '#ef4444' : resultado.score_pixels > 0.4 ? '#f59e0b' : '#22c55e'
                          }}>
                            {(resultado.score_pixels * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {resultado.score_ela != null && (
                        <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3">
                          <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">ELA</p>
                          <p className="font-mono text-sm font-bold" style={{
                            color: resultado.score_ela > 0.6 ? '#ef4444' : resultado.score_ela > 0.4 ? '#f59e0b' : '#22c55e'
                          }}>
                            {(resultado.score_ela * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {resultado.score_wavelet != null && (
                        <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3">
                          <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">Wavelet</p>
                          <p className="font-mono text-sm font-bold" style={{
                            color: resultado.score_wavelet > 0.6 ? '#ef4444' : resultado.score_wavelet > 0.4 ? '#f59e0b' : '#22c55e'
                          }}>
                            {(resultado.score_wavelet * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {resultado.indicadores_ia && (
                        <div className="rounded-md border border-[#27272a] bg-[#09090b] p-3">
                          <p className="text-[10px] text-[#52525b] uppercase tracking-wider mb-1">EXIF</p>
                          <p className="font-mono text-sm font-bold" style={{
                            color: resultado.indicadores_ia.pontuacao_suspeita >= 3 ? '#ef4444'
                              : resultado.indicadores_ia.pontuacao_suspeita >= 2 ? '#f59e0b' : '#22c55e'
                          }}>
                            {resultado.indicadores_ia.nivel_suspeita.replace('_', ' ')}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {abaAtiva === 'forense' && temForense && (
            <ForensicPanel resultado={resultado} />
          )}

          {abaAtiva === 'estatisticas' && (
            <StatsPanel resultado={resultado} />
          )}
        </div>
      </div>
    </div>
  )
}
