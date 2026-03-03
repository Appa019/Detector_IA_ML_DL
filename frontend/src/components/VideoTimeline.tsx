/**
 * VideoTimeline — Gráfico de linha mostrando score por frame no vídeo
 * Destaca regiões suspeitas e frames com maior score de IA
 */

import { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea,
} from 'recharts'
import { Film, AlertTriangle } from 'lucide-react'
import type { ResultadoVideo } from '../api'

interface VideoTimelineProps {
  resultado: ResultadoVideo
}

function corDoScore(v: number): string {
  if (v < 0.25) return '#22c55e'
  if (v < 0.50) return '#14b8a6'
  if (v < 0.75) return '#f59e0b'
  return '#ef4444'
}

// Tooltip customizado para a linha do tempo
function TooltipTimeline({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const score = payload[0]?.value as number
  return (
    <div className="bg-[#1f1f23] border border-[#3f3f46] rounded px-2.5 py-2 text-xs shadow-xl">
      <p className="text-[#71717a] mb-1">Frame {label}</p>
      <p className="font-mono font-bold" style={{ color: corDoScore(score / 100) }}>
        {score.toFixed(1)}%
      </p>
    </div>
  )
}

export function VideoTimeline({ resultado }: VideoTimelineProps) {
  const { timeline, frames_suspeitos, total_frames } = resultado

  const dados = useMemo(() =>
    timeline.map((item) => ({
      frame: item.frame,
      score: Math.round(item.score * 100),
    })), [timeline])

  if (!dados || dados.length === 0) {
    return (
      <div className="rounded-md border border-[#27272a] bg-[#18181b] p-6 flex items-center justify-center">
        <p className="text-xs text-[#52525b]">Nenhuma timeline disponível</p>
      </div>
    )
  }

  // Identificar regiões suspeitas (score > 75%) para ReferenceArea
  const regioesSuspeitas = useMemo(() => {
    const regioes: Array<{ inicio: number; fim: number }> = []
    let iniciou = false
    let inicio = 0
    dados.forEach((d) => {
      if (d.score >= 75 && !iniciou) {
        iniciou = true
        inicio = d.frame
      } else if (d.score < 75 && iniciou) {
        iniciou = false
        regioes.push({ inicio, fim: d.frame })
      }
    })
    if (iniciou) regioes.push({ inicio, fim: dados[dados.length - 1].frame })
    return regioes
  }, [dados])

  const maiorScore = Math.max(...dados.map((d) => d.score))
  const frameMaisSuspeito = dados.find((d) => d.score === maiorScore)

  return (
    <div className="animate-fade-in-up delay-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-[#71717a]">
          Linha do Tempo
        </h3>
        <div className="flex items-center gap-3 text-[10px] text-[#52525b]">
          <span className="flex items-center gap-1">
            <Film size={10} />
            {total_frames} frames
          </span>
          {frames_suspeitos.length > 0 && (
            <span className="flex items-center gap-1 text-[#ef4444]">
              <AlertTriangle size={10} />
              {frames_suspeitos.length} suspeitos
            </span>
          )}
        </div>
      </div>

      {/* Gráfico de linha */}
      <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={dados} margin={{ top: 8, right: 8, bottom: 0, left: -24 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />

            {/* Linhas de threshold */}
            <ReferenceLine y={75} stroke="#ef444433" strokeDasharray="4 4" label={false} />
            <ReferenceLine y={50} stroke="#f59e0b22" strokeDasharray="4 4" label={false} />
            <ReferenceLine y={25} stroke="#14b8a622" strokeDasharray="4 4" label={false} />

            {/* Regiões altamente suspeitas destacadas */}
            {regioesSuspeitas.map((r, i) => (
              <ReferenceArea
                key={i}
                x1={r.inicio}
                x2={r.fim}
                fill="#ef444408"
                stroke="#ef444420"
              />
            ))}

            <XAxis
              dataKey="frame"
              tick={{ fill: '#52525b', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={{ stroke: '#27272a' }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: '#52525b', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={false}
              tickLine={false}
              tickCount={5}
            />
            <Tooltip content={<TooltipTimeline />} cursor={{ stroke: '#3f3f46', strokeWidth: 1 }} />

            <Line
              type="monotone"
              dataKey="score"
              stroke="#6366f1"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, fill: '#6366f1', stroke: '#09090b', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Legenda das faixas */}
        <div className="flex items-center gap-4 mt-3 pt-3 border-t border-[#27272a]">
          {[
            { cor: '#22c55e', label: 'Real (0–25%)' },
            { cor: '#14b8a6', label: 'Poss. Real (25–50%)' },
            { cor: '#f59e0b', label: 'Poss. IA (50–75%)' },
            { cor: '#ef4444', label: 'IA (75–100%)' },
          ].map(({ cor, label }) => (
            <div key={label} className="flex items-center gap-1">
              <span className="w-2 h-0.5 rounded" style={{ backgroundColor: cor }} />
              <span className="text-[10px] text-[#52525b]">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Frame mais suspeito */}
      {frameMaisSuspeito && (
        <div className="flex items-center gap-2 rounded-md border border-[#3f3f46] bg-[#18181b] px-3 py-2">
          <AlertTriangle size={12} className="text-[#f59e0b] flex-shrink-0" />
          <span className="text-xs text-[#a1a1aa]">
            Frame mais suspeito:
          </span>
          <span className="font-mono text-xs font-bold text-[#fafafa]">
            #{frameMaisSuspeito.frame}
          </span>
          <span
            className="font-mono text-xs font-bold ml-auto"
            style={{ color: corDoScore(frameMaisSuspeito.score / 100) }}
          >
            {frameMaisSuspeito.score}%
          </span>
        </div>
      )}
    </div>
  )
}
