/**
 * StatsPanel — Painel de estatisticas detalhadas da analise
 * Concordancia, intervalo de confianca, radar de modelos, tabela detalhada
 */

import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import { CheckCircle2, AlertTriangle, TrendingUp, Cpu, Clock } from 'lucide-react'
import type { ResultadoAnalise } from '../api'

interface StatsPanelProps {
  resultado: ResultadoAnalise
}

function corDoScore(v: number): string {
  if (v < 0.25) return '#22c55e'
  if (v < 0.50) return '#14b8a6'
  if (v < 0.75) return '#f59e0b'
  return '#ef4444'
}

// Tooltip customizado para o grafico Recharts
function TooltipCustomizado({ active, payload }: any) {
  if (!active || !payload?.length) return null
  const { modelo, score } = payload[0].payload
  return (
    <div className="bg-[#1f1f23] border border-[#3f3f46] rounded px-2 py-1 text-xs shadow-xl">
      <p className="text-[#a1a1aa] mb-0.5 truncate max-w-[160px]">{modelo}</p>
      <p className="font-mono font-bold" style={{ color: corDoScore(score / 100) }}>
        {score.toFixed(1)}%
      </p>
    </div>
  )
}

function TooltipRadar({ active, payload }: any) {
  if (!active || !payload?.length) return null
  const item = payload[0]
  return (
    <div className="bg-[#1f1f23] border border-[#3f3f46] rounded px-2 py-1 text-xs shadow-xl">
      <p className="text-[#a1a1aa] mb-0.5">{item.payload.modelo}</p>
      <p className="font-mono font-bold text-[#818cf8]">
        {item.value.toFixed(1)}%
      </p>
    </div>
  )
}

export function StatsPanel({ resultado }: StatsPanelProps) {
  const { concordancia, incerteza, intervalo_confianca, scores_individuais } = resultado

  const concordanciaPct = Math.round(concordancia * 100)
  const incertezaPct = Math.round(incerteza * 100)
  const ci = intervalo_confianca

  // Dados para o grafico de distribuicao
  const dadosGrafico = useMemo(() =>
    scores_individuais.map((s) => ({
      modelo: s.modelo_nome,
      score: Math.round(s.score * 100),
    })), [scores_individuais])

  // Dados para o radar de concordancia
  const dadosRadar = useMemo(() =>
    scores_individuais.map((s) => ({
      modelo: s.modelo_nome.length > 12 ? s.modelo_nome.slice(0, 12) + '...' : s.modelo_nome,
      modeloCompleto: s.modelo_nome,
      score: Math.round(s.score * 100),
    })), [scores_individuais])

  // Icone e cor da concordancia
  const corConcordancia = concordanciaPct >= 75 ? '#22c55e'
    : concordanciaPct >= 50 ? '#f59e0b' : '#ef4444'
  const IconeConcordancia = concordanciaPct >= 75 ? CheckCircle2 : AlertTriangle

  return (
    <div className="animate-fade-in-up delay-3 space-y-4">
      <h3 className="text-xs font-semibold uppercase tracking-widest text-[#71717a]">
        Estatisticas
      </h3>

      {/* Grid de metricas */}
      <div className="grid grid-cols-3 gap-2">
        {/* Concordancia */}
        <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
          <div className="flex items-center gap-1.5 mb-1">
            <IconeConcordancia size={12} style={{ color: corConcordancia }} />
            <span className="text-[10px] text-[#71717a] uppercase tracking-wider">Concordancia</span>
          </div>
          <p className="font-mono text-xl font-bold" style={{ color: corConcordancia }}>
            {concordanciaPct}%
          </p>
          <p className="text-[10px] text-[#52525b] mt-0.5">entre modelos</p>
        </div>

        {/* Incerteza */}
        <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
          <div className="flex items-center gap-1.5 mb-1">
            <TrendingUp size={12} className="text-[#6366f1]" />
            <span className="text-[10px] text-[#71717a] uppercase tracking-wider">Incerteza</span>
          </div>
          <p className="font-mono text-xl font-bold text-[#a1a1aa]">
            +/-{incertezaPct}%
          </p>
          <p className="text-[10px] text-[#52525b] mt-0.5">desvio padrao</p>
        </div>

        {/* Tempo */}
        <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
          <div className="flex items-center gap-1.5 mb-1">
            <span className="w-2 h-2 rounded-full bg-[#6366f1] flex-shrink-0" />
            <span className="text-[10px] text-[#71717a] uppercase tracking-wider">Tempo</span>
          </div>
          <p className="font-mono text-xl font-bold text-[#fafafa]">
            {resultado.tempo_total_ms < 1000
              ? `${resultado.tempo_total_ms.toFixed(0)}ms`
              : `${(resultado.tempo_total_ms / 1000).toFixed(1)}s`}
          </p>
          <p className="text-[10px] text-[#52525b] mt-0.5">tempo total</p>
        </div>
      </div>

      {/* Intervalo de confianca */}
      <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-[#71717a] uppercase tracking-wider">
            Intervalo de Confianca 95%
          </span>
          <span className="font-mono text-xs text-[#a1a1aa]">
            [{(ci[0] * 100).toFixed(1)}%, {(ci[1] * 100).toFixed(1)}%]
          </span>
        </div>

        {/* Visualizacao do intervalo */}
        <div className="relative h-4 bg-[#27272a] rounded-full overflow-hidden">
          {/* Regiao do intervalo */}
          <div
            className="absolute inset-y-0 rounded-full"
            style={{
              left: `${ci[0] * 100}%`,
              width: `${(ci[1] - ci[0]) * 100}%`,
              backgroundColor: `${corDoScore(resultado.score_calibrado)}33`,
              borderLeft: `2px solid ${corDoScore(resultado.score_calibrado)}`,
              borderRight: `2px solid ${corDoScore(resultado.score_calibrado)}`,
            }}
          />
          {/* Ponto central (score calibrado) */}
          <div
            className="absolute inset-y-0 w-0.5 rounded-full"
            style={{
              left: `${resultado.score_calibrado * 100}%`,
              backgroundColor: corDoScore(resultado.score_calibrado),
              boxShadow: `0 0 6px ${corDoScore(resultado.score_calibrado)}`,
            }}
          />
        </div>

        {/* Rotulos de escala */}
        <div className="flex justify-between mt-1">
          {['0%', '25%', '50%', '75%', '100%'].map((l) => (
            <span key={l} className="text-[9px] text-[#52525b] font-mono">{l}</span>
          ))}
        </div>
      </div>

      {/* Radar de concordancia entre modelos */}
      {dadosRadar.length >= 3 && (
        <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
          <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-3">
            Radar de Concordancia
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={dadosRadar} cx="50%" cy="50%" outerRadius="75%">
              <PolarGrid stroke="#27272a" />
              <PolarAngleAxis
                dataKey="modelo"
                tick={{ fill: '#52525b', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              />
              <PolarRadiusAxis
                domain={[0, 100]}
                tick={{ fill: '#3f3f46', fontSize: 8, fontFamily: 'JetBrains Mono, monospace' }}
                axisLine={false}
                tickCount={4}
              />
              <Tooltip content={<TooltipRadar />} />
              <Radar
                name="Score"
                dataKey="score"
                stroke="#818cf8"
                fill="#818cf8"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Distribuicao dos scores por modelo (barras) */}
      {dadosGrafico.length > 1 && (
        <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
          <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-3">
            Distribuicao por Modelo
          </p>
          <ResponsiveContainer width="100%" height={100}>
            <BarChart data={dadosGrafico} barSize={20} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <XAxis
                dataKey="modelo"
                tick={false}
                axisLine={{ stroke: '#27272a' }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#52525b', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
                axisLine={false}
                tickLine={false}
                tickCount={3}
              />
              <Tooltip content={<TooltipCustomizado />} cursor={{ fill: '#ffffff08' }} />
              <Bar dataKey="score" radius={[2, 2, 0, 0]}>
                {dadosGrafico.map((entry) => (
                  <Cell key={entry.modelo} fill={corDoScore(entry.score / 100)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Tabela detalhada por modelo */}
      {scores_individuais.length > 0 && (
        <div className="rounded-md border border-[#27272a] bg-[#18181b] overflow-hidden">
          <p className="text-[10px] text-[#71717a] uppercase tracking-wider px-3 pt-3 pb-2">
            Detalhes por Modelo
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[#27272a]">
                <th className="text-left px-3 py-1.5 text-[#52525b] font-medium text-[10px] uppercase tracking-wider">Modelo</th>
                <th className="text-right px-3 py-1.5 text-[#52525b] font-medium text-[10px] uppercase tracking-wider">Score</th>
                <th className="text-right px-3 py-1.5 text-[#52525b] font-medium text-[10px] uppercase tracking-wider">Tempo</th>
                <th className="text-center px-3 py-1.5 text-[#52525b] font-medium text-[10px] uppercase tracking-wider">Decisao</th>
              </tr>
            </thead>
            <tbody>
              {scores_individuais.map((s) => (
                <tr key={s.modelo_id} className="border-b border-[#1f1f23] hover:bg-[#1f1f23]">
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-1.5">
                      <Cpu size={10} className="text-[#3f3f46]" />
                      <span className="text-[#a1a1aa] truncate max-w-[180px]">{s.modelo_nome}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <span className="font-mono font-bold" style={{ color: corDoScore(s.score) }}>
                      {(s.score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <span className="font-mono text-[#71717a] flex items-center justify-end gap-1">
                      <Clock size={9} className="text-[#3f3f46]" />
                      {s.tempo_ms < 1000
                        ? `${s.tempo_ms.toFixed(0)}ms`
                        : `${(s.tempo_ms / 1000).toFixed(1)}s`}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span
                      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium"
                      style={{
                        color: corDoScore(s.score),
                        backgroundColor: `${corDoScore(s.score)}18`,
                      }}
                    >
                      {s.score < 0.5 ? 'Real' : 'IA'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Barra de calibracao */}
      <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
        <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2">
          Distribuicao de Confianca dos Modelos
        </p>
        <div className="flex gap-1 h-6">
          {scores_individuais.map((s) => {
            const pct = s.score * 100
            return (
              <div
                key={s.modelo_id}
                className="flex-1 rounded-sm relative group"
                style={{ backgroundColor: `${corDoScore(s.score)}33` }}
              >
                <div
                  className="absolute bottom-0 left-0 right-0 rounded-sm transition-all"
                  style={{
                    height: `${pct}%`,
                    backgroundColor: corDoScore(s.score),
                  }}
                />
                {/* Tooltip on hover */}
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:block">
                  <div className="bg-[#1f1f23] border border-[#3f3f46] rounded px-1.5 py-0.5 text-[9px] font-mono text-[#a1a1aa] whitespace-nowrap">
                    {s.modelo_nome.slice(0, 15)}: {pct.toFixed(0)}%
                  </div>
                </div>
              </div>
            )
          })}
        </div>
        <div className="flex justify-between mt-1 text-[9px] text-[#3f3f46]">
          <span>Real</span>
          <span>IA</span>
        </div>
      </div>
    </div>
  )
}
