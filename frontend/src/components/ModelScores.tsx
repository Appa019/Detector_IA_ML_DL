/**
 * ModelScores — Tabela de scores individuais por modelo do ensemble
 * Barras horizontais com gradiente de cor por score
 */

import { Clock } from 'lucide-react'
import type { ScoreIndividual } from '../api'

interface ModelScoresProps {
  scores: ScoreIndividual[]
}

function corDaBarra(score: number): string {
  if (score < 0.25) return '#22c55e'
  if (score < 0.50) return '#14b8a6'
  if (score < 0.75) return '#f59e0b'
  return '#ef4444'
}


export function ModelScores({ scores }: ModelScoresProps) {
  if (!scores || scores.length === 0) return null

  const ordenados = [...scores].sort((a, b) => b.score - a.score)

  return (
    <div className="animate-fade-in-up delay-2">
      <h3 className="text-xs font-semibold uppercase tracking-widest text-[#71717a] mb-3">
        Scores por Modelo
      </h3>

      <div className="flex flex-col gap-2">
        {ordenados.map((item, i) => {
          const pct = Math.round(item.score * 100)
          const cor = corDaBarra(item.score)

          return (
            <div
              key={item.modelo_id}
              className="group rounded-md border border-[#27272a] bg-[#18181b] p-3 hover:border-[#3f3f46] transition-colors duration-150"
              style={{ animationDelay: `${i * 0.06}s` }}
            >
              {/* Cabeçalho da linha */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 min-w-0">
                  {/* Indicador de cor */}
                  <span
                    className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: cor }}
                  />
                  <span className="text-xs text-[#a1a1aa] truncate font-medium">
                    {item.modelo_nome}
                  </span>
                </div>

                <div className="flex items-center gap-3 flex-shrink-0">
                  {/* Tempo de inferência */}
                  <span className="flex items-center gap-1 text-[10px] text-[#52525b]">
                    <Clock size={10} />
                    {item.tempo_ms.toFixed(0)}ms
                  </span>

                  {/* Score em destaque */}
                  <span
                    className="font-mono text-sm font-bold tabular-nums"
                    style={{ color: cor }}
                  >
                    {pct}%
                  </span>
                </div>
              </div>

              {/* Barra de progresso */}
              <div className="relative h-1.5 rounded-full bg-[#27272a] overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 rounded-full animate-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: cor,
                    boxShadow: `0 0 6px ${cor}66`,
                    animationDelay: `${0.3 + i * 0.07}s`,
                  }}
                />
              </div>

              {/* Arquitetura como metadado sutil */}
              <div className="mt-1.5 flex items-center gap-1">
                <span className="text-[10px] text-[#52525b] font-mono">
                  {item.modelo_id}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
