/**
 * GaugeChart — Medidor circular SVG de 0–100%
 * Exibe o score de IA com cores graduadas e animação de entrada.
 */

import { useMemo } from 'react'

interface GaugeChartProps {
  /** Valor de 0 a 1 (0.0 a 1.0) */
  valor: number
  classificacao: string
  /** Largura/altura em px */
  tamanho?: number
}

// Retorna cor baseada no score (0–1)
function corDoScore(v: number): string {
  if (v < 0.25) return '#22c55e'
  if (v < 0.50) return '#14b8a6'
  if (v < 0.75) return '#f59e0b'
  return '#ef4444'
}

function labelDoScore(v: number): string {
  if (v < 0.25) return 'Provavelmente Real'
  if (v < 0.50) return 'Possivelmente Real'
  if (v < 0.75) return 'Possivelmente IA'
  return 'Provavelmente IA'
}

function corFundoDoScore(v: number): string {
  if (v < 0.25) return '#14532d'
  if (v < 0.50) return '#134e4a'
  if (v < 0.75) return '#78350f'
  return '#7f1d1d'
}

export function GaugeChart({ valor, classificacao, tamanho = 220 }: GaugeChartProps) {
  // Geometria do arco — começa em 135° e vai até 405° (270° de ângulo total)
  const raio = 80
  const cx = tamanho / 2
  const cy = tamanho / 2 + 10
  const anguloInicio = 135
  const anguloTotal = 270
  const circunferencia = 2 * Math.PI * raio

  const { tracejado } = useMemo(() => {
    // Comprimento do arco ativo que representamos no SVG:
    // Usamos stroke-dasharray em uma "fatia" de 270°/360° da circunferência
    const fracao270 = (anguloTotal / 360) * circunferencia
    const comprimento = fracao270 * Math.min(Math.max(valor, 0), 1)
    const gap = fracao270 - comprimento
    // O restante do círculo (90°) deve ser invisível
    const invisivel = (90 / 360) * circunferencia
    return {
      comprimentoArco: comprimento,
      tracejado: `${comprimento} ${gap + invisivel}`,
    }
  }, [valor, circunferencia])

  // Offset para iniciar o arco em 135° (sentido horário desde o topo)
  // strokeDashoffset desloca o início do traçado
  const offset = useMemo(() => {
    // A posição "topo" do círculo SVG corresponde a -90° / -circunferência/4
    // Queremos iniciar em 135° = topo + 135°
    const inicioRad = ((anguloInicio - 90) / 360) * circunferencia
    return circunferencia - inicioRad
  }, [circunferencia])

  const cor = corDoScore(valor)
  const pct = Math.round(valor * 100)

  // Calcular posição do ponteiro (tick)
  const anguloAtual = anguloInicio + valor * anguloTotal
  const radAtual = ((anguloAtual - 90) * Math.PI) / 180
  const tickX = cx + (raio + 10) * Math.cos(radAtual)
  const tickY = cy + (raio + 10) * Math.sin(radAtual)
  const tickX2 = cx + (raio + 18) * Math.cos(radAtual)
  const tickY2 = cy + (raio + 18) * Math.sin(radAtual)

  return (
    <div className="flex flex-col items-center gap-3 animate-fade-in-up">
      <div className="relative" style={{ width: tamanho, height: tamanho }}>
        <svg
          width={tamanho}
          height={tamanho}
          viewBox={`0 0 ${tamanho} ${tamanho}`}
          aria-label={`Score de IA: ${pct}%`}
        >
          {/* Trilha de fundo (cinza) */}
          <circle
            cx={cx}
            cy={cy}
            r={raio}
            fill="none"
            stroke="#27272a"
            strokeWidth={12}
            strokeDasharray={`${(anguloTotal / 360) * circunferencia} ${(90 / 360) * circunferencia}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(0)"
          />

          {/* Arco ativo com cor do score */}
          <circle
            cx={cx}
            cy={cy}
            r={raio}
            fill="none"
            stroke={cor}
            strokeWidth={12}
            strokeDasharray={tracejado}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="animate-gauge"
            style={{
              filter: `drop-shadow(0 0 8px ${cor}66)`,
              transition: 'stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)',
            }}
          />

          {/* Marcadores de escala a cada 25% */}
          {[0, 0.25, 0.5, 0.75, 1].map((v) => {
            const ang = anguloInicio + v * anguloTotal
            const r = ((ang - 90) * Math.PI) / 180
            const mx1 = cx + (raio - 16) * Math.cos(r)
            const my1 = cy + (raio - 16) * Math.sin(r)
            const mx2 = cx + (raio - 8) * Math.cos(r)
            const my2 = cy + (raio - 8) * Math.sin(r)
            return (
              <line
                key={v}
                x1={mx1} y1={my1} x2={mx2} y2={my2}
                stroke="#52525b"
                strokeWidth={1.5}
                strokeLinecap="round"
              />
            )
          })}

          {/* Ponteiro */}
          <line
            x1={tickX} y1={tickY} x2={tickX2} y2={tickY2}
            stroke={cor}
            strokeWidth={3}
            strokeLinecap="round"
            style={{ transition: 'all 1.2s cubic-bezier(0.4,0,0.2,1)' }}
          />

          {/* Percentual central */}
          <text
            x={cx}
            y={cy - 4}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#fafafa"
            fontSize={tamanho * 0.18}
            fontWeight="700"
            fontFamily="'JetBrains Mono', 'Fira Code', monospace"
            style={{ letterSpacing: '-1px' }}
          >
            {pct}%
          </text>

          {/* Rótulo secundário */}
          <text
            x={cx}
            y={cy + tamanho * 0.11}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#71717a"
            fontSize={11}
            fontFamily="'Inter', system-ui, sans-serif"
          >
            score calibrado
          </text>

          {/* Rótulos 0% e 100% nas extremidades */}
          {[
            { label: '0%',    ang: anguloInicio,              offset: 14 },
            { label: '100%',  ang: anguloInicio + anguloTotal, offset: 14 },
          ].map(({ label, ang, offset: off }) => {
            const r = ((ang - 90) * Math.PI) / 180
            const lx = cx + (raio + off + 8) * Math.cos(r)
            const ly = cy + (raio + off + 8) * Math.sin(r)
            return (
              <text key={label} x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
                fill="#52525b" fontSize={9} fontFamily="'JetBrains Mono', monospace">
                {label}
              </text>
            )
          })}
        </svg>
      </div>

      {/* Badge de classificação */}
      <div
        className="px-3 py-1 rounded text-xs font-semibold uppercase tracking-widest transition-all duration-300"
        style={{ backgroundColor: corFundoDoScore(valor), color: cor, border: `1px solid ${cor}44` }}
      >
        {classificacao || labelDoScore(valor)}
      </div>
    </div>
  )
}
