/**
 * ResultCarousel — Navegação entre múltiplos resultados de análise
 * Exibe um ResultPanel por vez com setas, dots e teclado
 */

import { useEffect, useCallback, useRef } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { ResultadoAnalise, ResultadoVideo } from '../api'
import { ResultPanel } from './ResultPanel'

interface ItemResultado {
  resultado: ResultadoAnalise | ResultadoVideo
  nomeArquivo: string
  arquivoOriginal: File | null
}

interface ResultCarouselProps {
  resultados: ItemResultado[]
  indiceAtivo: number
  onMudarIndice: (indice: number) => void
}

function corDot(score: number): string {
  if (score < 0.25) return '#22c55e'
  if (score < 0.50) return '#14b8a6'
  if (score < 0.75) return '#f59e0b'
  return '#ef4444'
}

export function ResultCarousel({ resultados, indiceAtivo, onMudarIndice }: ResultCarouselProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  const irAnterior = useCallback(() => {
    onMudarIndice(Math.max(0, indiceAtivo - 1))
  }, [indiceAtivo, onMudarIndice])

  const irProximo = useCallback(() => {
    onMudarIndice(Math.min(resultados.length - 1, indiceAtivo + 1))
  }, [indiceAtivo, resultados.length, onMudarIndice])

  /* Navegação por teclado */
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'ArrowLeft') { e.preventDefault(); irAnterior() }
      if (e.key === 'ArrowRight') { e.preventDefault(); irProximo() }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [irAnterior, irProximo])

  const item = resultados[indiceAtivo]
  if (!item) return null

  const temAnterior = indiceAtivo > 0
  const temProximo  = indiceAtivo < resultados.length - 1

  return (
    <div ref={containerRef} className="space-y-3">
      {/* Barra de navegação */}
      <div className="flex items-center justify-between">
        {/* Seta esquerda */}
        <button
          onClick={irAnterior}
          disabled={!temAnterior}
          className="p-1.5 rounded hover:bg-[#27272a] text-[#71717a] hover:text-[#fafafa] disabled:opacity-25 disabled:cursor-not-allowed transition-colors"
          aria-label="Resultado anterior"
        >
          <ChevronLeft size={18} />
        </button>

        {/* Centro: dots + contador */}
        <div className="flex flex-col items-center gap-1.5">
          {/* Dots */}
          <div className="flex items-center gap-1.5">
            {resultados.map((r, i) => (
              <button
                key={i}
                onClick={() => onMudarIndice(i)}
                className="transition-all duration-200"
                aria-label={`Resultado ${i + 1}: ${r.nomeArquivo}`}
              >
                <span
                  className="block rounded-full transition-all duration-200"
                  style={{
                    width:  i === indiceAtivo ? 20 : 8,
                    height: 8,
                    backgroundColor: corDot(r.resultado.score_final),
                    opacity: i === indiceAtivo ? 1 : 0.4,
                    borderRadius: i === indiceAtivo ? 4 : 999,
                  }}
                />
              </button>
            ))}
          </div>

          {/* Contador + nome do arquivo */}
          <div className="flex items-center gap-2 text-[11px]">
            <span className="text-[#a1a1aa] font-mono">
              {indiceAtivo + 1} / {resultados.length}
            </span>
            <span className="text-[#52525b]">&mdash;</span>
            <span className="text-[#71717a] truncate max-w-[200px]" title={item.nomeArquivo}>
              {item.nomeArquivo}
            </span>
          </div>
        </div>

        {/* Seta direita */}
        <button
          onClick={irProximo}
          disabled={!temProximo}
          className="p-1.5 rounded hover:bg-[#27272a] text-[#71717a] hover:text-[#fafafa] disabled:opacity-25 disabled:cursor-not-allowed transition-colors"
          aria-label="Próximo resultado"
        >
          <ChevronRight size={18} />
        </button>
      </div>

      {/* Painel de resultado com transição */}
      <div className="carousel-slide" key={indiceAtivo}>
        <ResultPanel
          resultado={item.resultado}
          nomeArquivo={item.nomeArquivo}
          arquivoOriginal={item.arquivoOriginal}
        />
      </div>
    </div>
  )
}
