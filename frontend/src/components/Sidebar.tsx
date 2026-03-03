/**
 * Sidebar — Painel lateral esquerdo
 * Status da GPU, lista de modelos com toggles, informações de VRAM
 */

import { useState } from 'react'
import { ChevronDown, Cpu, Layers, MemoryStick } from 'lucide-react'
import clsx from 'clsx'
import type { InfoGPU, InfoModelo } from '../api'

interface SidebarProps {
  gpu: InfoGPU | null
  modelos: InfoModelo[]
  modelosAtivos: Set<string>
  onToggleModelo: (id: string) => void
  carregando: boolean
}

interface SecaoProps {
  titulo: string
  icone: React.ReactNode
  children: React.ReactNode
  defaultAberta?: boolean
}

// Componente de seção colapsável
function Secao({ titulo, icone, children, defaultAberta = true }: SecaoProps) {
  const [aberta, setAberta] = useState(defaultAberta)

  return (
    <div className="border-b border-[#27272a] last:border-b-0">
      <button
        onClick={() => setAberta(!aberta)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#1f1f23] transition-colors duration-150 group"
      >
        <div className="flex items-center gap-2">
          <span className="text-[#52525b] group-hover:text-[#71717a] transition-colors">{icone}</span>
          <span className="text-[10px] font-semibold uppercase tracking-widest text-[#71717a]">
            {titulo}
          </span>
        </div>
        <ChevronDown
          size={12}
          className={clsx(
            'text-[#52525b] transition-transform duration-200',
            aberta ? 'rotate-0' : '-rotate-90',
          )}
        />
      </button>

      {aberta && (
        <div className="animate-fade-in">
          {children}
        </div>
      )}
    </div>
  )
}

// Barra de uso de VRAM
function BarraVRAM({ usada, total }: { usada: number; total: number }) {
  const pct = total > 0 ? (usada / total) * 100 : 0
  const cor = pct > 80 ? '#ef4444' : pct > 60 ? '#f59e0b' : '#22c55e'

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px]">
        <span className="text-[#52525b]">VRAM usada</span>
        <span className="font-mono text-[#a1a1aa]">
          {(usada / 1024).toFixed(1)} / {(total / 1024).toFixed(1)} GB
        </span>
      </div>
      <div className="h-1.5 bg-[#27272a] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            backgroundColor: cor,
            boxShadow: `0 0 4px ${cor}66`,
          }}
        />
      </div>
      <div className="flex justify-end">
        <span className="font-mono text-[10px]" style={{ color: cor }}>
          {pct.toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

// Badge de VRAM de cada modelo
function BadgeVRAM({ mb }: { mb: number }) {
  const gb = mb >= 1024 ? `${(mb / 1024).toFixed(1)}GB` : `${mb}MB`
  return (
    <span className="font-mono text-[10px] text-[#52525b] bg-[#27272a] px-1.5 py-0.5 rounded border border-[#3f3f46]">
      {gb}
    </span>
  )
}

// Skeleton de carregamento
function SkeletonGPU() {
  return (
    <div className="px-4 pb-4 space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="h-3 bg-[#27272a] rounded animate-skeleton" style={{ width: `${70 + i * 10}%` }} />
      ))}
    </div>
  )
}

export function Sidebar({ gpu, modelos, modelosAtivos, onToggleModelo, carregando }: SidebarProps) {
  // Agrupa modelos por tipo
  const porTipo = modelos.reduce((acc, m) => {
    const k = m.tipo || 'outro'
    if (!acc[k]) acc[k] = []
    acc[k].push(m)
    return acc
  }, {} as Record<string, InfoModelo[]>)

  const vramTotalSelecionada = modelos
    .filter((m) => modelosAtivos.has(m.id))
    .reduce((acc, m) => acc + m.vram_fp16_mb, 0)

  return (
    <aside className="w-[280px] flex-shrink-0 bg-[#18181b] border-r border-[#27272a] flex flex-col h-full overflow-y-auto">

      {/* Status GPU */}
      <Secao titulo="Hardware" icone={<Cpu size={12} />}>
        {carregando ? <SkeletonGPU /> : (
          <div className="px-4 pb-4 space-y-3">
            {gpu ? (
              <>
                {/* Nome da GPU */}
                <div className="rounded border border-[#27272a] bg-[#09090b] p-2.5">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={clsx(
                      'w-1.5 h-1.5 rounded-full flex-shrink-0',
                      gpu.disponivel ? 'bg-[#22c55e]' : 'bg-[#ef4444]',
                    )} />
                    <span className="text-[10px] text-[#52525b] uppercase tracking-wider">
                      {gpu.disponivel ? 'GPU Ativa' : 'CPU Mode'}
                    </span>
                  </div>
                  <p className="text-xs text-[#fafafa] font-medium truncate" title={gpu.nome}>
                    {gpu.nome || 'CPU'}
                  </p>
                  <p className="text-[10px] text-[#52525b] font-mono mt-0.5">{gpu.dispositivo}</p>
                </div>

                {/* Barra VRAM */}
                {gpu.disponivel && (
                  <BarraVRAM usada={gpu.vram_usada_mb} total={gpu.vram_total_mb} />
                )}
              </>
            ) : (
              <p className="text-[10px] text-[#52525b]">Informações de GPU indisponíveis</p>
            )}
          </div>
        )}
      </Secao>

      {/* Lista de modelos */}
      <Secao titulo="Modelos" icone={<Layers size={12} />} defaultAberta>
        <div className="px-4 pb-4 space-y-1">

          {/* VRAM selecionada */}
          {vramTotalSelecionada > 0 && (
            <div className="flex items-center justify-between mb-2 pt-0.5">
              <span className="text-[10px] text-[#52525b]">VRAM selecionada</span>
              <span className="font-mono text-[10px] text-[#6366f1]">
                {(vramTotalSelecionada / 1024).toFixed(1)} GB
              </span>
            </div>
          )}

          {carregando ? (
            <div className="space-y-2">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-8 bg-[#27272a] rounded animate-skeleton" />
              ))}
            </div>
          ) : modelos.length === 0 ? (
            <p className="text-[10px] text-[#52525b]">Nenhum modelo disponível</p>
          ) : (
            Object.entries(porTipo).map(([tipo, lista]) => (
              <div key={tipo} className="mb-3 last:mb-0">
                <p className="text-[9px] text-[#52525b] uppercase tracking-widest mb-1.5 font-semibold">
                  {tipo}
                </p>
                {lista.map((modelo) => {
                  const ativo = modelosAtivos.has(modelo.id)
                  return (
                    <label
                      key={modelo.id}
                      className={clsx(
                        'flex items-center gap-2.5 rounded px-2 py-1.5 cursor-pointer transition-colors duration-150 mb-0.5',
                        ativo
                          ? 'bg-[#1e1b4b] hover:bg-[#1e1b4b]'
                          : 'hover:bg-[#1f1f23]',
                      )}
                    >
                      {/* Checkbox customizado */}
                      <div
                        className={clsx(
                          'w-3.5 h-3.5 rounded-sm border flex items-center justify-center flex-shrink-0 transition-all duration-150',
                          ativo
                            ? 'bg-[#6366f1] border-[#6366f1]'
                            : 'bg-transparent border-[#3f3f46]',
                        )}
                        onClick={() => onToggleModelo(modelo.id)}
                      >
                        {ativo && (
                          <svg width="8" height="6" viewBox="0 0 8 6" fill="none">
                            <path d="M1 3L3 5L7 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                      </div>
                      <input
                        type="checkbox"
                        className="sr-only"
                        checked={ativo}
                        onChange={() => onToggleModelo(modelo.id)}
                        aria-label={`Ativar modelo ${modelo.nome_exibicao}`}
                      />

                      <span className={clsx(
                        'text-xs flex-1 min-w-0 truncate',
                        ativo ? 'text-[#a1a1aa]' : 'text-[#71717a]',
                      )} title={modelo.nome_exibicao}>
                        {modelo.nome_exibicao}
                      </span>

                      <BadgeVRAM mb={modelo.vram_fp16_mb} />
                    </label>
                  )
                })}
              </div>
            ))
          )}
        </div>
      </Secao>

      {/* Informação de VRAM dos modelos */}
      <Secao titulo="Info" icone={<MemoryStick size={12} />} defaultAberta={false}>
        <div className="px-4 pb-4 space-y-1.5 text-[10px] text-[#52525b]">
          <p>Selecione os modelos a usar na análise.</p>
          <p>Modelos desativados reduzem o consumo de VRAM mas podem diminuir a precisão.</p>
          <p className="text-[#3f3f46] mt-2 pt-2 border-t border-[#27272a]">
            Detector de IA v1.0.0
          </p>
        </div>
      </Secao>
    </aside>
  )
}
