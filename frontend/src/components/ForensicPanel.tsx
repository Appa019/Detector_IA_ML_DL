/**
 * ForensicPanel — Painel forense com abas de análise detalhada
 * GradCAM, FFT, Noise Print, Histograma RGB, Metadados EXIF
 */

import { useState, useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import {
  Layers, Activity, Fingerprint, BarChart3, FileText,
  AlertTriangle, CheckCircle2, ShieldAlert, Camera,
  Sparkles, Waves, ScanSearch,
} from 'lucide-react'
import clsx from 'clsx'
import type { ResultadoAnalise } from '../api'

interface ForensicPanelProps {
  resultado: ResultadoAnalise
}

type AbaForense = 'gradcam' | 'ela' | 'wavelet' | 'fft' | 'ruido' | 'histograma' | 'metadados'

const ABAS: { id: AbaForense; label: string; icone: typeof Layers }[] = [
  { id: 'gradcam', label: 'Regioes Suspeitas', icone: Layers },
  { id: 'ela', label: 'ELA', icone: Sparkles },
  { id: 'wavelet', label: 'Wavelet', icone: Waves },
  { id: 'fft', label: 'Espectral', icone: Activity },
  { id: 'ruido', label: 'Ruido', icone: Fingerprint },
  { id: 'histograma', label: 'Histograma', icone: BarChart3 },
  { id: 'metadados', label: 'Metadados', icone: FileText },
]

function corNivelSuspeita(nivel: string): string {
  switch (nivel) {
    case 'nenhuma': return '#22c55e'
    case 'baixa': return '#14b8a6'
    case 'media': return '#f59e0b'
    case 'alta': return '#ef4444'
    case 'muito_alta': return '#dc2626'
    default: return '#71717a'
  }
}

function TooltipHistograma({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[#1f1f23] border border-[#3f3f46] rounded px-2 py-1.5 text-xs shadow-xl">
      <p className="text-[#71717a] mb-1 font-mono">Intensidade: {label}</p>
      {payload.map((entry: any) => (
        <p key={entry.name} className="font-mono" style={{ color: entry.color }}>
          {entry.name}: {(entry.value * 100).toFixed(3)}%
        </p>
      ))}
    </div>
  )
}

// --- Sub-componente: Aba GradCAM ---
function AbaGradCAM({ resultado }: { resultado: ResultadoAnalise }) {
  const [opacidade, setOpacidade] = useState(70)

  if (!resultado.heatmap_gradcam) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-[#52525b]">
        <Layers size={24} className="mb-2 text-[#3f3f46]" />
        <p className="text-sm">GradCAM nao disponivel para esta analise</p>
        <p className="text-xs mt-1 text-[#3f3f46]">O modelo pode nao suportar GradCAM ou a biblioteca nao esta instalada</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-[#71717a]">
          Regioes mais quentes indicam areas que mais influenciaram a decisao do modelo
        </p>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#52525b]">Opacidade</span>
          <input
            type="range"
            min={0}
            max={100}
            value={opacidade}
            onChange={(e) => setOpacidade(Number(e.target.value))}
            className="w-20 h-1 accent-[#6366f1]"
          />
          <span className="text-[10px] font-mono text-[#71717a] w-8">{opacidade}%</span>
        </div>
      </div>

      <div className="rounded-lg border border-[#27272a] bg-[#09090b] p-2 flex items-center justify-center">
        <div className="relative">
          <img
            src={`data:image/png;base64,${resultado.heatmap_gradcam}`}
            alt="Mapa GradCAM"
            className="max-h-[360px] rounded"
            style={{ opacity: opacidade / 100 }}
          />
        </div>
      </div>

      {/* Legenda */}
      <div className="flex items-center justify-center gap-2">
        <div className="flex items-center gap-1">
          <div className="w-3 h-2 rounded-sm bg-[#440154]" />
          <span className="text-[9px] text-[#52525b]">Normal</span>
        </div>
        <div className="w-16 h-2 rounded-sm" style={{
          background: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)'
        }} />
        <div className="flex items-center gap-1">
          <div className="w-3 h-2 rounded-sm bg-[#fde725]" />
          <span className="text-[9px] text-[#52525b]">Suspeito</span>
        </div>
      </div>
    </div>
  )
}

// --- Sub-componente: Aba ELA ---
function AbaELA({ resultado }: { resultado: ResultadoAnalise }) {
  const scoreEla = resultado.score_ela
  const temMapa = !!resultado.mapa_ela

  if (scoreEla == null && !temMapa) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-[#52525b]">
        <Sparkles size={24} className="mb-2 text-[#3f3f46]" />
        <p className="text-sm">Analise ELA nao disponivel</p>
        <p className="text-xs mt-1 text-[#3f3f46]">Error Level Analysis detecta regioes manipuladas via re-compressao JPEG</p>
      </div>
    )
  }

  const elaPct = scoreEla != null ? Math.round(scoreEla * 100) : null
  const ehSuspeito = scoreEla != null && scoreEla > 0.6

  return (
    <div className="space-y-3">
      <p className="text-xs text-[#71717a]">
        Error Level Analysis (ELA): re-comprime a imagem em JPEG e compara com o original.
        Regioes com niveis de erro uniformes sugerem geracao por IA.
      </p>

      {temMapa && (
        <div className="rounded-lg border border-[#27272a] bg-[#09090b] p-2 flex items-center justify-center">
          <img
            src={`data:image/png;base64,${resultado.mapa_ela}`}
            alt="Mapa ELA"
            className="max-h-[360px] rounded"
          />
        </div>
      )}

      {elaPct != null && (
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
            <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2">Score ELA</p>
            <div className="flex items-end gap-2 mb-2">
              <span className="font-mono text-lg font-bold" style={{
                color: ehSuspeito ? '#ef4444' : scoreEla! > 0.4 ? '#f59e0b' : '#22c55e'
              }}>
                {elaPct}%
              </span>
            </div>
            <div className="h-2 bg-[#27272a] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${elaPct}%`,
                  backgroundColor: ehSuspeito ? '#ef4444' : scoreEla! > 0.4 ? '#f59e0b' : '#22c55e',
                }}
              />
            </div>
            <div className="flex justify-between mt-1 text-[9px] text-[#3f3f46]">
              <span>Real</span>
              <span>IA</span>
            </div>
          </div>

          <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
            <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2">Interpretacao</p>
            <p className="text-xs text-[#a1a1aa]">
              {ehSuspeito
                ? 'ELA uniforme detectado — tipico de imagens geradas por IA onde toda a imagem tem nivel de compressao similar.'
                : scoreEla! > 0.4
                  ? 'ELA moderadamente uniforme — pode indicar edicao ou geracao parcial.'
                  : 'ELA com variacao natural — consistente com fotografia real (texturas variadas).'}
            </p>
          </div>
        </div>
      )}

      {/* Legenda */}
      <div className="flex items-center justify-center gap-2">
        <div className="flex items-center gap-1">
          <div className="w-3 h-2 rounded-sm bg-[#000004]" />
          <span className="text-[9px] text-[#52525b]">Baixo erro</span>
        </div>
        <div className="w-16 h-2 rounded-sm" style={{
          background: 'linear-gradient(to right, #000004, #bc3754, #fcffa4)'
        }} />
        <div className="flex items-center gap-1">
          <div className="w-3 h-2 rounded-sm bg-[#fcffa4]" />
          <span className="text-[9px] text-[#52525b]">Alto erro</span>
        </div>
      </div>
    </div>
  )
}

// --- Sub-componente: Aba Wavelet / Consistencia de Ruido ---
function AbaWavelet({ resultado }: { resultado: ResultadoAnalise }) {
  const scoreWavelet = resultado.score_wavelet
  const temMapaIncons = !!resultado.mapa_inconsistencia_ruido

  if (scoreWavelet == null && !temMapaIncons) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-[#52525b]">
        <Waves size={24} className="mb-2 text-[#3f3f46]" />
        <p className="text-sm">Analise wavelet nao disponivel</p>
        <p className="text-xs mt-1 text-[#3f3f46]">Decomposicao multi-escala para detectar artefatos em diferentes resolucoes</p>
      </div>
    )
  }

  const waveletPct = scoreWavelet != null ? Math.round(scoreWavelet * 100) : null
  const ehSuspeito = scoreWavelet != null && scoreWavelet > 0.6

  return (
    <div className="space-y-3">
      <p className="text-xs text-[#71717a]">
        Analise wavelet e mapa de consistencia de ruido. Imagens de IA tendem a ter menos detalhes finos
        e ruido mais uniforme entre regioes.
      </p>

      {/* Mapa de inconsistencia de ruido */}
      {temMapaIncons && (
        <div>
          <p className="text-[10px] text-[#52525b] uppercase tracking-widest font-semibold mb-2">
            Mapa de Inconsistencia de Ruido
          </p>
          <div className="rounded-lg border border-[#27272a] bg-[#09090b] p-2 flex items-center justify-center">
            <img
              src={`data:image/png;base64,${resultado.mapa_inconsistencia_ruido}`}
              alt="Mapa de Inconsistencia de Ruido"
              className="max-h-[320px] rounded"
            />
          </div>
          <div className="flex items-center justify-center gap-2 mt-2">
            <div className="flex items-center gap-1">
              <div className="w-3 h-2 rounded-sm bg-[#000004]" />
              <span className="text-[9px] text-[#52525b]">Consistente</span>
            </div>
            <div className="w-16 h-2 rounded-sm" style={{
              background: 'linear-gradient(to right, #000004, #bc3754, #fcffa4)'
            }} />
            <div className="flex items-center gap-1">
              <div className="w-3 h-2 rounded-sm bg-[#fcffa4]" />
              <span className="text-[9px] text-[#52525b]">Inconsistente</span>
            </div>
          </div>
        </div>
      )}

      {waveletPct != null && (
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
            <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2">Score Wavelet</p>
            <div className="flex items-end gap-2 mb-2">
              <span className="font-mono text-lg font-bold" style={{
                color: ehSuspeito ? '#ef4444' : scoreWavelet! > 0.4 ? '#f59e0b' : '#22c55e'
              }}>
                {waveletPct}%
              </span>
            </div>
            <div className="h-2 bg-[#27272a] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${waveletPct}%`,
                  backgroundColor: ehSuspeito ? '#ef4444' : scoreWavelet! > 0.4 ? '#f59e0b' : '#22c55e',
                }}
              />
            </div>
            <div className="flex justify-between mt-1 text-[9px] text-[#3f3f46]">
              <span>Real</span>
              <span>IA</span>
            </div>
          </div>

          <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
            <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2">Interpretacao</p>
            <p className="text-xs text-[#a1a1aa]">
              {ehSuspeito
                ? 'Pouca energia em detalhes finos — tipico de geradores de IA que produzem texturas suavizadas.'
                : scoreWavelet! > 0.4
                  ? 'Energia de detalhes moderada — resultado inconclusivo, possivel edicao.'
                  : 'Alta energia em detalhes finos — consistente com fotografia real e texturas naturais.'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

// --- Sub-componente: Aba FFT ---
function AbaFFT({ resultado }: { resultado: ResultadoAnalise }) {
  const feat = resultado.features_frequencia

  if (!resultado.espectro_fft && !feat) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-[#52525b]">
        <Activity size={24} className="mb-2 text-[#3f3f46]" />
        <p className="text-sm">Analise espectral nao disponivel</p>
      </div>
    )
  }

  // Razao HF/LF: valores tipicos para IA ~0.1-0.3, para real ~0.4-0.7
  const razaoHF = feat?.razao_hf_lf ?? 0
  const razaoHFPct = Math.min(razaoHF * 100, 100)
  const ehSuspeitoFreq = razaoHF < 0.35

  return (
    <div className="space-y-3">
      <p className="text-xs text-[#71717a]">
        Distribuicao de energia no dominio de frequencia. Imagens de IA tendem a ter menos energia em altas frequencias.
      </p>

      {resultado.espectro_fft && (
        <div className="rounded-lg border border-[#27272a] bg-[#09090b] p-2 flex items-center justify-center">
          <img
            src={`data:image/png;base64,${resultado.espectro_fft}`}
            alt="Espectro FFT 2D"
            className="max-h-[320px] rounded"
          />
        </div>
      )}

      {feat && (
        <div className="grid grid-cols-2 gap-2">
          {/* Razao alta/baixa frequencia */}
          <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
            <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2">Razao Alta Freq.</p>
            <div className="flex items-end gap-2 mb-2">
              <span className="font-mono text-lg font-bold" style={{ color: ehSuspeitoFreq ? '#f59e0b' : '#22c55e' }}>
                {(razaoHF * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-[#27272a] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${razaoHFPct}%`,
                  backgroundColor: ehSuspeitoFreq ? '#f59e0b' : '#22c55e',
                }}
              />
            </div>
            <div className="flex justify-between mt-1 text-[9px] text-[#3f3f46]">
              <span>IA tipico</span>
              <span>Real tipico</span>
            </div>
          </div>

          {/* Energia espectral */}
          <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
            <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-2">Metricas Espectrais</p>
            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-[#52525b]">Media espectro</span>
                <span className="font-mono text-[#a1a1aa]">{feat.media_espectro.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#52525b]">Desvio espectro</span>
                <span className="font-mono text-[#a1a1aa]">{feat.desvio_espectro.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#52525b]">Inclinacao</span>
                <span className="font-mono text-[#a1a1aa]">{feat.inclinacao_espectral.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#52525b]">Razao DCT</span>
                <span className="font-mono text-[#a1a1aa]">{feat.razao_energia_dct.toFixed(3)}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// --- Sub-componente: Aba Ruido ---
function AbaRuido({ resultado }: { resultado: ResultadoAnalise }) {
  if (!resultado.noise_print) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-[#52525b]">
        <Fingerprint size={24} className="mb-2 text-[#3f3f46]" />
        <p className="text-sm">Analise de ruido nao disponivel</p>
      </div>
    )
  }

  const uniformidade = resultado.uniformidade ?? 0
  const uniformidadePct = Math.round(uniformidade * 100)
  const ehSuspeitoRuido = uniformidade > 0.6

  return (
    <div className="space-y-3">
      <p className="text-xs text-[#71717a]">
        Impressao digital de ruido. Imagens de IA tendem a ter ruido mais uniforme que fotografias reais.
      </p>

      <div className="rounded-lg border border-[#27272a] bg-[#09090b] p-2 flex items-center justify-center">
        <img
          src={`data:image/png;base64,${resultado.noise_print}`}
          alt="Noise Print"
          className="max-h-[320px] rounded"
        />
      </div>

      <div className="grid grid-cols-2 gap-2">
        {/* Score de pixels */}
        {resultado.score_pixels != null && (
          <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
            <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-1">Score Pixels</p>
            <p className="font-mono text-lg font-bold" style={{
              color: resultado.score_pixels > 0.6 ? '#ef4444' : resultado.score_pixels > 0.4 ? '#f59e0b' : '#22c55e'
            }}>
              {(resultado.score_pixels * 100).toFixed(1)}%
            </p>
            <p className="text-[10px] text-[#3f3f46] mt-0.5">probabilidade IA (pixels)</p>
          </div>
        )}

        {/* Uniformidade */}
        <div className="rounded-md border border-[#27272a] bg-[#18181b] p-3">
          <p className="text-[10px] text-[#71717a] uppercase tracking-wider mb-1">Uniformidade Ruido</p>
          <p className="font-mono text-lg font-bold" style={{
            color: ehSuspeitoRuido ? '#f59e0b' : '#22c55e'
          }}>
            {uniformidadePct}%
          </p>
          <p className="text-[10px] text-[#3f3f46] mt-0.5">
            {ehSuspeitoRuido ? 'alta (tipico IA)' : 'normal (tipico real)'}
          </p>
        </div>
      </div>

      {/* Legenda do noise print */}
      <div className="flex items-center justify-center gap-2">
        <div className="flex items-center gap-1">
          <div className="w-3 h-2 rounded-sm bg-[#2166ac]" />
          <span className="text-[9px] text-[#52525b]">Negativo</span>
        </div>
        <div className="w-16 h-2 rounded-sm" style={{
          background: 'linear-gradient(to right, #2166ac, #f7f7f7, #b2182b)'
        }} />
        <div className="flex items-center gap-1">
          <div className="w-3 h-2 rounded-sm bg-[#b2182b]" />
          <span className="text-[9px] text-[#52525b]">Positivo</span>
        </div>
      </div>
    </div>
  )
}

// --- Sub-componente: Aba Histograma RGB ---
function AbaHistograma({ resultado }: { resultado: ResultadoAnalise }) {
  const dados = useMemo(() => {
    if (!resultado.histograma_rgb) return []
    const { vermelho, verde, azul } = resultado.histograma_rgb
    // Downsample para performance (cada 4 bins)
    const passo = 4
    const pontos = []
    for (let i = 0; i < 256; i += passo) {
      let r = 0, g = 0, b = 0
      for (let j = i; j < Math.min(i + passo, 256); j++) {
        r += vermelho[j] ?? 0
        g += verde[j] ?? 0
        b += azul[j] ?? 0
      }
      pontos.push({
        intensidade: i,
        Vermelho: r / passo,
        Verde: g / passo,
        Azul: b / passo,
      })
    }
    return pontos
  }, [resultado.histograma_rgb])

  if (!resultado.histograma_rgb || dados.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-[#52525b]">
        <BarChart3 size={24} className="mb-2 text-[#3f3f46]" />
        <p className="text-sm">Histograma nao disponivel</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-[#71717a]">
        Distribuicao de intensidade por canal RGB. Picos abruptos ou distribuicoes muito suaves podem indicar manipulacao.
      </p>

      <div className="rounded-lg border border-[#27272a] bg-[#09090b] p-3">
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={dados} margin={{ top: 5, right: 5, bottom: 5, left: -15 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f23" />
            <XAxis
              dataKey="intensidade"
              tick={{ fill: '#52525b', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={{ stroke: '#27272a' }}
              tickLine={false}
              tickCount={5}
            />
            <YAxis
              tick={{ fill: '#52525b', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
            />
            <Tooltip content={<TooltipHistograma />} />
            <Area type="monotone" dataKey="Vermelho" stroke="#ef4444" fill="#ef444422" strokeWidth={1.5} dot={false} />
            <Area type="monotone" dataKey="Verde" stroke="#22c55e" fill="#22c55e22" strokeWidth={1.5} dot={false} />
            <Area type="monotone" dataKey="Azul" stroke="#3b82f6" fill="#3b82f622" strokeWidth={1.5} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legenda */}
      <div className="flex items-center justify-center gap-4 text-[10px]">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#ef4444]" /> Vermelho</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#22c55e]" /> Verde</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#3b82f6]" /> Azul</span>
      </div>
    </div>
  )
}

// --- Sub-componente: Aba Metadados ---
function AbaMetadados({ resultado }: { resultado: ResultadoAnalise }) {
  const metadados = resultado.metadados
  const indicadores = resultado.indicadores_ia

  const temDados = metadados && Object.keys(metadados).length > 0

  return (
    <div className="space-y-3">
      {/* Indicadores de IA */}
      {indicadores && (
        <div className={clsx(
          'rounded-md border p-3',
          indicadores.pontuacao_suspeita >= 3
            ? 'border-[#7f1d1d] bg-[#7f1d1d22]'
            : indicadores.pontuacao_suspeita >= 2
              ? 'border-[#78350f] bg-[#78350f22]'
              : 'border-[#27272a] bg-[#18181b]'
        )}>
          <div className="flex items-center gap-2 mb-2">
            <ShieldAlert size={14} style={{ color: corNivelSuspeita(indicadores.nivel_suspeita) }} />
            <span className="text-xs font-semibold" style={{ color: corNivelSuspeita(indicadores.nivel_suspeita) }}>
              Suspeita de IA: {indicadores.nivel_suspeita.replace('_', ' ')}
            </span>
            <span className="ml-auto text-[10px] font-mono text-[#52525b]">
              {indicadores.pontuacao_suspeita}/4
            </span>
          </div>

          {indicadores.indicadores.length > 0 ? (
            <ul className="space-y-1">
              {indicadores.indicadores.map((ind, i) => (
                <li key={i} className="flex items-start gap-1.5 text-xs text-[#a1a1aa]">
                  <AlertTriangle size={10} className="flex-shrink-0 mt-0.5 text-[#f59e0b]" />
                  {ind}
                </li>
              ))}
            </ul>
          ) : (
            <p className="flex items-center gap-1.5 text-xs text-[#22c55e]">
              <CheckCircle2 size={10} />
              Nenhum indicador de IA detectado nos metadados
            </p>
          )}

          {indicadores.software_ia_detectado && indicadores.software_identificado && (
            <div className="mt-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#7f1d1d44] border border-[#7f1d1d]">
              <span className="text-[10px] font-semibold text-[#ef4444]">
                Software IA: {indicadores.software_identificado}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Status EXIF */}
      {indicadores?.sem_exif && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-md border border-[#78350f] bg-[#78350f22]">
          <Camera size={12} className="text-[#f59e0b]" />
          <span className="text-xs text-[#f59e0b]">
            Sem metadados EXIF — comum em imagens geradas por IA
          </span>
        </div>
      )}

      {/* Tabela de metadados */}
      {temDados ? (
        <div className="rounded-md border border-[#27272a] bg-[#18181b] overflow-hidden">
          <div className="max-h-[300px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[#18181b]">
                <tr className="border-b border-[#27272a]">
                  <th className="text-left px-3 py-2 text-[#71717a] font-medium uppercase tracking-wider text-[10px]">Campo</th>
                  <th className="text-left px-3 py-2 text-[#71717a] font-medium uppercase tracking-wider text-[10px]">Valor</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metadados).map(([campo, valor]) => (
                  <tr key={campo} className="border-b border-[#1f1f23] hover:bg-[#1f1f23]">
                    <td className="px-3 py-1.5 text-[#a1a1aa] font-mono">{campo}</td>
                    <td className="px-3 py-1.5 text-[#71717a] font-mono truncate max-w-[240px]">
                      {String(valor)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : !indicadores?.sem_exif ? (
        <div className="flex flex-col items-center justify-center py-8 text-[#52525b]">
          <FileText size={24} className="mb-2 text-[#3f3f46]" />
          <p className="text-sm">Nenhum metadado disponivel</p>
        </div>
      ) : null}
    </div>
  )
}

// --- Componente principal ---
export function ForensicPanel({ resultado }: ForensicPanelProps) {
  const [abaAtiva, setAbaAtiva] = useState<AbaForense>('gradcam')

  // Verifica quais abas tem dados
  const abasDisponiveis = useMemo(() => {
    const disp: Record<AbaForense, boolean> = {
      gradcam: !!resultado.heatmap_gradcam,
      ela: !!(resultado.score_ela != null || resultado.mapa_ela),
      wavelet: !!(resultado.score_wavelet != null || resultado.mapa_inconsistencia_ruido),
      fft: !!(resultado.espectro_fft || resultado.features_frequencia),
      ruido: !!(resultado.noise_print || resultado.score_pixels != null),
      histograma: !!resultado.histograma_rgb,
      metadados: !!(resultado.metadados || resultado.indicadores_ia),
    }
    return disp
  }, [resultado])

  // Conta abas com dados
  const totalComDados = Object.values(abasDisponiveis).filter(Boolean).length

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-[#71717a]">
          Analise Forense
        </h3>
        <span className="text-[10px] text-[#3f3f46]">
          {totalComDados} de {ABAS.length} analises disponiveis
        </span>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 overflow-x-auto pb-1">
        {ABAS.map(({ id, label, icone: Icone }) => {
          const temDados = abasDisponiveis[id]
          return (
            <button
              key={id}
              onClick={() => setAbaAtiva(id)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs whitespace-nowrap transition-colors',
                abaAtiva === id
                  ? 'bg-[#27272a] text-[#fafafa]'
                  : temDados
                    ? 'text-[#71717a] hover:bg-[#1f1f23] hover:text-[#a1a1aa]'
                    : 'text-[#3f3f46] hover:bg-[#1f1f23]'
              )}
            >
              <Icone size={12} />
              {label}
              {temDados && (
                <span className="w-1.5 h-1.5 rounded-full bg-[#6366f1]" />
              )}
            </button>
          )
        })}
      </div>

      {/* Conteudo da aba */}
      <div className="min-h-[200px]">
        {abaAtiva === 'gradcam' && <AbaGradCAM resultado={resultado} />}
        {abaAtiva === 'ela' && <AbaELA resultado={resultado} />}
        {abaAtiva === 'wavelet' && <AbaWavelet resultado={resultado} />}
        {abaAtiva === 'fft' && <AbaFFT resultado={resultado} />}
        {abaAtiva === 'ruido' && <AbaRuido resultado={resultado} />}
        {abaAtiva === 'histograma' && <AbaHistograma resultado={resultado} />}
        {abaAtiva === 'metadados' && <AbaMetadados resultado={resultado} />}
      </div>
    </div>
  )
}
