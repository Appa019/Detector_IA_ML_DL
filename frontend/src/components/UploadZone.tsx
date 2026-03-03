/**
 * UploadZone — Área de upload com drag-and-drop
 * Suporta múltiplos arquivos: imagens (JPG, PNG, WebP) e vídeos (MP4, AVI, MOV, MKV)
 */

import { useState, useRef, useCallback } from 'react'
import { Upload, Image as IconeImagem, Film, X, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

export type TipoArquivo = 'imagem' | 'video' | null

interface UploadZoneProps {
  onAnalisar: (arquivos: File[], tipos: TipoArquivo[]) => void
  carregando: boolean
}

const EXTENSOES_IMAGEM = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
const EXTENSOES_VIDEO  = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
const MIME_IMAGEM = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
const MIME_VIDEO  = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm']

function detectarTipo(arquivo: File): TipoArquivo {
  if (MIME_IMAGEM.includes(arquivo.type)) return 'imagem'
  if (MIME_VIDEO.includes(arquivo.type)) return 'video'
  const ext = '.' + arquivo.name.split('.').pop()?.toLowerCase()
  if (EXTENSOES_IMAGEM.includes(ext)) return 'imagem'
  if (EXTENSOES_VIDEO.includes(ext)) return 'video'
  return null
}

function formatarTamanho(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

interface ArquivoValidado {
  arquivo: File
  tipo: TipoArquivo
  preview: string | null
}

export function UploadZone({ onAnalisar, carregando }: UploadZoneProps) {
  const [arrastando, setArrastando] = useState(false)
  const [arquivos, setArquivos] = useState<ArquivoValidado[]>([])
  const [erro, setErro] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const processarArquivos = useCallback((files: FileList | File[]) => {
    setErro(null)
    const novos: ArquivoValidado[] = []
    const erros: string[] = []

    for (const f of Array.from(files)) {
      const tipo = detectarTipo(f)
      if (!tipo) {
        erros.push(`"${f.name}": formato não suportado`)
        continue
      }
      if (f.size > 500 * 1024 * 1024) {
        erros.push(`"${f.name}": muito grande (limite: 500 MB)`)
        continue
      }
      const preview = tipo === 'imagem' ? URL.createObjectURL(f) : null
      novos.push({ arquivo: f, tipo, preview })
    }

    if (erros.length > 0) {
      setErro(erros.join('. '))
    }

    if (novos.length > 0) {
      setArquivos((prev) => [...prev, ...novos])
    }
  }, [])

  const removerArquivo = useCallback((indice: number) => {
    setArquivos((prev) => {
      const item = prev[indice]
      if (item.preview) URL.revokeObjectURL(item.preview)
      return prev.filter((_, i) => i !== indice)
    })
  }, [])

  const limparTodos = useCallback(() => {
    arquivos.forEach((a) => { if (a.preview) URL.revokeObjectURL(a.preview) })
    setArquivos([])
    setErro(null)
    if (inputRef.current) inputRef.current.value = ''
  }, [arquivos])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setArrastando(false)
    if (e.dataTransfer.files.length > 0) {
      processarArquivos(e.dataTransfer.files)
    }
  }, [processarArquivos])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processarArquivos(e.target.files)
    }
    // Limpa o input para permitir re-selecionar os mesmos arquivos
    if (inputRef.current) inputRef.current.value = ''
  }, [processarArquivos])

  const iniciarAnalise = useCallback(() => {
    if (arquivos.length === 0) return
    onAnalisar(
      arquivos.map((a) => a.arquivo),
      arquivos.map((a) => a.tipo),
    )
  }, [arquivos, onAnalisar])

  /* ─── Estado: analisando — oculta UploadZone ─── */
  if (carregando) {
    return null
  }

  /* ─── Estado: arquivos selecionados ─── */
  if (arquivos.length > 0) {
    const totalImagens = arquivos.filter((a) => a.tipo === 'imagem').length
    const totalVideos  = arquivos.filter((a) => a.tipo === 'video').length

    return (
      <div className="rounded-lg border border-[#3f3f46] bg-[#18181b] p-4 animate-fade-in space-y-3">
        {/* Grid de thumbnails */}
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2">
          {arquivos.map((item, i) => (
            <div
              key={`${item.arquivo.name}-${i}`}
              className="relative group rounded border border-[#27272a] bg-[#09090b] overflow-hidden aspect-square flex items-center justify-center"
            >
              {item.preview ? (
                <img src={item.preview} alt={item.arquivo.name} className="w-full h-full object-cover" />
              ) : (
                <Film size={24} className="text-[#52525b]" />
              )}

              {/* Overlay com nome */}
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-1.5 pt-4">
                <p className="text-[9px] text-[#a1a1aa] truncate" title={item.arquivo.name}>
                  {item.arquivo.name}
                </p>
                <span className="text-[8px] text-[#52525b] font-mono">{formatarTamanho(item.arquivo.size)}</span>
              </div>

              {/* Botão remover */}
              <button
                onClick={() => removerArquivo(i)}
                className="absolute top-1 right-1 p-0.5 rounded bg-black/60 text-[#71717a] hover:text-[#fafafa] opacity-0 group-hover:opacity-100 transition-opacity"
                aria-label={`Remover ${item.arquivo.name}`}
              >
                <X size={12} />
              </button>

              {/* Badge de tipo */}
              <span className={clsx(
                'absolute top-1 left-1 text-[8px] px-1 py-0.5 rounded font-semibold uppercase tracking-wider',
                item.tipo === 'imagem'
                  ? 'bg-[#1e1b4b] text-[#818cf8]'
                  : 'bg-[#134e4a] text-[#2dd4bf]',
              )}>
                {item.tipo === 'imagem' ? 'IMG' : 'VID'}
              </span>
            </div>
          ))}

          {/* Botão adicionar mais */}
          <label
            htmlFor="upload-arquivo-input-adicionar"
            className="rounded border-2 border-dashed border-[#27272a] hover:border-[#3f3f46] bg-[#09090b] aspect-square flex flex-col items-center justify-center cursor-pointer transition-colors group"
          >
            <Upload size={16} className="text-[#3f3f46] group-hover:text-[#52525b] transition-colors" />
            <span className="text-[9px] text-[#3f3f46] group-hover:text-[#52525b] mt-1">Adicionar</span>
            <input
              id="upload-arquivo-input-adicionar"
              type="file"
              multiple
              accept={[...EXTENSOES_IMAGEM, ...EXTENSOES_VIDEO].join(',')}
              onChange={handleChange}
              className="hidden"
            />
          </label>
        </div>

        {/* Resumo + ações */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-[10px] text-[#71717a]">
            {totalImagens > 0 && (
              <span className="flex items-center gap-1">
                <IconeImagem size={10} className="text-[#818cf8]" />
                {totalImagens} imagem{totalImagens !== 1 ? 's' : ''}
              </span>
            )}
            {totalVideos > 0 && (
              <span className="flex items-center gap-1">
                <Film size={10} className="text-[#2dd4bf]" />
                {totalVideos} vídeo{totalVideos !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={limparTodos}
              className="px-3 py-1.5 rounded text-[11px] text-[#71717a] hover:text-[#a1a1aa] hover:bg-[#27272a] transition-colors"
            >
              Limpar tudo
            </button>
            <button
              onClick={iniciarAnalise}
              className="py-1.5 px-4 rounded bg-[#6366f1] hover:bg-[#818cf8] text-white text-sm font-semibold transition-colors duration-150 flex items-center justify-center gap-2"
            >
              <Upload size={14} />
              {arquivos.length === 1
                ? `Analisar ${arquivos[0].tipo === 'imagem' ? 'Imagem' : 'Vídeo'}`
                : `Analisar ${arquivos.length} arquivos`
              }
            </button>
          </div>
        </div>

        {/* Erros */}
        {erro && (
          <div className="flex items-start gap-2 p-3 rounded border border-[#7f1d1d] bg-[#7f1d1d22] text-[#ef4444] text-xs">
            <AlertCircle size={12} className="flex-shrink-0 mt-0.5" />
            <span>{erro}</span>
          </div>
        )}
      </div>
    )
  }

  /* ─── Estado padrão: drop zone ─── */
  const idInput = 'upload-arquivo-input'

  return (
    <div className="space-y-2">
      <input
        id={idInput}
        ref={inputRef}
        type="file"
        multiple
        accept={[...EXTENSOES_IMAGEM, ...EXTENSOES_VIDEO].join(',')}
        onChange={handleChange}
        className="hidden"
      />

      <label
        htmlFor={idInput}
        onDragOver={(e) => { e.preventDefault(); setArrastando(true) }}
        onDragLeave={() => setArrastando(false)}
        onDrop={handleDrop}
        className={clsx(
          'relative block rounded-lg border-2 border-dashed transition-all duration-200 cursor-pointer group',
          arrastando
            ? 'border-[#6366f1] bg-[#1e1b4b]'
            : 'border-[#3f3f46] bg-[#18181b] hover:border-[#52525b] hover:bg-[#1f1f23]',
        )}
      >
        <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
          <div className={clsx(
            'w-12 h-12 rounded-lg border flex items-center justify-center mb-4 transition-all duration-200',
            arrastando
              ? 'border-[#6366f1] bg-[#1e1b4b] text-[#818cf8]'
              : 'border-[#27272a] bg-[#09090b] text-[#52525b] group-hover:border-[#3f3f46] group-hover:text-[#71717a]',
          )}>
            <Upload size={20} />
          </div>

          <p className="text-sm font-medium text-[#a1a1aa] mb-1">
            {arrastando ? 'Solte os arquivos aqui' : 'Arraste arquivos ou clique para selecionar'}
          </p>
          <p className="text-xs text-[#52525b]">
            Imagens: JPG, PNG, WebP &mdash; Vídeos: MP4, AVI, MOV, MKV
          </p>
          <p className="text-[10px] text-[#3f3f46] mt-1">
            Selecione múltiplos arquivos de uma vez
          </p>

          <div className="flex items-center gap-3 my-4 w-full max-w-xs">
            <div className="flex-1 h-px bg-[#27272a]" />
            <span className="text-[10px] text-[#3f3f46] uppercase tracking-widest">ou</span>
            <div className="flex-1 h-px bg-[#27272a]" />
          </div>

          <div className="flex gap-2">
            {[
              { icone: <IconeImagem size={11} />, label: 'Imagem' },
              { icone: <Film size={11} />, label: 'Vídeo' },
            ].map(({ icone, label }) => (
              <span key={label} className="flex items-center gap-1 text-[10px] text-[#52525b] bg-[#09090b] border border-[#27272a] rounded px-2 py-1">
                {icone}
                {label}
              </span>
            ))}
          </div>
        </div>

        {arrastando && (
          <div className="absolute inset-0 rounded-lg border-2 border-[#6366f1] pointer-events-none" />
        )}
      </label>

      {/* Mensagem de erro de formato */}
      {erro && (
        <div className="flex items-start gap-2 p-3 rounded border border-[#7f1d1d] bg-[#7f1d1d22] text-[#ef4444] text-xs">
          <AlertCircle size={12} className="flex-shrink-0 mt-0.5" />
          <span>{erro}</span>
        </div>
      )}
    </div>
  )
}
