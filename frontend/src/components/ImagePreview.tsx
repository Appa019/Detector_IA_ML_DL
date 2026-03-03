/**
 * ImagePreview — Preview da imagem original com zoom e informacoes do arquivo
 */

import { useState, useRef, useCallback } from 'react'
import { ZoomIn, ZoomOut, Maximize2, X, FileImage } from 'lucide-react'

interface ImagePreviewProps {
  arquivo?: File | null
  classificacao?: string
  cor?: string
}

function formatarTamanho(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function ImagePreview({ arquivo, classificacao, cor }: ImagePreviewProps) {
  const [zoom, setZoom] = useState(1)
  const [expandido, setExpandido] = useState(false)
  const [dimensoes, setDimensoes] = useState<{ w: number; h: number } | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [urlPreview] = useState(() => arquivo ? URL.createObjectURL(arquivo) : null)

  const handleImageLoad = useCallback(() => {
    if (imgRef.current) {
      setDimensoes({
        w: imgRef.current.naturalWidth,
        h: imgRef.current.naturalHeight,
      })
    }
  }, [])

  if (!arquivo || !urlPreview) return null

  const extensao = arquivo.name.split('.').pop()?.toUpperCase() ?? '?'

  return (
    <>
      <div className="space-y-2">
        {/* Info do arquivo */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-[#71717a]">
            <FileImage size={12} />
            <span className="font-mono truncate max-w-[200px]">{arquivo.name}</span>
          </div>
          <div className="flex items-center gap-3 text-[10px] text-[#52525b]">
            {dimensoes && (
              <span className="font-mono">{dimensoes.w}x{dimensoes.h}</span>
            )}
            <span className="font-mono">{extensao}</span>
            <span className="font-mono">{formatarTamanho(arquivo.size)}</span>
          </div>
        </div>

        {/* Imagem com controles de zoom */}
        <div className="rounded-lg border border-[#27272a] bg-[#09090b] p-2 relative group">
          <div className="overflow-hidden rounded flex items-center justify-center"
            style={{ maxHeight: '300px' }}
          >
            <img
              ref={imgRef}
              src={urlPreview}
              alt="Preview"
              onLoad={handleImageLoad}
              className="max-w-full max-h-[300px] object-contain transition-transform"
              style={{ transform: `scale(${zoom})` }}
            />
          </div>

          {/* Badge de classificacao */}
          {classificacao && (
            <div
              className="absolute top-4 left-4 px-2 py-0.5 rounded text-[10px] font-semibold border"
              style={{
                color: cor,
                borderColor: `${cor}44`,
                backgroundColor: `${cor}22`,
              }}
            >
              {classificacao}
            </div>
          )}

          {/* Controles de zoom */}
          <div className="absolute bottom-4 right-4 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
              className="w-6 h-6 rounded bg-[#18181b]/80 border border-[#27272a] flex items-center justify-center text-[#a1a1aa] hover:text-[#fafafa] transition-colors"
            >
              <ZoomOut size={12} />
            </button>
            <span className="text-[10px] font-mono text-[#71717a] px-1">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
              className="w-6 h-6 rounded bg-[#18181b]/80 border border-[#27272a] flex items-center justify-center text-[#a1a1aa] hover:text-[#fafafa] transition-colors"
            >
              <ZoomIn size={12} />
            </button>
            <button
              onClick={() => setExpandido(true)}
              className="w-6 h-6 rounded bg-[#18181b]/80 border border-[#27272a] flex items-center justify-center text-[#a1a1aa] hover:text-[#fafafa] transition-colors"
            >
              <Maximize2 size={12} />
            </button>
          </div>
        </div>
      </div>

      {/* Modal expandido */}
      {expandido && (
        <div
          className="fixed inset-0 z-50 bg-[#09090b]/90 flex items-center justify-center p-8"
          onClick={() => setExpandido(false)}
        >
          <button
            onClick={() => setExpandido(false)}
            className="absolute top-4 right-4 w-8 h-8 rounded-lg bg-[#18181b] border border-[#27272a] flex items-center justify-center text-[#a1a1aa] hover:text-[#fafafa] transition-colors"
          >
            <X size={16} />
          </button>
          <img
            src={urlPreview}
            alt="Preview expandido"
            className="max-w-full max-h-full object-contain rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  )
}
