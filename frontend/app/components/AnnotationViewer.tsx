'use client'

import { useState } from 'react'
import { Annotation } from '../types'
import { Copy, Check } from 'lucide-react'

interface AnnotationViewerProps {
  annotation: Annotation
}

export default function AnnotationViewer({ annotation }: AnnotationViewerProps) {
  const [copied, setCopied] = useState(false)
  const [viewMode, setViewMode] = useState<'text' | 'json'>('text')

  const handleCopy = () => {
    const textToCopy = viewMode === 'json' 
      ? JSON.stringify(annotation.response_data || annotation.response, null, 2)
      : annotation.response
    
    navigator.clipboard.writeText(textToCopy)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-3">
      {/* View mode toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setViewMode('text')}
          className={`px-3 py-1 rounded text-sm ${
            viewMode === 'text'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          Text
        </button>
        {annotation.response_data && (
          <button
            onClick={() => setViewMode('json')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'json'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            JSON
          </button>
        )}
      </div>

      {/* Content area */}
      <div className="relative">
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-2 rounded hover:bg-gray-100"
          aria-label="Copy to clipboard"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-500" />
          ) : (
            <Copy className="w-4 h-4 text-gray-500" />
          )}
        </button>

        <div className="bg-gray-50 rounded-lg p-4 pr-12 max-h-96 overflow-y-auto">
          {viewMode === 'text' ? (
            <div className="whitespace-pre-wrap text-sm">{annotation.response}</div>
          ) : (
            <pre className="text-xs overflow-x-auto">
              <code>{JSON.stringify(annotation.response_data || annotation.response, null, 2)}</code>
            </pre>
          )}
        </div>
      </div>

      {/* Metadata */}
      {annotation.metadata && (
        <div className="text-xs text-gray-500 space-y-1">
          {annotation.metadata.processing_time && (
            <div>Processing time: {annotation.metadata.processing_time.toFixed(2)}s</div>
          )}
          {annotation.metadata.temperature !== undefined && (
            <div>Temperature: {annotation.metadata.temperature}</div>
          )}
          {annotation.metadata.timestamp && (
            <div>Generated: {new Date(annotation.metadata.timestamp).toLocaleString()}</div>
          )}
        </div>
      )}
    </div>
  )
}