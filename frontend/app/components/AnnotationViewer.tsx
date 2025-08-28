'use client'

import { useState } from 'react'
import { PromptAnnotation } from '../types'
import { Copy, Check, FileText, FileJson, Activity, Zap } from 'lucide-react'

interface AnnotationViewerProps {
  annotation: PromptAnnotation
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

  // Check if we have structured JSON data
  const hasJsonData = annotation.response_data && Object.keys(annotation.response_data).length > 0

  // Render JSON data in a structured way
  const renderJsonData = (data: any) => {
    if (!data) return null

    // If it's an array of items
    if (Array.isArray(data)) {
      return (
        <div className="space-y-3">
          {data.map((item, index) => (
            <div key={index} className="bg-gray-800/30 rounded-lg p-3 border border-purple-500/10">
              <div className="text-xs text-purple-400 mb-2">Item {index + 1}</div>
              {renderJsonData(item)}
            </div>
          ))}
        </div>
      )
    }

    // If it's an object with properties
    if (typeof data === 'object') {
      return (
        <div className="space-y-2">
          {Object.entries(data).map(([key, value]) => (
            <div key={key} className="flex flex-col gap-1">
              <span className="text-xs text-purple-400 font-medium">
                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
              </span>
              {typeof value === 'object' ? (
                <div className="ml-3 border-l-2 border-purple-500/20 pl-3">
                  {renderJsonData(value)}
                </div>
              ) : (
                <span className="text-sm text-gray-300 ml-3">
                  {String(value)}
                </span>
              )}
            </div>
          ))}
        </div>
      )
    }

    // Primitive value
    return <span className="text-sm text-gray-300">{String(data)}</span>
  }

  return (
    <div className="space-y-4">
      {/* Prompt Text */}
      <div className="bg-gray-800/30 rounded-lg p-3 border border-purple-500/10">
        <div className="text-xs text-purple-400 mb-1">Prompt</div>
        <div className="text-sm text-gray-300 leading-relaxed">
          {annotation.prompt_text}
        </div>
      </div>

      {/* View mode toggle */}
      {hasJsonData && (
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('text')}
            className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition-all ${
              viewMode === 'text'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                : 'bg-gray-800/30 text-gray-400 hover:bg-gray-800/50 border border-gray-700/30'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            Text
          </button>
          <button
            onClick={() => setViewMode('json')}
            className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition-all ${
              viewMode === 'json'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                : 'bg-gray-800/30 text-gray-400 hover:bg-gray-800/50 border border-gray-700/30'
            }`}
          >
            <FileJson className="w-3.5 h-3.5" />
            Structured
          </button>
        </div>
      )}

      {/* Content area */}
      <div className="relative">
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-2 rounded-lg bg-gray-800/50 hover:bg-gray-800/70 transition-all z-10"
          aria-label="Copy to clipboard"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-400" />
          ) : (
            <Copy className="w-4 h-4 text-gray-400" />
          )}
        </button>

        <div className="bg-gray-800/30 rounded-lg p-4 pr-12 max-h-[400px] overflow-y-auto border border-purple-500/10">
          {viewMode === 'text' ? (
            <div className="whitespace-pre-wrap text-sm text-gray-300 leading-relaxed">
              {annotation.response}
            </div>
          ) : hasJsonData ? (
            <div className="text-sm">
              {renderJsonData(annotation.response_data)}
            </div>
          ) : (
            <pre className="text-xs overflow-x-auto text-gray-300">
              <code>{JSON.stringify(annotation.response, null, 2)}</code>
            </pre>
          )}
        </div>
      </div>

      {/* Metrics */}
      {(annotation.token_metrics || annotation.performance_metrics) && (
        <div className="grid grid-cols-2 gap-3">
          {annotation.token_metrics && (
            <div className="bg-gray-800/30 rounded-lg p-3 border border-purple-500/10">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-purple-400" />
                <span className="text-xs font-medium text-gray-400">Token Usage</span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Input:</span>
                  <span className="text-gray-300">{annotation.token_metrics.input_tokens}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Output:</span>
                  <span className="text-gray-300">{annotation.token_metrics.output_tokens}</span>
                </div>
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-gray-500">Total:</span>
                  <span className="text-purple-300">{annotation.token_metrics.total_tokens}</span>
                </div>
              </div>
            </div>
          )}

          {annotation.performance_metrics && (
            <div className="bg-gray-800/30 rounded-lg p-3 border border-purple-500/10">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-purple-400" />
                <span className="text-xs font-medium text-gray-400">Performance</span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Speed:</span>
                  <span className="text-gray-300">
                    {annotation.performance_metrics.tokens_per_second.toFixed(1)} t/s
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Time:</span>
                  <span className="text-gray-300">
                    {(annotation.performance_metrics.total_duration_ms / 1000).toFixed(2)}s
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}