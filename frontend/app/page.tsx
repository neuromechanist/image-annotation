'use client'

import { useState, useEffect, useMemo } from 'react'
import Image from 'next/image'
import ThumbnailRibbon from './components/ThumbnailRibbon'
import AnnotationViewer from './components/AnnotationViewer'
import { ImageData, Annotation, PromptAnnotation } from './types'
import { Brain, Sparkles, ChevronDown, Loader2, ExternalLink } from 'lucide-react'

export default function Dashboard() {
  const [images, setImages] = useState<ImageData[]>([])
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)
  const [annotations, setAnnotations] = useState<Record<string, Annotation[]>>({})
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [selectedPromptKey, setSelectedPromptKey] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [imageLoading, setImageLoading] = useState(false)

  // Load image list
  useEffect(() => {
    async function loadImageList() {
      try {
        const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ''
        const response = await fetch(`${basePath}/image-list.json`)
        if (response.ok) {
          const data = await response.json()
          const imageList: ImageData[] = data.images.map((imageName: string) => ({
            id: imageName,
            thumbnailPath: `${basePath}/thumbnails/${imageName}.jpg`,
            imagePath: `${basePath}/downsampled/${imageName}.jpg`,
            annotationPath: `${basePath}/annotations/nsd/${imageName}_annotations.json`
          }))
          setImages(imageList)
        } else {
          console.error('Failed to load image list')
        }
      } catch (error) {
        console.error('Error loading image list:', error)
      } finally {
        setLoading(false)
      }
    }
    
    loadImageList()
  }, [])

  // Load annotations for selected image
  useEffect(() => {
    if (images.length > 0 && selectedImageIndex < images.length) {
      loadAnnotationsForImage(images[selectedImageIndex].id)
    }
  }, [selectedImageIndex, images])

  // Get available models from current image annotations
  const availableModels = useMemo(() => {
    if (!images[selectedImageIndex]) return []
    const imageAnnotations = annotations[images[selectedImageIndex].id] || []
    return Array.from(new Set(imageAnnotations.map(a => a.model)))
  }, [annotations, selectedImageIndex, images])

  // Get available prompt keys for selected model
  const availablePromptKeys = useMemo(() => {
    if (!images[selectedImageIndex] || !selectedModel) return []
    const imageAnnotations = annotations[images[selectedImageIndex].id] || []
    const modelAnnotation = imageAnnotations.find(a => a.model === selectedModel)
    if (modelAnnotation && modelAnnotation.prompts) {
      return Object.keys(modelAnnotation.prompts)
    }
    return []
  }, [annotations, selectedImageIndex, images, selectedModel])

  // Get current prompt annotation
  const currentPromptAnnotation = useMemo(() => {
    if (!images[selectedImageIndex] || !selectedModel || !selectedPromptKey) return null
    const imageAnnotations = annotations[images[selectedImageIndex].id] || []
    const modelAnnotation = imageAnnotations.find(a => a.model === selectedModel)
    return modelAnnotation?.prompts[selectedPromptKey] || null
  }, [annotations, selectedImageIndex, images, selectedModel, selectedPromptKey])

  async function loadAnnotationsForImage(imageId: string) {
    setImageLoading(true)
    try {
      const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ''
      const response = await fetch(`${basePath}/annotations/nsd/${imageId}_annotations.json`)
      if (response.ok) {
        const data = await response.json()
        setAnnotations(prev => ({ ...prev, [imageId]: data.annotations || [] }))
        
        // Auto-select first model and prompt if nothing selected
        if (data.annotations && data.annotations.length > 0) {
          const firstAnnotation = data.annotations[0]
          
          // Keep sticky selection or set new defaults
          const hasModel = data.annotations.some((a: Annotation) => a.model === selectedModel)
          if (!hasModel || !selectedModel) {
            setSelectedModel(firstAnnotation.model)
            const firstPromptKey = Object.keys(firstAnnotation.prompts)[0]
            setSelectedPromptKey(firstPromptKey)
          } else {
            // Check if current prompt key exists in new model
            const modelAnnotation = data.annotations.find((a: Annotation) => a.model === selectedModel)
            if (modelAnnotation && modelAnnotation.prompts) {
              const hasPromptKey = Object.keys(modelAnnotation.prompts).includes(selectedPromptKey)
              if (!hasPromptKey) {
                const firstPromptKey = Object.keys(modelAnnotation.prompts)[0]
                setSelectedPromptKey(firstPromptKey)
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error loading annotations:', error)
    } finally {
      setImageLoading(false)
    }
  }

  const handleImageSelect = (index: number) => {
    setSelectedImageIndex(index)
  }

  // Format prompt key for display
  const formatPromptKey = (key: string) => {
    return key.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ')
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-purple-400 animate-spin mx-auto mb-4" />
          <div className="text-xl text-gray-200">Loading neural interface...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 flex flex-col">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-xl border-b border-purple-500/20">
        <div className="px-3 md:px-6 py-3 md:py-4 flex flex-col md:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2 md:gap-3">
            <div className="p-1.5 md:p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
              <Brain className="w-5 h-5 md:w-6 md:h-6 text-white" />
            </div>
            <h1 className="text-lg md:text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              The Annotation Garden Project
            </h1>
          </div>
          <div className="flex items-center gap-2 text-xs md:text-sm text-gray-400">
            <Sparkles className="w-3 h-3 md:w-4 md:h-4 text-purple-400" />
            <span>AI-Powered Vision Analysis</span>
          </div>
        </div>
      </header>
      
      <main className="flex-1 flex flex-col">
        <div className="flex-1 p-3 md:p-6 flex flex-col lg:flex-row gap-4 md:gap-6 min-h-0">
          {/* Image Viewer - Full width on mobile, constrained on desktop */}
          <div className="flex flex-col gap-4 lg:max-w-[600px] lg:min-w-[400px] w-full">
            <div className="relative bg-black/40 backdrop-blur-md rounded-2xl border border-purple-500/20 p-2 h-[50vh] md:h-full md:max-h-[600px] flex items-center justify-center">
              {imageLoading && (
                <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-10 flex items-center justify-center rounded-2xl">
                  <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
                </div>
              )}
              {images[selectedImageIndex] && (
                <div className="flex items-center justify-center">
                  <img
                    src={images[selectedImageIndex].imagePath}
                    alt={`NSD Image ${selectedImageIndex + 1}`}
                    className="max-w-full max-h-[580px] object-contain rounded-lg"
                  />
                </div>
              )}
            </div>

            {/* Image Info Bar */}
            <div className="bg-black/40 backdrop-blur-md rounded-xl border border-purple-500/20 px-4 py-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-400">Image ID:</span>
                  <span className="text-sm font-mono text-purple-300">
                    {images[selectedImageIndex]?.id || 'Loading...'}
                  </span>
                </div>
                <div className="text-sm text-gray-400">
                  {selectedImageIndex + 1} / {images.length}
                </div>
              </div>
            </div>
          </div>

          {/* Controls and Annotations - Full width on mobile, side panel on desktop */}
          <div className="flex-1 flex flex-col gap-4 min-w-0 w-full lg:w-auto">
            {/* Model Selection */}
            <div className="bg-black/40 backdrop-blur-md rounded-xl border border-purple-500/20 p-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Vision Model
              </label>
              <div className="relative">
                <select
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value)
                    // Reset prompt selection when model changes
                    setSelectedPromptKey('')
                  }}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-purple-500/30 rounded-lg text-gray-200 appearance-none focus:outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-400/20 transition-all"
                >
                  <option value="">Select a model</option>
                  {availableModels.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
              </div>
            </div>

            {/* Prompt Selection */}
            <div className="bg-black/40 backdrop-blur-md rounded-xl border border-purple-500/20 p-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Analysis Type
              </label>
              <div className="relative">
                <select
                  value={selectedPromptKey}
                  onChange={(e) => setSelectedPromptKey(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-purple-500/30 rounded-lg text-gray-200 appearance-none focus:outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-400/20 transition-all"
                  disabled={!selectedModel}
                >
                  <option value="">Select analysis type</option>
                  {availablePromptKeys.map(key => (
                    <option key={key} value={key}>
                      {formatPromptKey(key)}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
              </div>
            </div>

            {/* Annotation Display - Fixed height on mobile, flexible on desktop */}
            <div className="h-[40vh] lg:h-auto lg:flex-1 bg-black/40 backdrop-blur-md rounded-xl border border-purple-500/20 p-4 overflow-hidden flex flex-col">
              <h3 className="font-semibold text-gray-200 mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-purple-400" />
                AI Analysis
              </h3>
              <div className="flex-1 overflow-auto">
                {currentPromptAnnotation ? (
                  <AnnotationViewer annotation={currentPromptAnnotation} />
                ) : (
                  <div className="text-gray-500 text-center py-8">
                    {!selectedModel ? 'Select a vision model to begin' : 
                     !selectedPromptKey ? 'Choose an analysis type' :
                     'No analysis available'}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Thumbnail Ribbon */}
        <div className="px-3 md:px-6 pb-3">
          <ThumbnailRibbon
            images={images}
            selectedIndex={selectedImageIndex}
            onSelect={handleImageSelect}
          />
        </div>

        {/* Footer */}
        <footer className="bg-black/30 backdrop-blur-xl border-t border-purple-500/20 px-6 py-3">
          <div className="text-center text-sm text-gray-400">
            © 2025{' '}
            <a 
              href="https://neuromechanist.github.io" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-purple-400 hover:text-purple-300 transition-colors inline-flex items-center gap-1"
            >
              Seyed Yahya Shirazi
              <ExternalLink className="w-3 h-3" />
            </a>
            , Swartz Center for Computational Neuroscience, UC San Diego
          </div>
        </footer>
      </main>
    </div>
  )
}