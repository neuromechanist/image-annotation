'use client'

import { useState, useEffect, useMemo } from 'react'
import Image from 'next/image'
import ThumbnailRibbon from './components/ThumbnailRibbon'
import AnnotationViewer from './components/AnnotationViewer'
import { ImageData, Annotation } from './types'

export default function Dashboard() {
  const [images, setImages] = useState<ImageData[]>([])
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)
  const [annotations, setAnnotations] = useState<Record<string, Annotation[]>>({})
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [selectedPrompt, setSelectedPrompt] = useState<string>('')
  const [loading, setLoading] = useState(true)

  // Load image list
  useEffect(() => {
    async function loadImages() {
      try {
        // Get list of images from thumbnails directory
        const imageFiles: ImageData[] = []
        // For now, we'll hardcode the pattern, but in production this would come from an API
        for (let i = 1; i <= 100; i++) {
          const paddedNum = String(i).padStart(4, '0')
          const imageName = `shared${paddedNum}_nsd`
          // We'll need to check which files exist
          imageFiles.push({
            id: imageName,
            thumbnailPath: `/thumbnails/${imageName}`,
            imagePath: `/downsampled/${imageName}`,
            annotationPath: `/annotations/nsd/${imageName}_annotations.json`
          })
        }
        
        // Actually load the real file list
        const response = await fetch('/api/images')
        if (response.ok) {
          const data = await response.json()
          setImages(data)
        } else {
          // Fallback: create list from known pattern
          const tempImages = await loadImageList()
          setImages(tempImages)
        }
      } catch (error) {
        console.error('Error loading images:', error)
        // Use fallback method
        const tempImages = await loadImageList()
        setImages(tempImages)
      } finally {
        setLoading(false)
      }
    }

    loadImages()
  }, [])

  // Load annotations for selected image
  useEffect(() => {
    if (images.length > 0 && selectedImageIndex < images.length) {
      loadAnnotationsForImage(images[selectedImageIndex].id)
    }
  }, [selectedImageIndex, images])

  // Get available models and prompts for current image
  const availableModels = useMemo(() => {
    if (!images[selectedImageIndex]) return []
    const imageAnnotations = annotations[images[selectedImageIndex].id] || []
    return [...new Set(imageAnnotations.map(a => a.model))]
  }, [annotations, selectedImageIndex, images])

  const availablePrompts = useMemo(() => {
    if (!images[selectedImageIndex]) return []
    const imageAnnotations = annotations[images[selectedImageIndex].id] || []
    if (selectedModel) {
      return [...new Set(imageAnnotations.filter(a => a.model === selectedModel).map(a => a.prompt))]
    }
    return [...new Set(imageAnnotations.map(a => a.prompt))]
  }, [annotations, selectedImageIndex, images, selectedModel])

  // Get current annotation
  const currentAnnotation = useMemo(() => {
    if (!images[selectedImageIndex]) return null
    const imageAnnotations = annotations[images[selectedImageIndex].id] || []
    return imageAnnotations.find(a => 
      a.model === selectedModel && a.prompt === selectedPrompt
    ) || null
  }, [annotations, selectedImageIndex, images, selectedModel, selectedPrompt])

  async function loadImageList(): Promise<ImageData[]> {
    // Fallback method to create image list
    const imageList: ImageData[] = []
    for (let i = 1; i <= 100; i++) {
      const paddedNum = String(i).padStart(4, '0')
      // Parse the actual filenames from annotations
      try {
        const response = await fetch(`/annotations/nsd/shared${paddedNum}_nsd`, { method: 'HEAD' })
        if (response.ok) {
          // Extract actual NSD number from response or use pattern matching
        }
      } catch {}
    }
    return imageList
  }

  async function loadAnnotationsForImage(imageId: string) {
    try {
      // Try to load annotations from JSON file
      const response = await fetch(`/api/annotations/${imageId}`)
      if (response.ok) {
        const data = await response.json()
        setAnnotations(prev => ({ ...prev, [imageId]: data.annotations || [] }))
        
        // Set default model and prompt if sticky selections don't exist
        if (data.annotations && data.annotations.length > 0) {
          // Try to keep sticky selection
          const hasModel = data.annotations.some((a: Annotation) => a.model === selectedModel)
          const hasPrompt = data.annotations.some((a: Annotation) => a.prompt === selectedPrompt)
          
          if (!hasModel || !selectedModel) {
            setSelectedModel(data.annotations[0].model)
          }
          if (!hasPrompt || !selectedPrompt) {
            setSelectedPrompt(data.annotations[0].prompt)
          }
        }
      }
    } catch (error) {
      console.error('Error loading annotations:', error)
    }
  }

  const handleImageSelect = (index: number) => {
    setSelectedImageIndex(index)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl">Loading images...</div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b px-6 py-4">
        <h1 className="text-2xl font-semibold">HED Image Annotation Dashboard</h1>
      </header>
      
      <main className="flex-1 flex flex-col p-6 overflow-hidden">
        <div className="flex gap-6 flex-1 min-h-0">
          {/* Left side: Image viewer */}
          <div className="flex-1 bg-white rounded-lg shadow-md p-4">
            {images[selectedImageIndex] && (
              <div className="h-full flex items-center justify-center">
                <img
                  src={images[selectedImageIndex].imagePath}
                  alt={`NSD Image ${selectedImageIndex + 1}`}
                  className="max-w-full max-h-full object-contain"
                />
              </div>
            )}
          </div>

          {/* Right side: Controls and annotation viewer */}
          <div className="w-96 flex flex-col gap-4">
            {/* Dropdowns */}
            <div className="bg-white rounded-lg shadow-md p-4 space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select a model</option>
                  {availableModels.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Prompt
                </label>
                <select
                  value={selectedPrompt}
                  onChange={(e) => setSelectedPrompt(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select a prompt</option>
                  {availablePrompts.map(prompt => (
                    <option key={prompt} value={prompt}>
                      {prompt.length > 50 ? prompt.substring(0, 50) + '...' : prompt}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Annotation viewer */}
            <div className="flex-1 bg-white rounded-lg shadow-md p-4 overflow-auto">
              <h3 className="font-semibold mb-2">Response</h3>
              {currentAnnotation ? (
                <AnnotationViewer annotation={currentAnnotation} />
              ) : (
                <p className="text-gray-500">Select a model and prompt to view annotation</p>
              )}
            </div>
          </div>
        </div>

        {/* Thumbnail ribbon at bottom */}
        <ThumbnailRibbon
          images={images}
          selectedIndex={selectedImageIndex}
          onSelect={handleImageSelect}
        />
      </main>
    </div>
  )
}