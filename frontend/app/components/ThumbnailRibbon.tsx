'use client'

import { useRef, useEffect } from 'react'
import { ImageData } from '../types'
import { ChevronLeft, ChevronRight } from 'lucide-react'

interface ThumbnailRibbonProps {
  images: ImageData[]
  selectedIndex: number
  onSelect: (index: number) => void
}

export default function ThumbnailRibbon({ images, selectedIndex, onSelect }: ThumbnailRibbonProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const thumbnailRefs = useRef<(HTMLButtonElement | null)[]>([])

  // Scroll to selected thumbnail when it changes
  useEffect(() => {
    if (thumbnailRefs.current[selectedIndex] && scrollContainerRef.current) {
      const thumbnail = thumbnailRefs.current[selectedIndex]
      const container = scrollContainerRef.current
      
      if (thumbnail) {
        const containerWidth = container.clientWidth
        const scrollLeft = thumbnail.offsetLeft - containerWidth / 2 + thumbnail.clientWidth / 2
        
        container.scrollTo({
          left: scrollLeft,
          behavior: 'smooth'
        })
      }
    }
  }, [selectedIndex])

  const scrollLeft = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollBy({
        left: -200,
        behavior: 'smooth'
      })
    }
  }

  const scrollRight = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollBy({
        left: 200,
        behavior: 'smooth'
      })
    }
  }

  const handleKeyNavigation = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft' && selectedIndex > 0) {
      onSelect(selectedIndex - 1)
    } else if (e.key === 'ArrowRight' && selectedIndex < images.length - 1) {
      onSelect(selectedIndex + 1)
    }
  }

  return (
    <div className="mt-4 bg-white rounded-lg shadow-md p-4">
      <div className="relative flex items-center">
        {/* Left scroll button */}
        <button
          onClick={scrollLeft}
          className="absolute left-0 z-10 bg-white/90 hover:bg-white rounded-full p-2 shadow-md"
          aria-label="Scroll left"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>

        {/* Thumbnail container */}
        <div
          ref={scrollContainerRef}
          className="flex gap-2 overflow-x-auto scrollbar-thin px-10 py-2"
          onKeyDown={handleKeyNavigation}
          tabIndex={0}
        >
          {images.map((image, index) => {
            // Extract the image number for display
            const imageNumber = image.id.match(/shared(\d+)/)?.[1] || String(index + 1)
            
            return (
              <button
                key={image.id}
                ref={el => thumbnailRefs.current[index] = el}
                onClick={() => onSelect(index)}
                className={`
                  relative flex-shrink-0 border-2 rounded-md overflow-hidden
                  transition-all duration-200 hover:scale-105
                  ${selectedIndex === index 
                    ? 'border-blue-500 shadow-lg' 
                    : 'border-gray-300 hover:border-gray-400'
                  }
                `}
                aria-label={`Select image ${imageNumber}`}
              >
                <img
                  src={image.thumbnailPath}
                  alt={`Thumbnail ${imageNumber}`}
                  className="w-24 h-24 object-cover"
                  loading="lazy"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs py-1 text-center">
                  {imageNumber}
                </div>
              </button>
            )
          })}
        </div>

        {/* Right scroll button */}
        <button
          onClick={scrollRight}
          className="absolute right-0 z-10 bg-white/90 hover:bg-white rounded-full p-2 shadow-md"
          aria-label="Scroll right"
        >
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
      
      {/* Image counter */}
      <div className="text-center mt-2 text-sm text-gray-600">
        Image {selectedIndex + 1} of {images.length}
      </div>
    </div>
  )
}