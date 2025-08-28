export interface ImageData {
  id: string
  thumbnailPath: string
  imagePath: string
  annotationPath: string
}

export interface Annotation {
  model: string
  prompt: string
  response: string
  response_data?: any
  metadata?: {
    processing_time?: number
    temperature?: number
    timestamp?: string
    [key: string]: any
  }
}

export interface AnnotationFile {
  image_id: string
  image_path: string
  annotations: Annotation[]
  metadata?: {
    processed_at?: string
    [key: string]: any
  }
}