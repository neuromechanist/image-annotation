import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export async function GET() {
  try {
    // Get the project root (3 levels up from app/api/images)
    const projectRoot = path.join(process.cwd(), '..')
    const thumbnailDir = path.join(projectRoot, 'data', 'thumbnails')
    const annotationsDir = path.join(projectRoot, 'annotations', 'nsd')
    
    // Read thumbnail files
    const thumbnailFiles = await fs.readdir(thumbnailDir)
    const jpgFiles = thumbnailFiles.filter(f => f.endsWith('.jpg')).sort()
    
    // Create image data array
    const images = jpgFiles.map(filename => {
      const baseName = filename.replace('.jpg', '')
      return {
        id: baseName,
        thumbnailPath: `/data/thumbnails/${filename}`,
        imagePath: `/images/downsampled/${filename}`,
        annotationPath: `/annotations/nsd/${baseName}_annotations.json`
      }
    })
    
    return NextResponse.json(images)
  } catch (error) {
    console.error('Error loading images:', error)
    
    // Return a fallback list for the first 100 images
    const fallbackImages = []
    for (let i = 1; i <= 100; i++) {
      const paddedNum = String(i).padStart(4, '0')
      // Parse actual NSD numbers from known files
      const nsdNumbers: Record<string, string> = {
        '0001': '02951', '0002': '02991', '0003': '03050', '0004': '03078',
        '0005': '03147', '0006': '03158', '0007': '03165', '0008': '03172',
        '0009': '03182', '0010': '03387', // Add more as needed
      }
      
      const nsdNum = nsdNumbers[paddedNum] || String(2950 + i).padStart(5, '0')
      const imageName = `shared${paddedNum}_nsd${nsdNum}`
      
      fallbackImages.push({
        id: imageName,
        thumbnailPath: `/data/thumbnails/${imageName}.jpg`,
        imagePath: `/images/downsampled/${imageName}.jpg`,
        annotationPath: `/annotations/nsd/${imageName}_annotations.json`
      })
    }
    
    return NextResponse.json(fallbackImages)
  }
}