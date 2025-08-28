import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export async function GET(
  request: Request,
  { params }: { params: { imageId: string } }
) {
  try {
    const projectRoot = path.join(process.cwd(), '..')
    const annotationPath = path.join(
      projectRoot,
      'annotations',
      'nsd',
      `${params.imageId}_annotations.json`
    )
    
    // Read the annotation file
    const fileContent = await fs.readFile(annotationPath, 'utf-8')
    const data = JSON.parse(fileContent)
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error loading annotation:', error)
    return NextResponse.json(
      { error: 'Annotation not found' },
      { status: 404 }
    )
  }
}