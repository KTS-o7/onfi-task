"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Document, Page, } from "react-pdf"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"
import "react-pdf/dist/esm/Page/TextLayer.css"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent } from "@/components/ui/card"
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCw, Download, Loader2 } from "lucide-react"
import { Skeleton } from "@/components/ui/skeleton"

import { pdfjs } from "react-pdf";
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

interface PDFViewerProps {
  fileUrl: string
  pageNumber: number
  highlightText: string | null
  onPageChange?: (page: number) => void
}

export default function PDFViewer({ fileUrl, pageNumber, highlightText, onPageChange }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const [currentPage, setCurrentPage] = useState<number>(pageNumber || 1)
  const [scale, setScale] = useState<number>(1.2)
  const [rotation, setRotation] = useState<number>(0)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [jumpToPage, setJumpToPage] = useState<string>("")

  useEffect(() => {
    if (pageNumber && pageNumber !== currentPage) {
      setCurrentPage(pageNumber)
    }
  }, [pageNumber])

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
    setIsLoading(false)
  }

  const changePage = (offset: number) => {
    const newPage = currentPage + offset
    if (newPage >= 1 && newPage <= (numPages || 1)) {
      setCurrentPage(newPage)
      onPageChange?.(newPage)
    }
  }

  const handleJumpToPage = (e: React.FormEvent) => {
    e.preventDefault()
    const pageNum = Number.parseInt(jumpToPage, 10)
    if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= (numPages || 1)) {
      setCurrentPage(pageNum)
      onPageChange?.(pageNum)
    }
    setJumpToPage("")
  }

  const zoomIn = () => {
    setScale((prevScale) => Math.min(prevScale + 0.2, 3))
  }

  const zoomOut = () => {
    setScale((prevScale) => Math.max(prevScale - 0.2, 0.6))
  }

  const rotate = () => {
    setRotation((prevRotation) => (prevRotation + 90) % 360)
  }

  // Custom text renderer to highlight text
  const textRenderer = (textItem: any) => {
    if (highlightText && textItem.str.includes(highlightText)) {
      return <mark className="bg-yellow-300">{textItem.str}</mark>
    }
    return textItem.str
  }

  return (
    <Card className="h-full flex flex-col">
      <div className="flex justify-between items-center p-3 border-b">
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="icon" onClick={() => changePage(-1)} disabled={currentPage <= 1}>
            <ChevronLeft className="h-4 w-4" />
          </Button>

          <form onSubmit={handleJumpToPage} className="flex items-center">
            <Input
              type="text"
              value={jumpToPage}
              onChange={(e) => setJumpToPage(e.target.value)}
              className="w-14 h-8 text-center"
              placeholder={`${currentPage}`}
            />
            <span className="mx-1 text-sm text-muted-foreground">/ {numPages || "?"}</span>
          </form>

          <Button variant="outline" size="icon" onClick={() => changePage(1)} disabled={currentPage >= (numPages || 1)}>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center space-x-2">
          <Button variant="outline" size="icon" onClick={zoomOut} title="Zoom Out">
            <ZoomOut className="h-4 w-4" />
          </Button>

          <div className="w-24">
            <Slider
              value={[scale * 50]}
              min={30}
              max={150}
              step={5}
              onValueChange={(value) => setScale(value[0] / 50)}
            />
          </div>

          <Button variant="outline" size="icon" onClick={zoomIn} title="Zoom In">
            <ZoomIn className="h-4 w-4" />
          </Button>

          <Button variant="outline" size="icon" onClick={rotate} title="Rotate">
            <RotateCw className="h-4 w-4" />
          </Button>

          <Button variant="outline" size="icon" title="Download">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <CardContent className="flex-1 overflow-auto bg-gray-100 p-0 flex justify-center">
        <Document
          file={fileUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={
            <div className="flex items-center justify-center h-full w-full">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
                <p className="text-muted-foreground">Loading PDF document...</p>
              </div>
            </div>
          }
          error={
            <div className="p-8 text-center">
              <p className="text-red-500 font-medium mb-2">Failed to load PDF</p>
              <p className="text-muted-foreground text-sm">Please check if the file is a valid PDF document</p>
            </div>
          }
        >
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <Skeleton className="h-[600px] w-[450px]" />
            </div>
          ) : (
            <Page
              pageNumber={currentPage}
              scale={scale}
              rotate={rotation}
              customTextRenderer={textRenderer}
              renderTextLayer={true}
              renderAnnotationLayer={true}
              className="shadow-md"
            />
          )}
        </Document>
      </CardContent>
    </Card>
  )
}

