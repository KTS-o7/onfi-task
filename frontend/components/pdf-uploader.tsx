"use client"

import type React from "react"
import { useState, useRef } from "react"
import { FileUp, Upload, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

interface PDFUploaderProps {
  onFileUpload: (file: File) => void
  title?: string
  description?: string
  buttonText?: string
}

export default function PDFUploader({ 
  onFileUpload, 
  title = "Upload Document",
  description = "Drag and drop your PDF file here, or click to browse",
  buttonText = "Select PDF File"
}: PDFUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0]
      if (file.type === "application/pdf") {
        setSelectedFile(file)
      } else {
        alert("Please upload a PDF file")
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0]
      if (file.type === "application/pdf") {
        setSelectedFile(file)
      } else {
        alert("Please upload a PDF file")
      }
    }
  }

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  const handleRemoveFile = () => {
    setSelectedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleUpload = () => {
    if (selectedFile) {
      onFileUpload(selectedFile)
    }
  }

  return (
    <Card className="w-full">
      <CardContent className="pt-6">
        <div
          className={`flex flex-col items-center justify-center p-10 border-2 border-dashed rounded-lg transition-all ${
            isDragging ? "border-primary bg-primary/5" : "border-gray-300"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {!selectedFile ? (
            <>
              <div className="w-16 h-16 mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <Upload className="w-8 h-8 text-primary" />
              </div>
              <h3 className="mb-2 text-lg font-semibold">{title}</h3>
              <p className="mb-6 text-sm text-muted-foreground text-center">
                {description}
              </p>
              <Button onClick={handleButtonClick} className="relative">
                <FileUp className="w-4 h-4 mr-2" />
                {buttonText}
              </Button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="application/pdf"
                className="hidden"
              />
            </>
          ) : (
            <div className="w-full">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center mr-3">
                    <FileUp className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium truncate max-w-[200px]">{selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
                <Button variant="ghost" size="icon" onClick={handleRemoveFile}>
                  <X className="w-4 h-4" />
                </Button>
              </div>
              <Button onClick={handleUpload} className="w-full">
                Process Document
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
