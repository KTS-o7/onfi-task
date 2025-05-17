"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import PDFUploader from "./pdf-uploader"
import ProcessingIndicator, { type ProcessingStep } from "./processing-indicator"
import { FileText, RefreshCw } from "lucide-react"
import { uploadMasterCircular, getTaskStatus } from "@/services/api"

export default function ChecklistGen() {
  const [file, setFile] = useState<File | null>(null)
  const [fileUrl, setFileUrl] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [progress, setProgress] = useState(0)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [checklist, setChecklist] = useState<any[] | null>(null)

  const processingSteps: ProcessingStep[] = [
    { id: "upload", label: "Uploading Master Circular", status: "completed" },
    { id: "extract", label: "Extracting text content", status: "processing" },
    { id: "analyze", label: "Analyzing Master Circular structure", status: "pending" },
    { id: "generate", label: "Generating compliance checklist", status: "pending" },
  ]

  const handleFileUpload = async (uploadedFile: File) => {
    setFile(uploadedFile)
    setFileUrl(URL.createObjectURL(uploadedFile))
    setChecklist(null)
    setIsProcessing(true)
    setCurrentStep(0)
    setProgress(25)
    setError(null)

    // Reset processing steps
    processingSteps.forEach((step, index) => {
      step.status = index === 0 ? "completed" : index === 1 ? "processing" : "pending"
    })

    // Upload file to backend using the master circular endpoint
    const { data, error } = await uploadMasterCircular(uploadedFile)
    
    if (error) {
      setError(error)
      setIsProcessing(false)
      return
    }

    if (data?.task_id) {
      setTaskId(data.task_id)
    } else {
      setError("No task ID received from server")
      setIsProcessing(false)
    }
  }

  // Poll for task status
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null

    if (isProcessing && taskId) {
      intervalId = setInterval(async () => {
        const { data, error } = await getTaskStatus(taskId)
        
        if (error) {
          setError(error)
          setIsProcessing(false)
          if (intervalId) clearInterval(intervalId)
          return
        }

        if (data) {
          // Update progress based on task status
          if (data.progress !== undefined) {
            setProgress(data.progress * 100)
          }

          // Update processing step based on task status
          if (data.status === "processing") {
            const newStep = Math.floor((data.progress || 0) * processingSteps.length)
            if (newStep !== currentStep && newStep < processingSteps.length) {
              setCurrentStep(newStep)
              
              // Update step statuses
              processingSteps.forEach((step, idx) => {
                if (idx < newStep) step.status = "completed"
                else if (idx === newStep) step.status = "processing"
                else step.status = "pending"
              })
            }
          }

          // Process completed task
          if (data.status === "completed" && data.result) {
            setIsProcessing(false)
            setProgress(100)
            
            // Set the generated checklist
            if (data.result.checklist) {
              setChecklist(data.result.checklist)
            }
            
            if (intervalId) clearInterval(intervalId)
          }

          // Handle failed task
          if (data.status === "failed") {
            setError(data.error || "Processing failed")
            setIsProcessing(false)
            if (intervalId) clearInterval(intervalId)
          }
        }
      }, 2000) // Check every 2 seconds
    }

    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [isProcessing, taskId, currentStep])

  return (
    <div className="flex flex-col space-y-4">
      {!file && (
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Upload Master Circular</h2>
          <p className="text-muted-foreground mb-4">Upload a Master Circular document to generate a compliance checklist</p>
          <PDFUploader onFileUpload={handleFileUpload} title="Upload Master Circular" description="Upload a Master Circular document to generate a compliance checklist" buttonText="Select Master Circular" />
        </div>
      )}

      {error && (
        <Card className="border-red-500">
          <CardContent className="pt-6">
            <h2 className="text-xl font-semibold text-red-500 mb-2">Error</h2>
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      {file && isProcessing && (
        <Card>
          <CardContent className="pt-6">
            <h2 className="text-xl font-semibold mb-4">Processing Master Circular</h2>
            <ProcessingIndicator 
              steps={processingSteps} 
              currentStep={currentStep} 
              progress={progress} 
            />
          </CardContent>
        </Card>
      )}

      {file && !isProcessing && checklist && (
        <>
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">
              <span className="text-primary">Master Circular Checklist:</span> {file.name}
            </h2>
            <Button
              variant="outline"
              onClick={() => {
                setFile(null)
                setFileUrl(null)
                setChecklist(null)
                setTaskId(null)
              }}
            >
              Upload New Master Circular
            </Button>
          </div>

          <Card>
            <CardContent className="pt-6">
              <div className="space-y-4">
                {checklist.map((item, index) => (
                  <div key={index} className="flex items-start space-x-4 p-4 border rounded-lg">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium">{item.title}</h3>
                      {item.description && (
                        <p className="text-muted-foreground mt-1">{item.description}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
