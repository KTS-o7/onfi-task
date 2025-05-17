"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import PDFUploader from "./pdf-uploader"
import ComplianceTable from "./compliance-table"
import PDFViewer from "./pdf-viewer"
import ProcessingIndicator, { type ProcessingStep } from "./processing-indicator"
import type { ComplianceData } from "@/types/compliance"
import { FileText, RefreshCw } from "lucide-react"
import { uploadSIDDocument, getTaskStatus, mapBackendToFrontendData, evaluateDocumentWithId } from "@/services/api"

export default function ComplianceChecker() {
  const [file, setFile] = useState<File | null>(null)
  const [fileUrl, setFileUrl] = useState<string | null>(null)
  const [complianceData, setComplianceData] = useState<ComplianceData[] | null>(null)
  const [showPdfViewer, setShowPdfViewer] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [highlightText, setHighlightText] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [progress, setProgress] = useState(0)
  const [selectedRow, setSelectedRow] = useState<number | null>(null)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [docId, setDocId] = useState<string | null>(null)

  const processingSteps: ProcessingStep[] = [
    { id: "upload", label: "Uploading document", status: "completed" },
    { id: "extract", label: "Extracting text content", status: "processing" },
    { id: "analyze", label: "Analyzing compliance requirements", status: "pending" },
    { id: "generate", label: "Generating compliance report", status: "pending" },
  ]

  const recomplianceSteps: ProcessingStep[] = [
    { id: "analyze", label: "Analyzing compliance requirements", status: "processing" },
    { id: "generate", label: "Generating compliance report", status: "pending" },
  ]

  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;

    if (isProcessing && taskId) {
      intervalId = setInterval(async () => {
        const { data, error } = await getTaskStatus(taskId);
        
        if (error) {
          setError(error);
          setIsProcessing(false);
          if (intervalId) clearInterval(intervalId);
          return;
        }

        if (data) {
          // Update progress based on task status
          if (data.progress !== undefined) {
            setProgress(data.progress * 100);
          }

          // Update processing step based on task status
          if (data.status === "processing") {
            // Determine current step based on progress
            const steps = docId ? recomplianceSteps : processingSteps;
            const newStep = Math.floor((data.progress || 0) * steps.length);
            if (newStep !== currentStep && newStep < steps.length) {
              setCurrentStep(newStep);
              
              // Update step statuses
              steps.forEach((step, idx) => {
                if (idx < newStep) step.status = "completed";
                else if (idx === newStep) step.status = "processing";
                else step.status = "pending";
              });
            }
          }

          // Process completed task
          if (data.status === "completed" && data.result) {
            setIsProcessing(false);
            setProgress(100);
            
            // Store document ID for re-evaluation
            if (data.result.doc_id && !docId) {
              setDocId(data.result.doc_id);
            }
            
            // Map backend data to frontend format
            if (data.result.report_items) {
              const mappedData = await mapBackendToFrontendData(data.result.report_items);
              setComplianceData(mappedData);
            }
            
            if (intervalId) clearInterval(intervalId);
          }

          // Handle failed task
          if (data.status === "failed") {
            setError(data.error || "Processing failed");
            setIsProcessing(false);
            if (intervalId) clearInterval(intervalId);
          }
        }
      }, 5000); // Check every 5 seconds
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isProcessing, taskId, currentStep, docId]);

  const handleFileUpload = async (uploadedFile: File) => {
    setFile(uploadedFile)
    setFileUrl(URL.createObjectURL(uploadedFile))
    setComplianceData(null)
    setShowPdfViewer(false)
    setSelectedRow(null)
    setIsProcessing(true)
    setCurrentStep(0)
    setProgress(25)
    setError(null)
    setDocId(null)

    // Reset processing steps
    processingSteps.forEach((step, index) => {
      step.status = index === 0 ? "completed" : index === 1 ? "processing" : "pending"
    })

    // Upload file to backend
    const { data, error } = await uploadSIDDocument(uploadedFile);
    
    if (error) {
      setError(error);
      setIsProcessing(false);
      return;
    }

    if (data?.task_id) {
      setTaskId(data.task_id);
    } else {
      setError("No task ID received from server");
      setIsProcessing(false);
    }
  }

  const handleRerunCompliance = async () => {
    if (!docId) {
      setError("Document ID not available for re-evaluation");
      return;
    }

    setIsProcessing(true);
    setCurrentStep(0);
    setProgress(25);
    setError(null);
    
    // Reset re-compliance steps
    recomplianceSteps.forEach((step, index) => {
      step.status = index === 0 ? "processing" : "pending";
    });

    // Call API to evaluate document without re-processing
    const { data, error } = await evaluateDocumentWithId(docId);
    
    if (error) {
      setError(error);
      setIsProcessing(false);
      return;
    }

    if (data?.task_id) {
      setTaskId(data.task_id);
    } else {
      setError("No task ID received from server");
      setIsProcessing(false);
    }
  }

  const handleRowClick = (index: number) => {
    setSelectedRow(index)
    setShowPdfViewer(true)
  }

  const handlePageClick = (pageNumber: number, text?: string) => {
    setCurrentPage(pageNumber)
    setHighlightText(text || null)
    setShowPdfViewer(true)
  }

  return (
    <div className="flex flex-col space-y-4">
      {!file && <PDFUploader onFileUpload={handleFileUpload} title= "Upload SID Document" description= "Upload a SID Document to check compliance" buttonText= "Select SID Document" />}

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
            <h2 className="text-xl font-semibold mb-4">
              {docId ? "Re-analyzing Compliance" : "Processing Document"}
            </h2>
            <ProcessingIndicator 
              steps={docId ? recomplianceSteps : processingSteps} 
              currentStep={currentStep} 
              progress={progress} 
            />
          </CardContent>
        </Card>
      )}

      {file && !isProcessing && complianceData && (
        <>
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">
              <span className="text-primary">Analysis Results:</span> {file.name}
            </h2>
            <div className="flex space-x-2">
              {docId && (
                <Button variant="outline" onClick={handleRerunCompliance}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Re-analyze Compliance
                </Button>
              )}
              {showPdfViewer && (
                <Button variant="outline" onClick={() => setShowPdfViewer(false)}>
                  <FileText className="h-4 w-4 mr-2" />
                  Hide PDF
                </Button>
              )}
              <Button
                variant="outline"
                onClick={() => {
                  setFile(null)
                  setFileUrl(null)
                  setComplianceData(null)
                  setShowPdfViewer(false)
                  setSelectedRow(null)
                  setTaskId(null)
                  setDocId(null)
                }}
              >
                Upload New File
              </Button>
            </div>
          </div>

          <div className={`grid gap-4 ${
            showPdfViewer 
              ? "grid-cols-1 lg:grid-cols-2" 
              : "grid-cols-1"
          }`}>
            <div className={`overflow-auto ${
              showPdfViewer 
                ? "" 
                : "w-full"
            }`}>
              <ComplianceTable
                data={complianceData}
                isLoading={false}
                onPageClick={handlePageClick}
                onRowClick={handleRowClick}
                selectedRow={selectedRow}
              />
            </div>
            {showPdfViewer && fileUrl && (
              <div className="h-[calc(100vh-250px)] border rounded-lg overflow-hidden">
                <PDFViewer 
                  fileUrl={fileUrl} 
                  pageNumber={currentPage} 
                  highlightText={highlightText}
                  onPageChange={setCurrentPage}
                />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
