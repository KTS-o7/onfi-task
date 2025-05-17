"use client"

import { CheckCircle, Clock, Loader2 } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

export type ProcessingStep = {
  id: string
  label: string
  status: "pending" | "processing" | "completed"
}

interface ProcessingIndicatorProps {
  steps: ProcessingStep[]
  currentStep: number
  progress: number
}

export default function ProcessingIndicator({ steps, currentStep, progress }: ProcessingIndicatorProps) {
  return (
    <div className="w-full max-w-md mx-auto">
      <Progress value={progress} className="h-2 mb-6" />

      <div className="space-y-4">
        {steps.map((step, index) => {
          const isActive = index === currentStep
          const isCompleted = step.status === "completed"
          const isPending = step.status === "pending"

          return (
            <div
              key={step.id}
              className={cn(
                "flex items-center p-3 rounded-lg transition-colors",
                isActive ? "bg-primary/10" : "bg-transparent",
              )}
            >
              <div className="mr-4">
                {isCompleted ? (
                  <CheckCircle className="h-6 w-6 text-green-500" />
                ) : isActive ? (
                  <Loader2 className="h-6 w-6 text-primary animate-spin" />
                ) : (
                  <Clock className="h-6 w-6 text-muted-foreground" />
                )}
              </div>

              <div className="flex-1">
                <p
                  className={cn(
                    "font-medium",
                    isActive ? "text-primary" : isCompleted ? "text-foreground" : "text-muted-foreground",
                  )}
                >
                  {step.label}
                </p>
              </div>

              <div className="ml-2 text-xs">
                {isCompleted ? (
                  <span className="text-green-500">Completed</span>
                ) : isActive ? (
                  <span className="text-primary">In progress</span>
                ) : (
                  <span className="text-muted-foreground">Pending</span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
