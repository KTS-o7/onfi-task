import { Suspense } from "react"
import ComplianceChecker from "@/components/compliance-checker"
import { Skeleton } from "@/components/ui/skeleton"
import { FileCheck, ImportIcon } from "lucide-react"
import ChecklistGen from "@/components/checklist_gen"

export default function Home() {
  return (
    <main className="container mx-auto p-4 py-8">
      <div className="flex items-center justify-center mb-8">
        <div className="bg-primary/10 p-3 rounded-full mr-3">
          <FileCheck className="h-8 w-8 text-primary" />
        </div>
        <div>
          <h1 className="text-3xl font-bold">AMC SID Compliance Checker</h1>
          <p className="text-muted-foreground">Upload and analyze Scheme Information Documents for compliance</p>
        </div>
      </div>

      <Suspense fallback={<Skeleton className="h-[600px] w-full" />}>
        <ChecklistGen />
        <ComplianceChecker />

      </Suspense>
    </main>
  )
}
