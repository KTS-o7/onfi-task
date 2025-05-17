"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Search, X } from "lucide-react"
import type { ComplianceData } from "@/types/compliance"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { motion } from "framer-motion"

interface ComplianceTableProps {
  data: ComplianceData[] | null
  isLoading: boolean
  onPageClick: (pageNumber: number, text?: string) => void
  onRowClick: (index: number) => void
  selectedRow: number | null
}

export default function ComplianceTable({
  data,
  isLoading,
  onPageClick,
  onRowClick,
  selectedRow,
}: ComplianceTableProps) {
  const [searchTerm, setSearchTerm] = useState("")

  const filteredData = data?.filter(
    (item) =>
      item.requirement.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.source.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.status.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.summary.toLowerCase().includes(searchTerm.toLowerCase()),
  )


  const handleCitationClick = (pageNumber: number) => {
    onPageClick(pageNumber)
  }

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "compliant":
        return <Badge className="bg-green-500 hover:bg-green-600">Compliant</Badge>
      case "non-compliant":
        return <Badge className="bg-red-500 hover:bg-red-600">Non-Compliant</Badge>
      case "in progress":
        return <Badge className="bg-yellow-500 hover:bg-yellow-600">In Progress</Badge>
      default:
        return <Badge>{status}</Badge>
    }
  }

  console.log("Compliance data:", data);

  return (
    <Card className="overflow-hidden">
      <div className="p-4 border-b">
        <div className="flex items-center space-x-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500" />
            <Input
              placeholder="Search compliance data..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          {searchTerm && (
            <Button variant="ghost" size="icon" onClick={() => setSearchTerm("")}>
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      <div className="max-h-[calc(100vh-350px)] overflow-auto">
        <Table>
          <TableHeader className="sticky top-0 bg-background z-10">
            <TableRow>
              <TableHead className="w-[250px] whitespace-normal">Compliance Requirement</TableHead>
              <TableHead className="w-[200px] whitespace-normal">Source</TableHead>
              <TableHead className="w-[300px] whitespace-normal">Rationale</TableHead>
              <TableHead className="w-[150px] whitespace-normal">Status</TableHead>
              <TableHead className="w-[250px] whitespace-normal">Findings Summary</TableHead>
              <TableHead className="w-[200px] whitespace-normal">Findings Citations</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, index) => (
                <TableRow key={index}>
                  {Array.from({ length: 6 }).map((_, cellIndex) => (
                    <TableCell key={cellIndex}>
                      <Skeleton className="h-6 w-full" />
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : filteredData && filteredData.length > 0 ? (
              filteredData.map((item, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className={`border-b transition-colors hover:bg-muted/50 cursor-pointer ${
                    selectedRow === index ? "bg-primary/10" : ""
                  }`}
                  onClick={() => onRowClick(index)}
                >
                  <TableCell className="font-medium whitespace-normal break-words">{item.requirement}</TableCell>
                  <TableCell className="whitespace-normal break-words">
                    <span
                      className="text-primary focus:outline-none "
                    >
                      {item.source}
                    </span>
                  </TableCell>
                  <TableCell className="whitespace-normal break-words">{item.rationale}</TableCell>
                  <TableCell className="whitespace-normal">{getStatusBadge(item.status)}</TableCell>
                  <TableCell className="whitespace-normal break-words">{item.summary}</TableCell>
                  <TableCell className="whitespace-normal">
                    <div className="flex flex-wrap gap-1">
                      {item.citations && item.citations.length > 0 ? (
                        [...new Set(item.citations)].map((citation, idx) => (
                          <button
                            key={idx}
                            className="px-2 py-1 text-xs bg-primary/10 text-primary rounded hover:bg-primary/20 focus:outline-none transition-colors"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleCitationClick(citation)
                            }}
                          >
                            page no.{citation}
                          </button>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground italic">No citations available</span>
                      )}
                    </div>
                  </TableCell>
                </motion.tr>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={6} className="h-24 text-center">
                  {data === null ? "Upload a PDF to view compliance data" : "No results found"}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </Card>
  )
}
