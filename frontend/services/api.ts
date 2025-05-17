type ApiResponse<T> = {
  data: T | null;
  error: string | null;
};

export type TaskStatus = {
  task_id: string;
  type: string;
  status: string;
  created_at: string;
  progress?: number;
  result?: any;
  error?: string;
};

export type ComplianceReportItem = {
  compliance_requirement: string;
  source: string;
  rationale: string;
  compliance_status: string;
  findings_summary: string;
  findings_citations: number[];
};

export type ComplianceReport = {
  report_items: ComplianceReportItem[];
  compliant_count: number;
  non_compliant_count: number;
  partial_compliant_count: number;
  unknown_count: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api";

export async function uploadSIDDocument(
  file: File
): Promise<ApiResponse<{ task_id: string }>> {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/upload/sid-document`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      return { data: null, error: data.error || "Failed to upload document" };
    }

    return { data: { task_id: data.task_id }, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    };
  }
}

export async function getTaskStatus(
  taskId: string
): Promise<ApiResponse<TaskStatus>> {
  try {
    const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`);
    const data = await response.json();

    if (!response.ok) {
      return { data: null, error: data.error || "Failed to get task status" };
    }

    return { data: data, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    };
  }
}

export async function mapBackendToFrontendData(
  complianceData: ComplianceReportItem[]
): Promise<any[]> {
  return complianceData.map((item) => ({
    requirement: item.compliance_requirement,
    source: item.source,
    rationale: item.rationale,
    status: item.compliance_status,
    summary: item.findings_summary,
    citations: item.findings_citations,
  }));
}

export async function evaluateDocumentWithId(docId: string) {
  try {
    const response = await fetch(`/api/evaluate-checklist/${docId}`, {
      method: "GET",
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        data: null,
        error: errorData.error || "Failed to evaluate document",
      };
    }

    const data = await response.json();
    return { data, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

export async function uploadMasterCircular(
  file: File
): Promise<ApiResponse<{ task_id: string }>> {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/upload/master-circular`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        data: null,
        error: data.error || "Failed to upload master circular",
      };
    }

    return { data: { task_id: data.task_id }, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    };
  }
}
