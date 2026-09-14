export type PublicDocumentStatus = "Uploaded" | "Processing" | "Ready" | "Failed";

export type DocumentListItem = {
  id: string;
  knowledge_base_id: string;
  original_filename: string;
  file_size_bytes: number;
  file_type: string;
  status: PublicDocumentStatus;
  uploaded_at: string;
  failure_message: string | null;
};

export type DocumentStatusPresentation = {
  label: PublicDocumentStatus;
  description: string | null;
  tone: "uploaded" | "processing" | "ready" | "failed";
};

type DocumentPollingOptions = {
  initialDocuments: DocumentListItem[];
  refresh: () => Promise<DocumentListItem[] | null>;
  schedule?: (callback: () => Promise<void>, delay: number) => unknown;
  cancel?: (token: unknown) => void;
};

// One request at a time, every two seconds, for at most four minutes per polling cycle.
export const DOCUMENT_POLL_INTERVAL_MS = 2_000;
export const DOCUMENT_POLL_MAX_ATTEMPTS = 120;

const publicDocumentStatuses = new Set<PublicDocumentStatus>([
  "Uploaded",
  "Processing",
  "Ready",
  "Failed",
]);

export function shouldPollDocuments(
  documents: DocumentListItem[],
  completedAttempts: number,
): boolean {
  return (
    completedAttempts < DOCUMENT_POLL_MAX_ATTEMPTS &&
    documents.some(
      (document) => document.status === "Uploaded" || document.status === "Processing",
    )
  );
}

export function startDocumentPolling({
  initialDocuments,
  refresh,
  schedule = (callback, delay) => setTimeout(() => void callback(), delay),
  cancel = (token) => clearTimeout(token as ReturnType<typeof setTimeout>),
}: DocumentPollingOptions): () => void {
  let currentDocuments = initialDocuments;
  let completedAttempts = 0;
  let timer: unknown = null;
  let stopped = false;

  function scheduleNext(): void {
    if (stopped || !shouldPollDocuments(currentDocuments, completedAttempts)) return;
    timer = schedule(async () => {
      timer = null;
      if (stopped) return;

      let refreshedDocuments: DocumentListItem[] | null;
      try {
        refreshedDocuments = await refresh();
      } catch {
        return;
      }
      if (stopped || refreshedDocuments === null) return;

      completedAttempts += 1;
      currentDocuments = refreshedDocuments;
      scheduleNext();
    }, DOCUMENT_POLL_INTERVAL_MS);
  }

  scheduleNext();
  return () => {
    stopped = true;
    if (timer !== null) cancel(timer);
    timer = null;
  };
}

export function parseAndSortDocuments(value: unknown): DocumentListItem[] {
  if (!Array.isArray(value)) {
    throw new Error("Document data is unavailable. Please try again.");
  }

  const documents = value.map(parseDocument);
  return documents.toSorted((left, right) => {
    const timeDifference = uploadTimestamp(right.uploaded_at) - uploadTimestamp(left.uploaded_at);
    return timeDifference || right.id.localeCompare(left.id);
  });
}

export function getDocumentStatusPresentation(
  status: PublicDocumentStatus,
  failureMessage: string | null,
): DocumentStatusPresentation {
  switch (status) {
    case "Uploaded":
      return {
        label: status,
        description: "Accepted and queued. Not searchable yet.",
        tone: "uploaded",
      };
    case "Processing":
      return {
        label: status,
        description: "Being prepared for retrieval. Not searchable yet.",
        tone: "processing",
      };
    case "Ready":
      return {
        label: status,
        description: "Available for future retrieval and AI answers.",
        tone: "ready",
      };
    case "Failed":
      return {
        label: status,
        description: failureMessage,
        tone: "failed",
      };
  }
}

function parseDocument(value: unknown): DocumentListItem {
  if (!isRecord(value)) {
    throw new Error("Document data is unavailable. Please try again.");
  }
  if (
    typeof value.status !== "string" ||
    !publicDocumentStatuses.has(value.status as PublicDocumentStatus)
  ) {
    throw new Error("Document status is unavailable. Please try again.");
  }
  if (
    typeof value.id !== "string" ||
    typeof value.knowledge_base_id !== "string" ||
    typeof value.original_filename !== "string" ||
    typeof value.file_size_bytes !== "number" ||
    typeof value.file_type !== "string" ||
    typeof value.uploaded_at !== "string" ||
    (value.failure_message !== null && typeof value.failure_message !== "string")
  ) {
    throw new Error("Document data is unavailable. Please try again.");
  }

  return {
    id: value.id,
    knowledge_base_id: value.knowledge_base_id,
    original_filename: value.original_filename,
    file_size_bytes: value.file_size_bytes,
    file_type: value.file_type,
    status: value.status as PublicDocumentStatus,
    uploaded_at: value.uploaded_at,
    failure_message: value.failure_message,
  };
}

function uploadTimestamp(value: string): number {
  const timestamp = Date.parse(value);
  return Number.isNaN(timestamp) ? Number.NEGATIVE_INFINITY : timestamp;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
