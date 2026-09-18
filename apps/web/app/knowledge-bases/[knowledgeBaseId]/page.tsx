"use client";

import Link from "next/link";
import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";

import {
  getDocumentStatusPresentation,
  parseAndSortDocuments,
  shouldPollDocuments,
  startDocumentPolling,
  type DocumentListItem,
} from "./document-status";
import { createAnswerRequestManager, type AnswerResponse } from "./answer-request";
import { createSourceSearchRequestManager, type SourceSearchResponse } from "./source-search";

type KnowledgeBase = {
  id: string;
  name: string;
};

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const maxFileSizeBytes = 25 * 1024 * 1024;
const supportedExtensions = [".pdf", ".txt", ".docx"];

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatUploadTime(uploadedAt: string): string {
  const date = new Date(uploadedAt);
  if (Number.isNaN(date.getTime())) return "Upload time unavailable";

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function validateFile(file: File): string | null {
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!supportedExtensions.includes(extension)) {
    return "Unsupported file type. Choose a PDF, TXT, or DOCX file.";
  }
  if (file.size > maxFileSizeBytes) {
    return "File is too large. The maximum size is 25 MB.";
  }
  return null;
}

async function errorMessage(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null);
  if (body && typeof body === "object" && "detail" in body && typeof body.detail === "string") {
    return body.detail;
  }
  return "Something went wrong. Please try again.";
}

function SourceSearchPanel({ knowledgeBaseId }: { knowledgeBaseId: string }) {
  const [question, setQuestion] = useState("");
  const [sourceSearchResponse, setSourceSearchResponse] = useState<SourceSearchResponse | null>(null);
  const [sourceSearchError, setSourceSearchError] = useState<string | null>(null);
  const [isSearchingSources, setIsSearchingSources] = useState(false);
  const [answerResponse, setAnswerResponse] = useState<AnswerResponse | null>(null);
  const [answerError, setAnswerError] = useState<string | null>(null);
  const [isGeneratingAnswer, setIsGeneratingAnswer] = useState(false);
  const sourceSearchRequests = useRef(createSourceSearchRequestManager());
  const answerRequests = useRef(createAnswerRequestManager());

  useEffect(() => () => {
    sourceSearchRequests.current.cancel();
    answerRequests.current.cancel();
  }, []);

  async function fetchSourceSearch(questionToSearch: string, signal: AbortSignal): Promise<unknown> {
    const response = await fetch(`${apiUrl}/knowledge-bases/${knowledgeBaseId}/sources/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionToSearch }),
      signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    return response.json();
  }

  async function searchSources(retry = false) {
    if (isSearchingSources || isGeneratingAnswer) return;
    setSourceSearchError(null);
    if (!retry) {
      setSourceSearchResponse(null);
      setAnswerResponse(null);
      setAnswerError(null);
    }
    setIsSearchingSources(true);
    let requestResult;
    try {
      requestResult = retry
        ? await sourceSearchRequests.current.retry(fetchSourceSearch)
        : await sourceSearchRequests.current.submit(question, fetchSourceSearch);
      if (!requestResult.current) return;
      if (requestResult.error) throw requestResult.error;
      if (!requestResult.response) throw new Error("Source search data is unavailable. Please try again.");
      setSourceSearchResponse(requestResult.response);
    } catch (error: unknown) {
      if (requestResult && !requestResult.current) return;
      const message =
        error instanceof Error ? error.message : "Could not search source documents. Please try again.";
      setSourceSearchError(message);
    } finally {
      if (!requestResult || requestResult.current) setIsSearchingSources(false);
    }
  }

  function handleSourceSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void searchSources();
  }

  async function fetchAnswer(questionToAsk: string, signal: AbortSignal): Promise<unknown> {
    const response = await fetch(`${apiUrl}/knowledge-bases/${knowledgeBaseId}/answers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionToAsk }),
      signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    return response.json();
  }

  async function askForAnswer(retry = false) {
    if (isSearchingSources || isGeneratingAnswer) return;
    setAnswerError(null);
    if (!retry) {
      setAnswerResponse(null);
      setSourceSearchResponse(null);
      setSourceSearchError(null);
    }
    setIsGeneratingAnswer(true);
    let requestResult;
    try {
      requestResult = retry
        ? await answerRequests.current.retry(fetchAnswer)
        : await answerRequests.current.submit(question, fetchAnswer);
      if (!requestResult.current) return;
      if (requestResult.error) throw requestResult.error;
      if (!requestResult.response) throw new Error("Answer data is unavailable. Please try again.");
      setAnswerResponse(requestResult.response);
    } catch (error: unknown) {
      if (requestResult && !requestResult.current) return;
      setAnswerError(
        error instanceof Error ? error.message : "Could not generate an answer. Please try again.",
      );
    } finally {
      if (!requestResult || requestResult.current) setIsGeneratingAnswer(false);
    }
  }

  return (
    <section className="source-search-panel" aria-labelledby="source-search-title">
      <h2 id="source-search-title">Find relevant sources</h2>
      <p className="muted">Searches only documents that are ready in this knowledge base.</p>
      <form onSubmit={handleSourceSearch} noValidate>
        <label htmlFor="source-question">Question</label>
        <textarea
          id="source-question"
          name="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          disabled={isSearchingSources || isGeneratingAnswer}
          aria-describedby={sourceSearchError ? "source-search-error" : answerError ? "answer-error" : undefined}
        />
        <div className="upload-actions">
          <button type="submit" disabled={isSearchingSources || isGeneratingAnswer}>
            {isSearchingSources ? "Finding sources…" : "Find relevant sources"}
          </button>
          <button type="button" onClick={() => void askForAnswer()} disabled={isSearchingSources || isGeneratingAnswer}>
            {isGeneratingAnswer ? "Generating answer…" : "Get answer"}
          </button>
        </div>
      </form>

      {isGeneratingAnswer && <p className="muted" role="status">Generating an answer from this knowledge base…</p>}
      {answerError && (
        <div className="empty-state" id="answer-error" role="alert">
          <p>{answerError}</p>
          <button type="button" onClick={() => void askForAnswer(true)} disabled={isGeneratingAnswer}>
            Retry answer
          </button>
        </div>
      )}
      {answerResponse?.outcome === "no_answer" && (
        <div className="empty-state" role="status"><p>{answerResponse.message}</p></div>
      )}
      {answerResponse?.outcome === "answer" && (
        <div className="answer-result" aria-live="polite">
          <h3>Answer</h3>
          <p>{answerResponse.answer}</p>
        </div>
      )}

      {isSearchingSources && <p className="muted" role="status">Finding relevant sources…</p>}
      {sourceSearchError && (
        <div className="empty-state" id="source-search-error" role="alert">
          <p>{sourceSearchError}</p>
          <button type="button" onClick={() => void searchSources(true)} disabled={isSearchingSources}>
            Retry
          </button>
        </div>
      )}
      {sourceSearchResponse?.outcome === "no_ready_documents" && (
        <div className="empty-state">
          <p>Documents must finish processing before they can be searched.</p>
        </div>
      )}
      {sourceSearchResponse?.outcome === "no_results" && (
        <div className="empty-state">
          <p>No relevant source passages were found for this question.</p>
        </div>
      )}
      {sourceSearchResponse?.outcome === "results" && (
        <ul className="source-results" aria-label="Relevant source passages">
          {sourceSearchResponse.results.map((result, index) => (
            <li key={`${result.source_filename}-${index}`}>
              <div>
                <strong>{result.source_filename}</strong>
                <span>{result.file_type.toUpperCase()}</span>
              </div>
              <p>{result.excerpt}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function KnowledgeBasePage() {
  const { knowledgeBaseId } = useParams<{ knowledgeBaseId: string }>();
  const [knowledgeBase, setKnowledgeBase] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDocumentsLoading, setIsDocumentsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [pollingCycle, setPollingCycle] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const activeDocumentsRequest = useRef<AbortController | null>(null);
  const documentsRequestId = useRef(0);
  const pollingDocuments = useRef(documents);

  const cancelDocumentsRequest = useCallback(() => {
    documentsRequestId.current += 1;
    activeDocumentsRequest.current?.abort();
    activeDocumentsRequest.current = null;
  }, []);

  const loadDocuments = useCallback(async ({ background = false } = {}): Promise<
    DocumentListItem[] | null
  > => {
    activeDocumentsRequest.current?.abort();
    const controller = new AbortController();
    const requestId = documentsRequestId.current + 1;
    documentsRequestId.current = requestId;
    activeDocumentsRequest.current = controller;
    if (!background) setIsDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const response = await fetch(`${apiUrl}/knowledge-bases/${knowledgeBaseId}/documents`, {
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const loadedDocuments = parseAndSortDocuments(await response.json());
      if (controller.signal.aborted || requestId !== documentsRequestId.current) return null;
      setDocuments(loadedDocuments);
      return loadedDocuments;
    } catch (error: unknown) {
      if (controller.signal.aborted || requestId !== documentsRequestId.current) return null;
      const message =
        error instanceof Error ? error.message : "Could not load the documents. Please try again.";
      setDocumentsError(message);
      return null;
    } finally {
      if (requestId === documentsRequestId.current) {
        activeDocumentsRequest.current = null;
        if (!background) setIsDocumentsLoading(false);
      }
    }
  }, [knowledgeBaseId]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadKnowledgeBase() {
      try {
        const response = await fetch(`${apiUrl}/knowledge-bases/${knowledgeBaseId}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        setKnowledgeBase((await response.json()) as KnowledgeBase);
        setPageError(null);
        await loadDocuments();
      } catch (error: unknown) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setPageError(error instanceof Error ? error.message : "Could not load this knowledge base.");
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    }

    void loadKnowledgeBase();
    return () => {
      controller.abort();
      cancelDocumentsRequest();
    };
  }, [cancelDocumentsRequest, knowledgeBaseId, loadDocuments]);

  const pollingRequired =
    !isDocumentsLoading && !documentsError && shouldPollDocuments(documents, 0);
  useEffect(() => {
    pollingDocuments.current = documents;
  }, [documents]);

  useEffect(() => {
    if (!pollingRequired) return;
    const stopPolling = startDocumentPolling({
      initialDocuments: pollingDocuments.current,
      refresh: () => loadDocuments({ background: true }),
    });
    return () => {
      stopPolling();
      cancelDocumentsRequest();
    };
  }, [cancelDocumentsRequest, knowledgeBaseId, loadDocuments, pollingCycle, pollingRequired]);

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setSuccessMessage(null);
    if (!file) return;

    const validationError = validateFile(file);
    if (validationError) {
      setSelectedFile(null);
      setFileError(validationError);
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
    setFileError(null);
  }

  function cancelSelection() {
    setSelectedFile(null);
    setFileError(null);
    if (fileInput.current) fileInput.current.value = "";
  }

  function openFilePicker() {
    fileInput.current?.click();
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) return;

    setFileError(null);
    setSuccessMessage(null);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch(`${apiUrl}/knowledge-bases/${knowledgeBaseId}/documents`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(await errorMessage(response));

      const [document] = parseAndSortDocuments([await response.json()]);
      cancelDocumentsRequest();
      setDocuments((currentDocuments) =>
        parseAndSortDocuments([
          document,
          ...currentDocuments.filter((currentDocument) => currentDocument.id !== document.id),
        ]),
      );
      setIsDocumentsLoading(false);
      setDocumentsError(null);
      setSuccessMessage(`Uploaded ${selectedFile.name}.`);
      setPollingCycle((currentCycle) => currentCycle + 1);
      cancelSelection();
    } catch (error: unknown) {
      setFileError(error instanceof Error ? error.message : "Could not upload the document.");
    } finally {
      setIsUploading(false);
    }
  }

  if (isLoading) return <main className="knowledge-base-detail"><p className="muted">Loading knowledge base…</p></main>;
  if (pageError) return <main className="knowledge-base-detail"><p className="form-error" role="alert">{pageError}</p></main>;
  if (!knowledgeBase) return null;

  return (
    <main className="knowledge-base-detail">
      <Link className="back-link" href="/knowledge-bases">← Knowledge Bases</Link>
      <section className="page-heading">
        <p className="eyebrow">Knowledge Base</p>
        <h1>{knowledgeBase.name}</h1>
        <p className="intro">Upload source documents and follow their preparation status here.</p>
      </section>

      <section className="document-list" aria-labelledby="documents-title">
        <div className="list-heading">
          <h2 id="documents-title">Documents</h2>
          {!isDocumentsLoading && !documentsError && <span>{documents.length}</span>}
        </div>

        {isDocumentsLoading && <p className="muted">Loading documents...</p>}
        {documentsError && (
          <div className="empty-state" role="alert">
            <p>{documentsError}</p>
            <button type="button" onClick={() => void loadDocuments()}>Retry</button>
          </div>
        )}
        {!isDocumentsLoading && !documentsError && documents.length === 0 && (
          <div className="empty-state">
            <p>No documents uploaded yet. Upload a source file to add it to this knowledge base.</p>
            <button type="button" onClick={openFilePicker}>Upload a document</button>
          </div>
        )}
        {!isDocumentsLoading && !documentsError && documents.length > 0 && (
          <ul>
            {documents.map((document) => {
              const status = getDocumentStatusPresentation(
                document.status,
                document.failure_message,
              );
              return (
                <li key={document.id}>
                  <div>
                    <strong>{document.original_filename}</strong>
                    <span>{document.file_type.toUpperCase()} &middot; {formatFileSize(document.file_size_bytes)}</span>
                    <time dateTime={document.uploaded_at}>Uploaded {formatUploadTime(document.uploaded_at)}</time>
                  </div>
                  <div
                    className={`document-status document-status--${status.tone}`}
                    aria-label={`Processing status: ${status.label}`}
                    aria-live="polite"
                  >
                    <strong>
                      <span className="document-status-dot" aria-hidden="true" />
                      {status.label}
                    </strong>
                    {status.description && <span>{status.description}</span>}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <SourceSearchPanel key={knowledgeBaseId} knowledgeBaseId={knowledgeBaseId} />

      <section className="upload-panel" aria-labelledby="upload-title">
        <h2 id="upload-title">Upload document</h2>
        <p className="muted">PDF, TXT, or DOCX · Maximum 25 MB · One file per upload</p>
        <form onSubmit={handleUpload} noValidate>
          <label className="file-label" htmlFor="document-file">Select a file</label>
          <input
            ref={fileInput}
            id="document-file"
            name="file"
            type="file"
            accept=".pdf,.txt,.docx"
            onChange={handleFileSelection}
            disabled={isUploading}
            aria-describedby={fileError ? "document-file-error" : undefined}
          />

          {selectedFile && (
            <div className="selected-file">
              <div>
                <strong>{selectedFile.name}</strong>
                <span>{formatFileSize(selectedFile.size)}</span>
              </div>
              <button type="button" className="text-button" onClick={cancelSelection} disabled={isUploading}>
                Cancel
              </button>
            </div>
          )}

          <div className="upload-actions">
            <button type="submit" disabled={!selectedFile || isUploading}>
              {isUploading ? "Uploading…" : "Upload document"}
            </button>
          </div>
          {fileError && <p className="form-error" id="document-file-error" role="alert">{fileError}</p>}
          {successMessage && <p className="success-message" role="status">{successMessage}</p>}
        </form>
      </section>
    </main>
  );
}
