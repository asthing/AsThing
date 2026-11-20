"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { DeleteConfirmation } from "../components/delete-confirmation";
import {
  createKnowledgeBaseDeletionManager,
  knowledgeBaseDeletionConfirmation,
} from "./knowledge-base-deletion";

type KnowledgeBase = {
  id: string;
  name: string;
  created_at: string;
};

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function formatCreatedAt(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(new Date(value));
}

async function errorMessage(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null);
  if (body && typeof body === "object" && "detail" in body && typeof body.detail === "string") {
    return body.detail;
  }
  return "Something went wrong. Please try again.";
}

export default function KnowledgeBasesPage() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [name, setName] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{ id: string; message: string } | null>(null);
  const [deleteSuccess, setDeleteSuccess] = useState<string | null>(null);
  const [deletionTarget, setDeletionTarget] = useState<KnowledgeBase | null>(null);
  const deletion = useMemo(() => createKnowledgeBaseDeletionManager(), []);
  const nameInput = useRef<HTMLInputElement>(null);

  useEffect(() => () => deletion.cancel(), [deletion]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadKnowledgeBases() {
      try {
        const response = await fetch(`${apiUrl}/knowledge-bases`, { signal: controller.signal });
        if (!response.ok) throw new Error(await errorMessage(response));
        const loaded = (await response.json()) as KnowledgeBase[];
        if (!controller.signal.aborted) setKnowledgeBases(deletion.filter(loaded));
      } catch (error: unknown) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoadError(error instanceof Error ? error.message : "Could not load knowledge bases.");
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    }

    void loadKnowledgeBases();
    return () => controller.abort();
  }, [deletion]);

  async function handleDelete(knowledgeBase: KnowledgeBase) {
    if (deletingId) return;
    setDeletingId(knowledgeBase.id);
    setDeleteError(null);
    setDeleteSuccess(null);
    const result = await deletion.submit(
      knowledgeBase,
      () => true, // Only invoked after confirmation in the in-app dialog.
      async (id, signal) => {
        const response = await fetch(`${apiUrl}/knowledge-bases/${id}`, {
          method: "DELETE", signal,
        });
        if (response.status !== 204) throw new Error(await errorMessage(response));
      },
    );
    if (!result.current) return;
    setDeletingId(null);
    if (result.cancelled) return;
    if (result.error) {
      setDeleteError({
        id: knowledgeBase.id,
        message: result.error instanceof Error
          ? result.error.message : "Could not delete the knowledge base. Please try again.",
      });
      return;
    }
    setKnowledgeBases((current) => deletion.filter(current));
    setDeleteSuccess(`Deleted ${knowledgeBase.name} and its documents.`);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedName = name.trim();

    if (!trimmedName) {
      setFormError("Enter a name for your knowledge base.");
      nameInput.current?.focus();
      return;
    }
    if (trimmedName.length > 100) {
      setFormError("Name must be at most 100 characters.");
      return;
    }

    setFormError(null);
    setIsSubmitting(true);
    try {
      const response = await fetch(`${apiUrl}/knowledge-bases`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmedName }),
      });
      if (!response.ok) throw new Error(await errorMessage(response));

      const knowledgeBase = (await response.json()) as KnowledgeBase;
      setKnowledgeBases((current) => [knowledgeBase, ...current]);
      setName("");
    } catch (error: unknown) {
      setFormError(error instanceof Error ? error.message : "Could not create the knowledge base.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="knowledge-bases-page">
      {deletionTarget && (
        <DeleteConfirmation
          key={deletionTarget.id}
          title="Delete knowledge base?"
          message={knowledgeBaseDeletionConfirmation(deletionTarget.name)}
          confirmLabel="Delete knowledge base"
          onCancel={() => setDeletionTarget(null)}
          onConfirm={() => {
            setDeletionTarget(null);
            void handleDelete(deletionTarget);
          }}
        />
      )}
      <section className="page-heading">
        <p className="eyebrow">Workspace</p>
        <h1>Knowledge Bases</h1>
        <p className="intro">Create a place for documents you want to organize and search together.</p>
      </section>

      <section className="create-panel" aria-labelledby="create-title">
        <h2 id="create-title">Create knowledge base</h2>
        <form onSubmit={handleSubmit} noValidate>
          <label htmlFor="knowledge-base-name">Name</label>
          <div className="create-row">
            <input
              ref={nameInput}
              id="knowledge-base-name"
              name="name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={101}
              aria-describedby={formError ? "knowledge-base-error" : undefined}
              aria-invalid={Boolean(formError)}
              placeholder="For example, Engineering Notes"
            />
            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Creating…" : "Create knowledge base"}
            </button>
          </div>
          {formError && <p className="form-error" id="knowledge-base-error" role="alert">{formError}</p>}
        </form>
      </section>

      <section className="knowledge-base-list" aria-labelledby="list-title">
        <div className="list-heading">
          <h2 id="list-title">Your knowledge bases</h2>
          {!isLoading && <span>{knowledgeBases.length}</span>}
        </div>

        {isLoading && <p className="muted">Loading knowledge bases…</p>}
        {loadError && <p className="form-error" role="alert">{loadError}</p>}
        {deleteSuccess && <p role="status">{deleteSuccess}</p>}
        {!isLoading && !loadError && knowledgeBases.length === 0 && (
          <div className="empty-state">
            <p>No knowledge bases yet. Create one to organize your documents by topic.</p>
            <button type="button" onClick={() => nameInput.current?.focus()}>Create knowledge base</button>
          </div>
        )}
        {!isLoading && !loadError && knowledgeBases.length > 0 && (
          <ul>
            {knowledgeBases.map((knowledgeBase) => (
              <li key={knowledgeBase.id}>
                <div>
                  <strong>{knowledgeBase.name}</strong>
                  <time dateTime={knowledgeBase.created_at}>{formatCreatedAt(knowledgeBase.created_at)}</time>
                </div>
                <div>
                  <Link className="secondary-link" href={`/knowledge-bases/${knowledgeBase.id}`}>Open</Link>
                  <button type="button" disabled={deletingId !== null}
                    onClick={() => setDeletionTarget(knowledgeBase)}>
                    {deletingId === knowledgeBase.id ? "Deleting…"
                      : deleteError?.id === knowledgeBase.id ? "Retry deletion" : "Delete"}
                  </button>
                  {deleteError?.id === knowledgeBase.id && (
                    <p className="form-error" role="alert">{deleteError.message}</p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
