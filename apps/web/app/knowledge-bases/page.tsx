"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

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
  const nameInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadKnowledgeBases() {
      try {
        const response = await fetch(`${apiUrl}/knowledge-bases`, { signal: controller.signal });
        if (!response.ok) throw new Error(await errorMessage(response));
        setKnowledgeBases((await response.json()) as KnowledgeBase[]);
      } catch (error: unknown) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoadError(error instanceof Error ? error.message : "Could not load knowledge bases.");
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    }

    void loadKnowledgeBases();
    return () => controller.abort();
  }, []);

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
                <strong>{knowledgeBase.name}</strong>
                <time dateTime={knowledgeBase.created_at}>{formatCreatedAt(knowledgeBase.created_at)}</time>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
