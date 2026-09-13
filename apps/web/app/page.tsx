"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type ApiState = "checking" | "ready" | "unavailable";

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [apiState, setApiState] = useState<ApiState>("checking");

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${apiUrl}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Health check failed");
        setApiState("ready");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setApiState("unavailable");
      });

    return () => controller.abort();
  }, []);

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">AI Knowledge Workspace</p>
        <h1>Your sources. Clear answers.</h1>
        <p className="intro">
          Build a trusted knowledge base from your documents and ask questions with
          citations back to the original source.
        </p>
        <div className={`status status--${apiState}`} role="status">
          <span aria-hidden="true" />
          API {apiState === "checking" ? "checking" : apiState}
        </div>
        <p><Link className="primary-link" href="/knowledge-bases">Open knowledge bases</Link></p>
      </section>

      <section className="foundation" aria-labelledby="foundation-title">
        <p className="step">Phase 1 · Foundation</p>
        <h2 id="foundation-title">Local workspace is taking shape</h2>
        <div className="services">
          <article><strong>Next.js</strong><span>Member workspace</span></article>
          <article><strong>FastAPI</strong><span>Core API</span></article>
          <article><strong>PostgreSQL</strong><span>Source of truth</span></article>
          <article><strong>Redis</strong><span>Jobs and cache</span></article>
          <article><strong>MinIO</strong><span>Document storage</span></article>
        </div>
      </section>
    </main>
  );
}
