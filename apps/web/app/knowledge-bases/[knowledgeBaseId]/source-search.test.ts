import assert from "node:assert/strict";
import test from "node:test";

import {
  SourceSearchValidationError,
  createSourceSearchRequestManager,
  parseSourceSearchResponse,
} from "./source-search.ts";


const sourceResult = {
  source_filename: "handbook.pdf",
  file_type: "pdf",
  excerpt: "Annual leave is available after probation.",
};


test("source search trims a question and rejects whitespace-only input", async () => {
  const manager = createSourceSearchRequestManager();
  const questions: string[] = [];

  const result = await manager.submit("  How do I request leave?  ", async (question) => {
    questions.push(question);
    return { outcome: "results", results: [sourceResult] };
  });

  await assert.rejects(
    () => manager.submit("   ", async () => ({ outcome: "no_results", results: [] })),
    SourceSearchValidationError,
  );
  assert.equal(result.current, true);
  assert.equal(result.response?.outcome, "results");
  assert.deepEqual(questions, ["How do I request leave?"]);
});


test("source search distinguishes no-ready-documents and no-results outcomes", () => {
  assert.deepEqual(parseSourceSearchResponse({ outcome: "no_ready_documents", results: [] }), {
    outcome: "no_ready_documents",
    results: [],
  });
  assert.deepEqual(parseSourceSearchResponse({ outcome: "no_results", results: [] }), {
    outcome: "no_results",
    results: [],
  });
});


test("source search preserves the last trimmed question for Retry", async () => {
  const manager = createSourceSearchRequestManager();
  const questions: string[] = [];

  const failed = await manager.submit("  Leave policy  ", async (question) => {
    questions.push(question);
    throw new Error("Source search is temporarily unavailable. Please try again.");
  });
  const retried = await manager.retry(async (question) => {
    questions.push(question);
    return { outcome: "no_results", results: [] };
  });

  assert.equal(failed.current, true);
  assert.ok(failed.error instanceof Error);
  assert.equal(retried.current, true);
  assert.equal(retried.response?.outcome, "no_results");
  assert.deepEqual(questions, ["Leave policy", "Leave policy"]);
});


test("a later source search aborts and supersedes a stale response", async () => {
  const manager = createSourceSearchRequestManager();
  let resolveFirst: ((value: { outcome: "no_results"; results: [] }) => void) | undefined;
  let firstSignal: AbortSignal | undefined;

  const first = manager.submit("first question", (_question, signal) => {
    firstSignal = signal;
    return new Promise((resolve) => {
      resolveFirst = resolve;
    });
  });
  const second = await manager.submit("second question", async () => ({
    outcome: "results",
    results: [sourceResult],
  }));
  resolveFirst?.({ outcome: "no_results", results: [] });
  const firstResult = await first;

  assert.equal(firstSignal?.aborted, true);
  assert.equal(firstResult.current, false);
  assert.equal(second.current, true);
  assert.equal(second.response?.outcome, "results");
});


test("source search rejects unsafe or oversized result payloads", () => {
  assert.throws(
    () => parseSourceSearchResponse({ outcome: "results", results: [] }),
    /Source search data is unavailable/,
  );
  assert.throws(
    () =>
      parseSourceSearchResponse({
        outcome: "results",
        results: Array.from({ length: 6 }, () => sourceResult),
      }),
    /Source search data is unavailable/,
  );
  assert.throws(
    () =>
      parseSourceSearchResponse({
        outcome: "results",
        results: [{ ...sourceResult, excerpt: "x".repeat(501) }],
      }),
    /Source search data is unavailable/,
  );
});
