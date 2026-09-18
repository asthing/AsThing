import assert from "node:assert/strict";
import test from "node:test";

import { createAnswerRequestManager, parseAnswerResponse } from "./answer-request.ts";

const answer = {
  outcome: "answer",
  answer: "Use the HR portal.",
  reason: null,
  message: null,
  references: [{ reference_key: "source-1", source_filename: "policy.txt", file_type: "txt", excerpt: "Use the HR portal." }],
};

test("answer request trims input and rejects blank questions", async () => {
  const manager = createAnswerRequestManager();
  const questions: string[] = [];
  assert.throws(() => manager.submit("   ", async () => answer), /Enter a question/);
  const result = await manager.submit("  How?  ", async (question) => {
    questions.push(question);
    return answer;
  });
  assert.equal(result.response?.outcome, "answer");
  assert.deepEqual(questions, ["How?"]);
});

test("duplicate submissions are prevented while a request is active", async () => {
  const manager = createAnswerRequestManager();
  let resolveFirst: ((value: typeof answer) => void) | undefined;
  const first = manager.submit("How?", () => new Promise((resolve) => { resolveFirst = resolve; }));
  assert.throws(() => manager.submit("Again?", async () => answer), /already being generated/);
  resolveFirst?.(answer);
  assert.equal((await first).response?.outcome, "answer");
  const retry = await manager.retry(async (question) => {
    assert.equal(question, "How?");
    return answer;
  });
  assert.equal(retry.response?.outcome, "answer");
});

test("no-answer outcomes are explicit and references remain API-only data", () => {
  assert.equal(parseAnswerResponse({ outcome: "no_answer", answer: null, reason: "no_ready_documents", message: "Wait for documents.", references: [] }).outcome, "no_answer");
  assert.equal(parseAnswerResponse({ outcome: "no_answer", answer: null, reason: "insufficient_evidence", message: "No answer.", references: [] }).outcome, "no_answer");
  assert.throws(() => parseAnswerResponse({ ...answer, references: [] }), /unavailable/);
  assert.throws(() => parseAnswerResponse({ ...answer, answer: "x".repeat(701) }), /unavailable/);
});

test("Retry uses the last trimmed question after a safe failure", async () => {
  const manager = createAnswerRequestManager();
  const calls: string[] = [];
  const failed = await manager.submit("  How?  ", async (question) => {
    calls.push(question);
    throw new Error("Answer generation is temporarily unavailable. Please try again.");
  });
  const retried = await manager.retry(async (question) => {
    calls.push(question);
    return answer;
  });
  assert.ok(failed.error);
  assert.equal(retried.response?.outcome, "answer");
  assert.deepEqual(calls, ["How?", "How?"]);
});

test("cancel aborts the request and rejects a stale response", async () => {
  const manager = createAnswerRequestManager();
  let resolveFirst: ((value: typeof answer) => void) | undefined;
  let signal: AbortSignal | undefined;
  const first = manager.submit("How?", (_question, requestSignal) => {
    signal = requestSignal;
    return new Promise((resolve) => { resolveFirst = resolve; });
  });
  manager.cancel();
  resolveFirst?.(answer);
  assert.equal(signal?.aborted, true);
  assert.equal((await first).current, false);
});
