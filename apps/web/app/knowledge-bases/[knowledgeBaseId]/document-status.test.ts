import assert from "node:assert/strict";
import test from "node:test";

import {
  DOCUMENT_POLL_INTERVAL_MS,
  DOCUMENT_POLL_MAX_ATTEMPTS,
  getDocumentStatusPresentation,
  parseAndSortDocuments,
  shouldPollDocuments,
  startDocumentPolling,
  type DocumentListItem,
} from "./document-status.ts";

function documentItem(
  id: string,
  status: DocumentListItem["status"],
  uploadedAt: string,
  failureMessage: string | null = null,
): DocumentListItem {
  return {
    id,
    knowledge_base_id: "11111111-1111-1111-1111-111111111111",
    original_filename: `${id}.txt`,
    file_size_bytes: 12,
    file_type: "txt",
    status,
    uploaded_at: uploadedAt,
    failure_message: failureMessage,
  };
}

test("polling has a bounded four-minute lifetime for cold local-model startup", () => {
  assert.equal(DOCUMENT_POLL_INTERVAL_MS * DOCUMENT_POLL_MAX_ATTEMPTS, 4 * 60 * 1_000);
});

test("polling continues only for nonterminal documents and below the attempt limit", () => {
  assert.equal(
    shouldPollDocuments(
      [documentItem("queued", "Uploaded", "2026-09-14T10:00:00Z")],
      0,
    ),
    true,
  );
  assert.equal(
    shouldPollDocuments(
      [documentItem("processing", "Processing", "2026-09-14T10:00:00Z")],
      DOCUMENT_POLL_MAX_ATTEMPTS - 1,
    ),
    true,
  );
  assert.equal(
    shouldPollDocuments(
      [
        documentItem("ready", "Ready", "2026-09-14T10:00:00Z"),
        documentItem("failed", "Failed", "2026-09-14T09:00:00Z", "Safe failure."),
      ],
      0,
    ),
    false,
  );
  assert.equal(
    shouldPollDocuments(
      [documentItem("processing", "Processing", "2026-09-14T10:00:00Z")],
      DOCUMENT_POLL_MAX_ATTEMPTS,
    ),
    false,
  );
});

test("document refresh responses are sorted newest first with stable ID ordering", () => {
  const sorted = parseAndSortDocuments([
    documentItem("older", "Ready", "2026-09-14T09:00:00Z"),
    documentItem("same-a", "Processing", "2026-09-14T11:00:00Z"),
    documentItem("newest", "Uploaded", "2026-09-14T12:00:00Z"),
    documentItem("same-b", "Failed", "2026-09-14T11:00:00Z", "Safe failure."),
  ]);

  assert.deepEqual(
    sorted.map((document) => document.id),
    ["newest", "same-b", "same-a", "older"],
  );
});

test("internal processing states are rejected rather than rendered", () => {
  const internalDocument = {
    ...documentItem("internal", "Processing", "2026-09-14T10:00:00Z"),
    status: "CleanupPending",
  };

  assert.throws(
    () => parseAndSortDocuments([internalDocument]),
    /Document status is unavailable/,
  );
});

test("status presentation explains public states and uses only the supplied failure message", () => {
  assert.deepEqual(getDocumentStatusPresentation("Uploaded", null), {
    label: "Uploaded",
    description: "Accepted and queued. Not searchable yet.",
    tone: "uploaded",
  });
  assert.deepEqual(getDocumentStatusPresentation("Processing", null), {
    label: "Processing",
    description: "Being prepared for retrieval. Not searchable yet.",
    tone: "processing",
  });
  assert.deepEqual(getDocumentStatusPresentation("Ready", null), {
    label: "Ready",
    description: "Available for future retrieval and AI answers.",
    tone: "ready",
  });
  assert.deepEqual(getDocumentStatusPresentation("Failed", "Safe backend message."), {
    label: "Failed",
    description: "Safe backend message.",
    tone: "failed",
  });
  assert.deepEqual(getDocumentStatusPresentation("Failed", null), {
    label: "Failed",
    description: null,
    tone: "failed",
  });
});

test("polling waits for each refresh and stops after a terminal response", async () => {
  const scheduled: Array<() => Promise<void>> = [];
  const delays: number[] = [];
  const responses = [
    [documentItem("document", "Processing", "2026-09-14T10:00:00Z")],
    [documentItem("document", "Ready", "2026-09-14T10:00:00Z")],
  ];
  let releaseFirstRefresh: () => void = () => assert.fail("Refresh resolver was not set.");
  let refreshCalls = 0;

  startDocumentPolling({
    initialDocuments: [documentItem("document", "Uploaded", "2026-09-14T10:00:00Z")],
    refresh: async () => {
      const response = responses[refreshCalls];
      refreshCalls += 1;
      if (refreshCalls === 1) {
        await new Promise<void>((resolve) => {
          releaseFirstRefresh = resolve;
        });
      }
      return response;
    },
    schedule: (callback, delay) => {
      scheduled.push(callback);
      delays.push(delay);
      return scheduled.length;
    },
    cancel: () => undefined,
  });

  assert.equal(scheduled.length, 1);
  const firstRefresh = scheduled.shift();
  assert.ok(firstRefresh);
  const firstRefreshPromise = firstRefresh();
  assert.equal(refreshCalls, 1);
  assert.equal(scheduled.length, 0);
  releaseFirstRefresh();
  await firstRefreshPromise;

  assert.equal(scheduled.length, 1);
  const secondRefresh = scheduled.shift();
  assert.ok(secondRefresh);
  await secondRefresh();

  assert.equal(refreshCalls, 2);
  assert.equal(scheduled.length, 0);
  assert.deepEqual(delays, [2_000, 2_000]);
});

test("stopping polling cancels the pending timer", () => {
  const cancelledTokens: unknown[] = [];

  const stop = startDocumentPolling({
    initialDocuments: [documentItem("document", "Processing", "2026-09-14T10:00:00Z")],
    refresh: async () => [],
    schedule: () => "timer-token",
    cancel: (token) => cancelledTokens.push(token),
  });

  stop();

  assert.deepEqual(cancelledTokens, ["timer-token"]);
});
