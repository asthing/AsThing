import assert from "node:assert/strict";
import test from "node:test";

import { createDocumentDeletionManager, deletionConfirmation } from "./document-deletion.ts";
import { shouldPollDocuments, type DocumentListItem } from "./document-status.ts";

const document: DocumentListItem = {
  id: "document", knowledge_base_id: "kb", original_filename: "same.txt", file_type: "txt",
  file_size_bytes: 10, status: "Ready", uploaded_at: "2026-11-07T00:00:00Z", failure_message: null,
};

test("confirmation names the file, warns permanently, and cancellation sends nothing", async () => {
  const manager = createDocumentDeletionManager("kb");
  const result = await manager.submit(document, (message) => {
    assert.equal(message, deletionConfirmation("same.txt"));
    assert.match(message, /Permanently delete/);
    assert.match(message, /cannot be undone/);
    return false;
  }, async () => assert.fail("cancelled deletion sent a request"));
  assert.equal(result.cancelled, true);
  assert.deepEqual(manager.filterDocuments([document]), [document]);
});

test("only confirmed completion removes a row and late lists cannot restore it", async () => {
  const manager = createDocumentDeletionManager("kb");
  const retained = { ...document, id: "retained" };
  const failed = await manager.submit(document, () => true, async () => { throw new Error("safe"); });
  assert.ok(failed.error);
  assert.deepEqual(manager.filterDocuments([retained, document]), [retained, document]);
  const success = await manager.submit(document, () => true, async () => undefined);
  assert.equal(success.current, true);
  assert.deepEqual(manager.filterDocuments([retained, document]), [retained]);
  assert.deepEqual(manager.filterDocuments([document]), []);
});

test("duplicate submissions are prevented; pending retry does not reconfirm", async () => {
  const manager = createDocumentDeletionManager("kb");
  let resolve: () => void = () => assert.fail("no pending request");
  const first = manager.submit({ ...document, deletion_pending: true },
    () => assert.fail("already accepted deletion must not reconfirm"),
    () => new Promise<void>((done) => { resolve = done; }));
  await assert.rejects(manager.submit(document, () => true, async () => undefined), /in progress/);
  resolve();
  assert.equal((await first).error, null);
});

test("unmount/knowledge-base cancellation aborts and ignores stale completion", async () => {
  const manager = createDocumentDeletionManager("kb");
  let signal: AbortSignal | undefined;
  let resolve: () => void = () => assert.fail("no pending request");
  const first = manager.submit(document, () => true, (_id, suppliedSignal) => {
    signal = suppliedSignal;
    return new Promise<void>((done) => { resolve = done; });
  });
  manager.cancel();
  resolve();
  assert.equal((await first).current, false);
  assert.ok(signal?.aborted);
  assert.deepEqual(manager.filterDocuments([document]), [document]);
  await assert.rejects(createDocumentDeletionManager("other").submit(
    document, () => true, async () => assert.fail("wrong knowledge base"),
  ), /reload this knowledge base/);
});

test("pending deletion joins existing bounded polling even for Ready/Failed", () => {
  assert.equal(shouldPollDocuments([{ ...document, deletion_pending: true }], 0), true);
  assert.equal(shouldPollDocuments([{ ...document, status: "Failed", deletion_pending: true }], 119), true);
  assert.equal(shouldPollDocuments([{ ...document, deletion_pending: true }], 120), false);
  assert.equal(shouldPollDocuments([document], 0), false);
});
