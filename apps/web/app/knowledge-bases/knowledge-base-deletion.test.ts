import assert from "node:assert/strict";
import test from "node:test";
import {
  createKnowledgeBaseDeletionManager,
  knowledgeBaseDeletionConfirmation,
} from "./knowledge-base-deletion.ts";

const target = { id: "target", name: "Engineering notes" };
const keeper = { id: "keeper", name: "Other knowledge base" };

test("confirmation names knowledge base and permanently deletes all documents", () => {
  const message = knowledgeBaseDeletionConfirmation(target.name);
  assert.ok(message.includes(target.name));
  assert.ok(message.includes("all its documents"));
  assert.ok(message.includes("cannot be undone"));
});

test("cancellation never sends DELETE or changes the list", async () => {
  const manager = createKnowledgeBaseDeletionManager();
  let calls = 0;
  const result = await manager.submit(target, () => false, async () => { calls++; });
  assert.equal(result.cancelled, true);
  assert.equal(calls, 0);
  assert.deepEqual(manager.filter([target, keeper]), [target, keeper]);
});

test("successful removal filters stale lists and preserves remaining order", async () => {
  const manager = createKnowledgeBaseDeletionManager();
  await manager.submit(target, () => true, async id => { assert.equal(id, target.id); });
  assert.deepEqual(manager.filter([target, keeper]), [keeper]);
  assert.deepEqual(manager.filter([target]), []);
});

test("failure retains row; Retry only removes it after its own success", async () => {
  const manager = createKnowledgeBaseDeletionManager();
  const failure = new Error("Safe cleanup failure.");
  const result = await manager.submit(target, () => true, async () => { throw failure; });
  assert.equal(result.error, failure);
  assert.deepEqual(manager.filter([target]), [target]);
  const retry = await manager.submit(target, () => true, async () => {});
  assert.equal(retry.error, null);
  assert.deepEqual(manager.filter([target]), []);
});

test("duplicate submission is blocked while deletion is in progress", async () => {
  const manager = createKnowledgeBaseDeletionManager();
  let resolve!: () => void;
  const first = manager.submit(target, () => true, () => new Promise<void>(r => { resolve = r; }));
  await assert.rejects(manager.submit(keeper, () => true, async () => {}), /already in progress/);
  resolve();
  assert.equal((await first).current, true);
});

test("unmount aborts and rejects stale completion state", async () => {
  const manager = createKnowledgeBaseDeletionManager();
  let resolve!: () => void;
  let signal!: AbortSignal;
  const pending = manager.submit(target, () => true, (_id, currentSignal) => {
    signal = currentSignal;
    return new Promise<void>(r => { resolve = r; });
  });
  manager.cancel();
  assert.equal(signal.aborted, true);
  resolve();
  assert.equal((await pending).current, false);
  assert.deepEqual(manager.filter([target]), [target]);
});
