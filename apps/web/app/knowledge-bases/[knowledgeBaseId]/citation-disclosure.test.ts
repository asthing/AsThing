import assert from "node:assert/strict";
import test from "node:test";

import {
  clearCitationDisclosure,
  citationDisclosureFor,
  toggleCitationDisclosure,
} from "./citation-disclosure.ts";

const referencedAnswer = {
  outcome: "answer" as const,
  answer: "Use the HR portal.",
  reason: null,
  message: null,
  references: [
    {
      reference_key: "source-1",
      source_filename: "policy.txt",
      file_type: "txt",
      excerpt: "Annual leave is requested in the HR portal.",
    },
  ],
};

test("a referenced answer starts with a collapsed citation disclosure", () => {
  const disclosure = citationDisclosureFor(referencedAnswer);

  assert.equal(disclosure.isExpanded, false);
  assert.deepEqual(disclosure.references, referencedAnswer.references);
});

test("no-answer and empty references render no citation disclosure", () => {
  const noAnswer = citationDisclosureFor({
    outcome: "no_answer" as const,
    answer: null,
    reason: "insufficient_evidence" as const,
    message: "No answer.",
    references: [],
  });
  const emptyReferences = citationDisclosureFor({ ...referencedAnswer, references: [] });

  assert.deepEqual(noAnswer, { isExpanded: false, references: [] });
  assert.deepEqual(emptyReferences, { isExpanded: false, references: [] });
});

test("starting a new request clears previous citations before a result arrives", () => {
  const previous = citationDisclosureFor(referencedAnswer);

  assert.equal(previous.references.length, 1);
  assert.deepEqual(clearCitationDisclosure(), { isExpanded: false, references: [] });
});

test("a failed request clears prior citations", () => {
  const previous = citationDisclosureFor(referencedAnswer);
  const failedRequest = clearCitationDisclosure();

  assert.equal(previous.references.length, 1);
  assert.equal(failedRequest.references.length, 0);
  assert.equal(failedRequest.isExpanded, false);
});

test("a Retry replaces citations with only its successful response references", () => {
  const retriedAnswer = {
    ...referencedAnswer,
    references: [
      {
        reference_key: "source-1",
        source_filename: "handbook.pdf",
        file_type: "pdf",
        excerpt: "Requests require manager approval.",
      },
    ],
  };
  const afterFailure = clearCitationDisclosure();
  const afterRetry = citationDisclosureFor(retriedAnswer);

  assert.equal(afterFailure.references.length, 0);
  assert.deepEqual(afterRetry.references, retriedAnswer.references);
  assert.notDeepEqual(afterRetry.references, referencedAnswer.references);
});

test("the citation disclosure toggles only when references are present", () => {
  const collapsed = citationDisclosureFor(referencedAnswer);
  const expanded = toggleCitationDisclosure(collapsed);

  assert.equal(expanded.isExpanded, true);
  assert.equal(toggleCitationDisclosure(expanded).isExpanded, false);
  assert.deepEqual(
    toggleCitationDisclosure(clearCitationDisclosure()),
    { isExpanded: false, references: [] },
  );
});
