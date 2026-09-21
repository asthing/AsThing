import type { AnswerReference, AnswerResponse } from "./answer-request";

export type CitationDisclosure = {
  isExpanded: boolean;
  references: AnswerReference[];
};

export function citationDisclosureFor(answer: AnswerResponse | null): CitationDisclosure {
  if (answer?.outcome !== "answer" || answer.references.length === 0) {
    return clearCitationDisclosure();
  }
  return { isExpanded: false, references: answer.references };
}

export function clearCitationDisclosure(): CitationDisclosure {
  return { isExpanded: false, references: [] };
}

export function toggleCitationDisclosure(disclosure: CitationDisclosure): CitationDisclosure {
  if (disclosure.references.length === 0) return disclosure;
  return { ...disclosure, isExpanded: !disclosure.isExpanded };
}
