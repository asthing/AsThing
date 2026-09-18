export type AnswerReference = {
  reference_key: string;
  source_filename: string;
  file_type: string;
  excerpt: string;
};

export type AnswerResponse =
  | {
      outcome: "answer";
      answer: string;
      reason: null;
      message: null;
      references: AnswerReference[];
    }
  | {
      outcome: "no_answer";
      answer: null;
      reason: "no_ready_documents" | "insufficient_evidence";
      message: string;
      references: [];
    };

type Fetcher = (question: string, signal: AbortSignal) => Promise<unknown>;

export class AnswerValidationError extends Error {
  constructor(message = "Enter a question before asking for an answer.") {
    super(message);
    this.name = "AnswerValidationError";
  }
}

export class AnswerBusyError extends Error {
  constructor() {
    super("An answer is already being generated.");
    this.name = "AnswerBusyError";
  }
}

export function parseAnswerResponse(value: unknown): AnswerResponse {
  if (!record(value) || !Array.isArray(value.references)) {
    throw new Error("Answer data is unavailable. Please try again.");
  }
  if (value.outcome === "no_answer") {
    if (
      value.answer !== null ||
      value.references.length !== 0 ||
      !["no_ready_documents", "insufficient_evidence"].includes(String(value.reason)) ||
      typeof value.message !== "string" ||
      !value.message
    ) throw new Error("Answer data is unavailable. Please try again.");
    return value as AnswerResponse;
  }
  if (
    value.outcome !== "answer" ||
    typeof value.answer !== "string" ||
    !value.answer.trim() ||
    value.answer.length > 700 ||
    value.reason !== null ||
    value.message !== null ||
    value.references.length < 1 ||
    value.references.length > 5 ||
    value.references.some(
      (reference: unknown) =>
        !record(reference) ||
        typeof reference.reference_key !== "string" ||
        typeof reference.source_filename !== "string" ||
        typeof reference.file_type !== "string" ||
        typeof reference.excerpt !== "string" ||
        reference.excerpt.length > 500,
    )
  ) throw new Error("Answer data is unavailable. Please try again.");
  return value as AnswerResponse;
}

export function createAnswerRequestManager() {
  let activeController: AbortController | null = null;
  let requestId = 0;
  let lastQuestion: string | null = null;

  async function execute(question: string, fetcher: Fetcher) {
    if (activeController) throw new AnswerBusyError();
    const controller = new AbortController();
    const currentId = ++requestId;
    activeController = controller;
    try {
      const response = parseAnswerResponse(await fetcher(question, controller.signal));
      return { current: currentId === requestId, response, error: null };
    } catch (error: unknown) {
      return { current: currentId === requestId, response: null, error };
    } finally {
      if (currentId === requestId) activeController = null;
    }
  }

  return {
    submit(question: string, fetcher: Fetcher) {
      if (activeController) throw new AnswerBusyError();
      const trimmed = question.trim();
      if (!trimmed) throw new AnswerValidationError();
      lastQuestion = trimmed;
      return execute(trimmed, fetcher);
    },
    retry(fetcher: Fetcher) {
      if (!lastQuestion) throw new AnswerValidationError();
      return execute(lastQuestion, fetcher);
    },
    cancel() {
      requestId += 1;
      activeController?.abort();
      activeController = null;
    },
  };
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
