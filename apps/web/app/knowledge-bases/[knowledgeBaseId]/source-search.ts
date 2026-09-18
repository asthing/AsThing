export type SourceSearchOutcome = "results" | "no_ready_documents" | "no_results";

export type SourceSearchResult = {
  source_filename: string;
  file_type: string;
  excerpt: string;
};

export type SourceSearchResponse = {
  outcome: SourceSearchOutcome;
  results: SourceSearchResult[];
};

export type SourceSearchFetcher = (
  question: string,
  signal: AbortSignal,
) => Promise<unknown>;

export type SourceSearchRequestResult = {
  current: boolean;
  response?: SourceSearchResponse;
  error?: unknown;
};

export class SourceSearchValidationError extends Error {
  constructor(message = "Enter a question before searching.") {
    super(message);
    this.name = "SourceSearchValidationError";
  }
}

const outcomes = new Set<SourceSearchOutcome>([
  "results",
  "no_ready_documents",
  "no_results",
]);
const maxResults = 5;
const maxExcerptCharacters = 500;

export function parseSourceSearchResponse(value: unknown): SourceSearchResponse {
  if (!isRecord(value) || !outcomes.has(value.outcome as SourceSearchOutcome) || !Array.isArray(value.results)) {
    throw new Error("Source search data is unavailable. Please try again.");
  }

  const results = value.results.map(parseSourceSearchResult);
  const outcome = value.outcome as SourceSearchOutcome;
  if (
    (outcome === "results" && (results.length === 0 || results.length > maxResults)) ||
    (outcome !== "results" && results.length !== 0)
  ) {
    throw new Error("Source search data is unavailable. Please try again.");
  }
  return { outcome, results };
}

export function createSourceSearchRequestManager() {
  let activeController: AbortController | null = null;
  let requestId = 0;
  let lastQuestion: string | null = null;

  async function execute(
    question: string,
    fetcher: SourceSearchFetcher,
  ): Promise<SourceSearchRequestResult> {
    activeController?.abort();
    const controller = new AbortController();
    const currentRequestId = requestId + 1;
    requestId = currentRequestId;
    activeController = controller;

    try {
      const response = parseSourceSearchResponse(await fetcher(question, controller.signal));
      return {
        current: currentRequestId === requestId,
        response,
      };
    } catch (error: unknown) {
      return {
        current: currentRequestId === requestId,
        error,
      };
    } finally {
      if (currentRequestId === requestId) activeController = null;
    }
  }

  return {
    async submit(rawQuestion: string, fetcher: SourceSearchFetcher): Promise<SourceSearchRequestResult> {
      const question = rawQuestion.trim();
      if (!question) throw new SourceSearchValidationError();
      lastQuestion = question;
      return execute(question, fetcher);
    },
    async retry(fetcher: SourceSearchFetcher): Promise<SourceSearchRequestResult> {
      if (!lastQuestion) throw new SourceSearchValidationError("Enter a question before retrying.");
      return execute(lastQuestion, fetcher);
    },
    cancel(): void {
      requestId += 1;
      activeController?.abort();
      activeController = null;
    },
  };
}

function parseSourceSearchResult(value: unknown): SourceSearchResult {
  if (
    !isRecord(value) ||
    typeof value.source_filename !== "string" ||
    typeof value.file_type !== "string" ||
    typeof value.excerpt !== "string" ||
    !value.excerpt ||
    value.excerpt.length > maxExcerptCharacters
  ) {
    throw new Error("Source search data is unavailable. Please try again.");
  }
  return {
    source_filename: value.source_filename,
    file_type: value.file_type,
    excerpt: value.excerpt,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
