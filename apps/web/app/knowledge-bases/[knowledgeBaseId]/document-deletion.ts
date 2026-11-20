import type { DocumentListItem } from "./document-status.ts";

type Fetcher = (documentId: string, signal: AbortSignal) => Promise<void>;

export function deletionConfirmation(filename: string): string {
  return `Permanently delete "${filename}"? Its original file and source content will be removed. This cannot be undone.`;
}

export function createDocumentDeletionManager(knowledgeBaseId: string) {
  let active: AbortController | null = null;
  let generation = 0;
  const deletedIds = new Set<string>();
  return {
    async submit(
      document: DocumentListItem,
      confirm: (message: string) => boolean,
      fetcher: Fetcher,
    ) {
      if (active) throw new Error("A document deletion is already in progress.");
      if (document.knowledge_base_id !== knowledgeBaseId) {
        throw new Error("Document data is unavailable. Please reload this knowledge base.");
      }
      if (!document.deletion_pending && !confirm(deletionConfirmation(document.original_filename))) {
        return { current: true, cancelled: true, error: null };
      }
      const controller = new AbortController();
      active = controller;
      const currentGeneration = ++generation;
      try {
        await fetcher(document.id, controller.signal);
        const current = currentGeneration === generation;
        if (current) deletedIds.add(document.id);
        return { current, cancelled: false, error: null };
      } catch (error: unknown) {
        return { current: currentGeneration === generation, cancelled: false, error };
      } finally {
        if (currentGeneration === generation) active = null;
      }
    },
    filterDocuments(documents: DocumentListItem[]) {
      return documents.filter((document) => !deletedIds.has(document.id));
    },
    cancel() {
      ++generation;
      active?.abort();
      active = null;
    },
  };
}
