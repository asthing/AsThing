type KnowledgeBase = { id: string; name: string };

export function knowledgeBaseDeletionConfirmation(name: string): string {
  return `Permanently delete "${name}" and all its documents? Original files and source content will be removed. This cannot be undone.`;
}

export function createKnowledgeBaseDeletionManager() {
  let active: AbortController | null = null;
  let generation = 0;
  const deletedIds = new Set<string>();
  return {
    async submit(
      knowledgeBase: KnowledgeBase,
      confirm: (message: string) => boolean,
      fetcher: (id: string, signal: AbortSignal) => Promise<void>,
    ) {
      if (active) throw new Error("A knowledge base deletion is already in progress.");
      if (!confirm(knowledgeBaseDeletionConfirmation(knowledgeBase.name))) {
        return { current: true, cancelled: true, error: null };
      }
      const controller = new AbortController();
      active = controller;
      const currentGeneration = ++generation;
      try {
        await fetcher(knowledgeBase.id, controller.signal);
        const current = currentGeneration === generation;
        if (current) deletedIds.add(knowledgeBase.id);
        return { current, cancelled: false, error: null };
      } catch (error: unknown) {
        return { current: currentGeneration === generation, cancelled: false, error };
      } finally {
        if (currentGeneration === generation) active = null;
      }
    },
    filter<T extends { id: string }>(items: T[]): T[] {
      return items.filter((item) => !deletedIds.has(item.id));
    },
    cancel() {
      ++generation;
      active?.abort();
      active = null;
    },
  };
}
