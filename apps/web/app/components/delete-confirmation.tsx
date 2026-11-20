"use client";

import { useEffect, useId, useRef } from "react";

type DeleteConfirmationProps = {
  title: string;
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
};

export function DeleteConfirmation({
  title, message, confirmLabel, onConfirm, onCancel,
}: DeleteConfirmationProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const settled = useRef(false);
  const id = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousFocus = document.activeElement;
    if (!dialog) return;
    dialog.showModal();
    cancelRef.current?.focus();
    return () => {
      dialog.close();
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) {
        previousFocus.focus();
      }
    };
  }, []);

  function finish(confirmed: boolean) {
    if (settled.current) return;
    settled.current = true;
    if (confirmed) onConfirm();
    else onCancel();
  }

  return (
    <dialog
      ref={dialogRef}
      className="delete-confirmation"
      aria-modal="true"
      aria-labelledby={`${id}-title`}
      aria-describedby={`${id}-message`}
      onCancel={(event) => {
        event.preventDefault();
        finish(false);
      }}
      onKeyDown={(event) => {
        if (event.key !== "Tab") return;
        if (event.shiftKey && document.activeElement === cancelRef.current) {
          event.preventDefault();
          confirmRef.current?.focus();
        } else if (!event.shiftKey && document.activeElement === confirmRef.current) {
          event.preventDefault();
          cancelRef.current?.focus();
        }
      }}
      onPointerDown={(event) => {
        if (event.target !== event.currentTarget) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
            event.clientY < bounds.top || event.clientY > bounds.bottom) {
          event.preventDefault();
          finish(false);
        }
      }}
    >
      <p className="delete-confirmation-eyebrow">Permanent deletion</p>
      <h2 id={`${id}-title`}>{title}</h2>
      <p id={`${id}-message`} className="delete-confirmation-message">{message}</p>
      <div className="delete-confirmation-actions">
        <button ref={cancelRef} type="button" className="cancel-button"
          onClick={() => finish(false)}>
          Cancel
        </button>
        <button ref={confirmRef} type="button" className="danger-button"
          onClick={() => finish(true)}>
          {confirmLabel}
        </button>
      </div>
    </dialog>
  );
}
