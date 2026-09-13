import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./styles.css";

export const metadata: Metadata = {
  title: "AI Knowledge Workspace",
  description: "Ask questions grounded in your own documents.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <Link className="brand" href="/">AI Knowledge Workspace</Link>
          <nav aria-label="Main navigation">
            <Link href="/knowledge-bases">Knowledge Bases</Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
