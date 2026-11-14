import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cursor Fleet Hub — Multi-Session tmux & Cloud Command Center",
  description:
    "Control local tmux Cursor CLI sessions and Cursor Cloud Agents from browser or phone with zero-polling push and permanent Postgres memory.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#090b10",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-dvh bg-[#090b10] text-zinc-100 selection:bg-indigo-500/30">
        {children}
      </body>
    </html>
  );
}
