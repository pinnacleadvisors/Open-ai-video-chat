import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "open-ai-video-chat",
  description: "Self-hosted, open-source AI avatar for live video calls",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
