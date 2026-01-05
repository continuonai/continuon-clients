import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ContinuonAI - Robot Fleet Management & Training',
  description: 'Manage your robot fleet, train models, and deploy AI-powered robotics solutions.',
  keywords: ['robotics', 'AI', 'machine learning', 'robot training', 'fleet management'],
  authors: [{ name: 'ContinuonAI Team' }],
  openGraph: {
    title: 'ContinuonAI',
    description: 'Robot Fleet Management & Training Platform',
    url: 'https://continuonai.com',
    siteName: 'ContinuonAI',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <div className="min-h-screen bg-background">
          {children}
        </div>
      </body>
    </html>
  );
}
