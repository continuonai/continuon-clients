# Plan 6: Web UX Dashboard

## Overview
Create the frontend dashboard for continuonai.com - React-based fleet management and training UI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUONAI.COM FRONTEND                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Next.js Application                    │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  Pages:                                                   │   │
│  │  ├── /                    Landing page                   │   │
│  │  ├── /dashboard           Main dashboard                 │   │
│  │  ├── /robots              Fleet management               │   │
│  │  ├── /robots/[id]         Robot detail                   │   │
│  │  ├── /models              Model browser                  │   │
│  │  ├── /training            Training jobs                  │   │
│  │  ├── /training/[id]       Job detail                     │   │
│  │  └── /settings            User settings                  │   │
│  │                                                           │   │
│  │  Components:                                              │   │
│  │  ├── RobotCard            Robot status card              │   │
│  │  ├── ModelCard            Model info card                │   │
│  │  ├── TrainingProgress     Training progress bar          │   │
│  │  ├── LiveFeed             Camera/telemetry feed          │   │
│  │  └── Charts               Loss/performance charts        │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                    ContinuonAI Backend API                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Project Structure
```
continuonai-web/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx              # Landing
│   │   ├── dashboard/
│   │   │   └── page.tsx          # Main dashboard
│   │   ├── robots/
│   │   │   ├── page.tsx          # Robot list
│   │   │   └── [id]/
│   │   │       └── page.tsx      # Robot detail
│   │   ├── models/
│   │   │   └── page.tsx          # Model browser
│   │   ├── training/
│   │   │   ├── page.tsx          # Training jobs
│   │   │   └── [id]/
│   │   │       └── page.tsx      # Job detail
│   │   └── settings/
│   │       └── page.tsx
│   ├── components/
│   │   ├── ui/                   # shadcn/ui components
│   │   ├── RobotCard.tsx
│   │   ├── RobotStatus.tsx
│   │   ├── ModelCard.tsx
│   │   ├── TrainingProgress.tsx
│   │   ├── LiveFeed.tsx
│   │   ├── LossChart.tsx
│   │   └── Navbar.tsx
│   ├── lib/
│   │   ├── api.ts               # API client
│   │   ├── auth.ts              # Firebase auth
│   │   └── utils.ts
│   ├── hooks/
│   │   ├── useRobots.ts
│   │   ├── useModels.ts
│   │   └── useTraining.ts
│   └── types/
│       └── index.ts
├── public/
├── package.json
├── tailwind.config.js
└── next.config.js
```

### Key Components

#### 1. Dashboard Page
```tsx
// src/app/dashboard/page.tsx
'use client';

import { useRobots } from '@/hooks/useRobots';
import { useTraining } from '@/hooks/useTraining';
import { RobotCard } from '@/components/RobotCard';
import { TrainingProgress } from '@/components/TrainingProgress';

export default function Dashboard() {
  const { robots, isLoading: robotsLoading } = useRobots();
  const { activeJobs } = useTraining();

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>

      {/* Fleet Overview */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Your Robots</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {robots?.map(robot => (
            <RobotCard key={robot.id} robot={robot} />
          ))}
        </div>
      </section>

      {/* Active Training */}
      {activeJobs?.length > 0 && (
        <section className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Active Training</h2>
          {activeJobs.map(job => (
            <TrainingProgress key={job.id} job={job} />
          ))}
        </section>
      )}

      {/* Quick Actions */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
        <div className="flex gap-4">
          <Button>Upload Episodes</Button>
          <Button>Start Training</Button>
          <Button>Browse Models</Button>
        </div>
      </section>
    </div>
  );
}
```

#### 2. Robot Card Component
```tsx
// src/components/RobotCard.tsx
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Robot } from '@/types';
import Link from 'next/link';

interface RobotCardProps {
  robot: Robot;
}

export function RobotCard({ robot }: RobotCardProps) {
  const statusColor = {
    online: 'bg-green-500',
    offline: 'bg-gray-500',
    training: 'bg-blue-500',
    error: 'bg-red-500',
  }[robot.status] || 'bg-gray-500';

  return (
    <Link href={`/robots/${robot.id}`}>
      <Card className="hover:shadow-lg transition-shadow cursor-pointer">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <h3 className="font-semibold">{robot.name}</h3>
            <p className="text-sm text-gray-500">{robot.device_id}</p>
          </div>
          <Badge className={statusColor}>{robot.status}</Badge>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">Model:</span>
              <span className="ml-2">{robot.model_version}</span>
            </div>
            <div>
              <span className="text-gray-500">Last seen:</span>
              <span className="ml-2">{formatTime(robot.last_seen)}</span>
            </div>
            <div>
              <span className="text-gray-500">Battery:</span>
              <span className="ml-2">{robot.battery}%</span>
            </div>
            <div>
              <span className="text-gray-500">Episodes:</span>
              <span className="ml-2">{robot.episode_count}</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
```

#### 3. Training Progress Component
```tsx
// src/components/TrainingProgress.tsx
import { Progress } from '@/components/ui/progress';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { TrainingJob } from '@/types';
import { LossChart } from './LossChart';

interface TrainingProgressProps {
  job: TrainingJob;
}

export function TrainingProgress({ job }: TrainingProgressProps) {
  const progress = (job.steps_completed / job.total_steps) * 100;

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="font-semibold">{job.model_id}</h3>
          <Badge>{job.status}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Progress</span>
              <span>{job.steps_completed} / {job.total_steps} steps</span>
            </div>
            <Progress value={progress} />
          </div>

          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Current Loss:</span>
              <span className="ml-2 font-mono">{job.current_loss?.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-gray-500">Best Loss:</span>
              <span className="ml-2 font-mono">{job.best_loss?.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-gray-500">ETA:</span>
              <span className="ml-2">{job.eta}</span>
            </div>
          </div>

          {job.loss_history && (
            <LossChart data={job.loss_history} />
          )}
        </div>
      </CardContent>
    </Card>
  );
}
```

#### 4. API Client
```tsx
// src/lib/api.ts
import { auth } from './auth';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.continuonai.com';

async function fetchWithAuth(url: string, options: RequestInit = {}) {
  const token = await auth.currentUser?.getIdToken();

  return fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });
}

export const api = {
  robots: {
    list: () => fetchWithAuth('/api/v1/robots').then(r => r.json()),
    get: (id: string) => fetchWithAuth(`/api/v1/robots/${id}`).then(r => r.json()),
    sendCommand: (id: string, cmd: any) =>
      fetchWithAuth(`/api/v1/robots/${id}/command`, {
        method: 'POST',
        body: JSON.stringify(cmd),
      }).then(r => r.json()),
  },

  models: {
    list: () => fetchWithAuth('/api/v1/models').then(r => r.json()),
    getDownloadUrl: (id: string, version: string) =>
      fetchWithAuth(`/api/v1/models/${id}/${version}/download`).then(r => r.json()),
  },

  training: {
    list: () => fetchWithAuth('/api/v1/training/jobs').then(r => r.json()),
    create: (config: any) =>
      fetchWithAuth('/api/v1/training/jobs', {
        method: 'POST',
        body: JSON.stringify(config),
      }).then(r => r.json()),
    cancel: (id: string) =>
      fetchWithAuth(`/api/v1/training/jobs/${id}/cancel`, { method: 'POST' }),
  },
};
```

### Real-time Updates (Firestore)
```tsx
// src/hooks/useRobotStatus.ts
import { useEffect, useState } from 'react';
import { doc, onSnapshot } from 'firebase/firestore';
import { db } from '@/lib/firebase';

export function useRobotStatus(robotId: string) {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const unsubscribe = onSnapshot(
      doc(db, 'robots', robotId),
      (doc) => setStatus(doc.data())
    );
    return unsubscribe;
  }, [robotId]);

  return status;
}
```

## Files to Create
| File | Description |
|------|-------------|
| `continuonai-web/` | New Next.js project |
| All files in structure above | React components and pages |

## Tech Stack
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Firebase Auth
- Firestore (real-time)
- Recharts (charts)

## Deployment
```bash
# Vercel deployment
vercel --prod
```

## Success Criteria
- User can log in
- Dashboard shows robot fleet
- Can browse and download models
- Can monitor training progress
- Real-time status updates
