'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { useTrainingJob } from '@/hooks/useTraining';
import { LossChart } from '@/components/LossChart';
import { Navbar } from '@/components/Navbar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { getStatusColor, formatTime, formatDuration } from '@/lib/utils';
import {
  ArrowLeft,
  Play,
  Square,
  RefreshCw,
  Download,
  Clock,
  TrendingDown,
  Award,
  Timer,
  Settings,
  FileText,
  GraduationCap,
} from 'lucide-react';

export default function TrainingJobDetailPage() {
  const params = useParams();
  const jobId = params.id as string;
  const { job, progress, eta, isLoading, cancel, getLogs, mutate } = useTrainingJob(jobId);
  const [logs, setLogs] = useState<string[]>([]);
  const [showLogs, setShowLogs] = useState(false);

  useEffect(() => {
    if (showLogs && job) {
      getLogs().then(setLogs).catch(console.error);
    }
  }, [showLogs, job, getLogs]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-muted rounded w-1/4" />
            <div className="h-4 bg-muted rounded w-1/2" />
            <div className="h-64 bg-muted rounded mt-8" />
          </div>
        </main>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Card className="max-w-md mx-auto">
            <CardContent className="p-8 text-center">
              <GraduationCap className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h2 className="text-xl font-semibold mb-2">Training Job Not Found</h2>
              <p className="text-muted-foreground mb-4">
                The training job you are looking for does not exist.
              </p>
              <Link href="/training">
                <Button>
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Training
                </Button>
              </Link>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  const statusColor = getStatusColor(job.status);
  const isRunning = job.status === 'running';
  const isQueued = job.status === 'queued';
  const isCompleted = job.status === 'completed';
  const isFailed = job.status === 'failed';

  const elapsedTime = job.started_at
    ? Math.floor((Date.now() - new Date(job.started_at).getTime()) / 1000)
    : 0;

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Link href="/training">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div className="flex-1">
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold">{job.model_id}</h1>
              <Badge className={statusColor}>
                {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
              </Badge>
              {isRunning && (
                <Play className="h-5 w-5 text-blue-500 animate-pulse" />
              )}
            </div>
            <p className="text-muted-foreground">
              Training Job ID: <span className="font-mono">{job.id}</span>
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={() => mutate()}>
              <RefreshCw className="h-4 w-4" />
            </Button>
            {(isRunning || isQueued) && (
              <Button variant="destructive" onClick={cancel}>
                <Square className="h-4 w-4 mr-2" />
                Cancel
              </Button>
            )}
            {isCompleted && (
              <Button>
                <Download className="h-4 w-4 mr-2" />
                Download Model
              </Button>
            )}
          </div>
        </div>

        {/* Progress Overview */}
        <Card className="mb-8">
          <CardContent className="p-6">
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="font-medium">Training Progress</span>
                <span className="font-mono">
                  {job.steps_completed.toLocaleString()} / {job.total_steps.toLocaleString()} steps
                </span>
              </div>
              <Progress value={progress} className="h-4" />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>{progress.toFixed(1)}% complete</span>
                <span>ETA: {eta}</span>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <MetricCard
                icon={TrendingDown}
                label="Current Loss"
                value={job.current_loss?.toFixed(4) || 'N/A'}
                color="text-blue-500"
              />
              <MetricCard
                icon={Award}
                label="Best Loss"
                value={job.best_loss?.toFixed(4) || 'N/A'}
                color="text-green-500"
              />
              <MetricCard
                icon={Timer}
                label="ETA"
                value={eta}
                color="text-orange-500"
              />
              <MetricCard
                icon={Clock}
                label="Elapsed"
                value={formatDuration(elapsedTime)}
                color="text-purple-500"
              />
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Loss Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingDown className="h-5 w-5" />
                  Loss Curve
                </CardTitle>
              </CardHeader>
              <CardContent>
                {job.loss_history && job.loss_history.length > 0 ? (
                  <LossChart data={job.loss_history} height={300} />
                ) : (
                  <div className="h-[300px] flex items-center justify-center bg-muted rounded-lg">
                    <p className="text-muted-foreground">
                      Loss data will appear here as training progresses
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Logs */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Training Logs
                  </CardTitle>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowLogs(!showLogs)}
                  >
                    {showLogs ? 'Hide Logs' : 'Show Logs'}
                  </Button>
                </div>
              </CardHeader>
              {showLogs && (
                <CardContent>
                  <div className="bg-muted rounded-lg p-4 max-h-[400px] overflow-auto font-mono text-sm">
                    {logs.length > 0 ? (
                      logs.map((log, i) => (
                        <div key={i} className="py-1 border-b border-border last:border-0">
                          {log}
                        </div>
                      ))
                    ) : (
                      <p className="text-muted-foreground">No logs available</p>
                    )}
                  </div>
                </CardContent>
              )}
            </Card>

            {/* Error Display */}
            {isFailed && job.error_message && (
              <Card className="border-red-200 bg-red-50">
                <CardHeader>
                  <CardTitle className="text-red-700">Error Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-red-600 font-mono text-sm">{job.error_message}</p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <ConfigRow label="Model Type" value={job.config.model_type} />
                <ConfigRow label="Learning Rate" value={job.config.learning_rate.toString()} />
                <ConfigRow label="Batch Size" value={job.config.batch_size.toString()} />
                <ConfigRow label="Epochs" value={job.config.epochs.toString()} />
                <ConfigRow
                  label="Augmentation"
                  value={job.config.augmentation ? 'Enabled' : 'Disabled'}
                />
                <ConfigRow
                  label="Early Stopping"
                  value={job.config.early_stopping ? 'Enabled' : 'Disabled'}
                />
              </CardContent>
            </Card>

            {/* Timeline */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Timeline
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <TimelineItem
                  label="Created"
                  time={job.created_at}
                  completed={true}
                />
                <TimelineItem
                  label="Started"
                  time={job.started_at}
                  completed={!!job.started_at}
                />
                <TimelineItem
                  label="Completed"
                  time={job.completed_at}
                  completed={!!job.completed_at}
                  isLast={true}
                />
              </CardContent>
            </Card>

            {/* Datasets */}
            <Card>
              <CardHeader>
                <CardTitle>Datasets Used</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {job.config.dataset_ids.length > 0 ? (
                    job.config.dataset_ids.map((datasetId) => (
                      <div
                        key={datasetId}
                        className="p-2 bg-muted rounded text-sm font-mono truncate"
                      >
                        {datasetId}
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No datasets specified</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="p-4 bg-muted rounded-lg">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`h-4 w-4 ${color}`} />
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <p className="text-xl font-bold font-mono">{value}</p>
    </div>
  );
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium font-mono">{value}</span>
    </div>
  );
}

function TimelineItem({
  label,
  time,
  completed,
  isLast,
}: {
  label: string;
  time?: string;
  completed: boolean;
  isLast?: boolean;
}) {
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <div
          className={`w-3 h-3 rounded-full ${
            completed ? 'bg-primary' : 'bg-muted border-2 border-border'
          }`}
        />
        {!isLast && (
          <div className={`w-0.5 flex-1 ${completed ? 'bg-primary' : 'bg-border'}`} />
        )}
      </div>
      <div className="pb-4">
        <p className="font-medium">{label}</p>
        <p className="text-sm text-muted-foreground">
          {time ? formatTime(time) : 'Pending'}
        </p>
      </div>
    </div>
  );
}
