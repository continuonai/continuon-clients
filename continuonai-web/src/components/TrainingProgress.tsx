'use client';

import Link from 'next/link';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { LossChart, LossChartMini } from './LossChart';
import { getStatusColor, formatDuration } from '@/lib/utils';
import type { TrainingJob } from '@/types';
import {
  Play,
  Pause,
  Square,
  Clock,
  TrendingDown,
  Award,
  Timer,
  ExternalLink,
} from 'lucide-react';

interface TrainingProgressProps {
  job: TrainingJob;
  showChart?: boolean;
  compact?: boolean;
  onCancel?: (id: string) => void;
}

export function TrainingProgress({
  job,
  showChart = true,
  compact = false,
  onCancel,
}: TrainingProgressProps) {
  const progress = (job.steps_completed / job.total_steps) * 100;
  const statusColor = getStatusColor(job.status);
  const isRunning = job.status === 'running';
  const isQueued = job.status === 'queued';

  if (compact) {
    return (
      <Card className="hover:shadow-md transition-shadow">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${statusColor} ${isRunning ? 'animate-pulse' : ''}`} />
              <span className="font-medium text-sm truncate max-w-[150px]">
                {job.model_id}
              </span>
            </div>
            <Badge variant="outline" className="text-xs">
              {job.status}
            </Badge>
          </div>
          <Progress value={progress} className="h-2 mb-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{job.steps_completed.toLocaleString()} / {job.total_steps.toLocaleString()}</span>
            <span>{progress.toFixed(1)}%</span>
          </div>
          {job.loss_history && <LossChartMini data={job.loss_history} />}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-l-4" style={{ borderLeftColor: isRunning ? 'hsl(var(--primary))' : undefined }}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-lg">{job.model_id}</h3>
              {isRunning && (
                <div className="flex items-center gap-1 text-blue-600">
                  <Play className="h-4 w-4 animate-pulse" />
                </div>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              {job.config.model_type} | Batch: {job.config.batch_size} | LR: {job.config.learning_rate}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={statusColor}>
              {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
            </Badge>
            <Link href={`/training/${job.id}`}>
              <Button variant="ghost" size="icon">
                <ExternalLink className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Progress Bar */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-mono">
                {job.steps_completed.toLocaleString()} / {job.total_steps.toLocaleString()} steps
              </span>
            </div>
            <Progress value={progress} className="h-3" />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>{progress.toFixed(1)}% complete</span>
              {job.eta && <span>ETA: {job.eta}</span>}
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
              <TrendingDown className="h-4 w-4 text-blue-500" />
              <div>
                <p className="text-xs text-muted-foreground">Current Loss</p>
                <p className="font-mono font-medium">
                  {job.current_loss?.toFixed(4) || 'N/A'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
              <Award className="h-4 w-4 text-green-500" />
              <div>
                <p className="text-xs text-muted-foreground">Best Loss</p>
                <p className="font-mono font-medium">
                  {job.best_loss?.toFixed(4) || 'N/A'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
              <Timer className="h-4 w-4 text-orange-500" />
              <div>
                <p className="text-xs text-muted-foreground">ETA</p>
                <p className="font-medium">{job.eta || 'Calculating...'}</p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
              <Clock className="h-4 w-4 text-purple-500" />
              <div>
                <p className="text-xs text-muted-foreground">Elapsed</p>
                <p className="font-medium">
                  {job.started_at
                    ? formatDuration(
                        Math.floor(
                          (Date.now() - new Date(job.started_at).getTime()) / 1000
                        )
                      )
                    : 'Not started'}
                </p>
              </div>
            </div>
          </div>

          {/* Loss Chart */}
          {showChart && job.loss_history && job.loss_history.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium mb-2">Training Progress</h4>
              <LossChart data={job.loss_history} height={200} />
            </div>
          )}

          {/* Action Buttons */}
          {(isRunning || isQueued) && onCancel && (
            <div className="flex justify-end gap-2 mt-4 pt-4 border-t">
              <Button
                variant="destructive"
                size="sm"
                onClick={() => onCancel(job.id)}
              >
                <Square className="h-4 w-4 mr-2" />
                Cancel Training
              </Button>
            </div>
          )}

          {/* Error Message */}
          {job.status === 'failed' && job.error_message && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">
                <strong>Error:</strong> {job.error_message}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default TrainingProgress;
