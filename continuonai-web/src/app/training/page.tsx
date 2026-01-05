'use client';

import { useState } from 'react';
import { useTraining, useTrainingStatuses, useCreateTrainingJob, useDefaultTrainingConfig } from '@/hooks/useTraining';
import { TrainingProgress } from '@/components/TrainingProgress';
import { Navbar } from '@/components/Navbar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import {
  GraduationCap,
  Search,
  Filter,
  Plus,
  RefreshCw,
  Play,
  Clock,
  CheckCircle,
  XCircle,
  TrendingDown,
} from 'lucide-react';
import type { TrainingStatus } from '@/types';

export default function TrainingPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<TrainingStatus | 'all'>('all');
  const { jobs, activeJobs, completedJobs, failedJobs, overallProgress, isLoading, mutate } = useTraining();
  const statusOptions = useTrainingStatuses();

  // Filter jobs
  const filteredJobs = jobs.filter((job) => {
    const matchesSearch = job.model_id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || job.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Status counts
  const statusCounts: Record<string, number> = {
    all: jobs.length,
    queued: jobs.filter((j) => j.status === 'queued').length,
    running: jobs.filter((j) => j.status === 'running').length,
    completed: jobs.filter((j) => j.status === 'completed').length,
    failed: jobs.filter((j) => j.status === 'failed').length,
    cancelled: jobs.filter((j) => j.status === 'cancelled').length,
  };

  const handleCancelJob = async (jobId: string) => {
    // In a real app, this would call the API
    console.log('Cancelling job:', jobId);
    mutate();
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold">Training Jobs</h1>
            <p className="text-muted-foreground">
              Manage and monitor your model training jobs
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={() => mutate()}>
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              New Training Job
            </Button>
          </div>
        </div>

        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Active Jobs</span>
                <Play className="h-4 w-4 text-blue-500" />
              </div>
              <p className="text-2xl font-bold">{activeJobs.length}</p>
              {activeJobs.length > 0 && (
                <div className="mt-2">
                  <div className="flex justify-between text-xs mb-1">
                    <span>Overall Progress</span>
                    <span>{overallProgress.toFixed(1)}%</span>
                  </div>
                  <Progress value={overallProgress} className="h-1" />
                </div>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Queued</span>
                <Clock className="h-4 w-4 text-yellow-500" />
              </div>
              <p className="text-2xl font-bold">{statusCounts.queued}</p>
              <p className="text-xs text-muted-foreground mt-2">Waiting to start</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Completed</span>
                <CheckCircle className="h-4 w-4 text-green-500" />
              </div>
              <p className="text-2xl font-bold">{completedJobs.length}</p>
              <p className="text-xs text-muted-foreground mt-2">Successfully trained</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Failed</span>
                <XCircle className="h-4 w-4 text-red-500" />
              </div>
              <p className="text-2xl font-bold">{failedJobs.length}</p>
              <p className="text-xs text-muted-foreground mt-2">Need attention</p>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search training jobs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Status Filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <Button
              variant={statusFilter === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setStatusFilter('all')}
            >
              All
              <Badge variant="secondary" className="ml-2">
                {statusCounts.all}
              </Badge>
            </Button>
            {statusOptions.map((status) => (
              <Button
                key={status.value}
                variant={statusFilter === status.value ? 'default' : 'outline'}
                size="sm"
                onClick={() => setStatusFilter(status.value)}
                className="gap-2"
              >
                <span className={`w-2 h-2 rounded-full ${status.color}`} />
                {status.label}
                <Badge variant="secondary" className="ml-1">
                  {statusCounts[status.value] || 0}
                </Badge>
              </Button>
            ))}
          </div>
        </div>

        {/* Active Training Jobs */}
        {activeJobs.length > 0 && (
          <section className="mb-8">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Play className="h-5 w-5 text-blue-500 animate-pulse" />
              Active Training
            </h2>
            <div className="space-y-4">
              {activeJobs.map((job) => (
                <TrainingProgress
                  key={job.id}
                  job={job}
                  showChart={true}
                  onCancel={handleCancelJob}
                />
              ))}
            </div>
          </section>
        )}

        {/* All Jobs */}
        <section>
          <h2 className="text-xl font-semibold mb-4">
            {statusFilter === 'all' ? 'All Jobs' : `${statusFilter.charAt(0).toUpperCase() + statusFilter.slice(1)} Jobs`}
          </h2>

          {isLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <Card key={i} className="animate-pulse">
                  <CardContent className="p-6">
                    <div className="h-4 bg-muted rounded w-1/4 mb-4" />
                    <div className="h-2 bg-muted rounded w-full mb-4" />
                    <div className="grid grid-cols-3 gap-4">
                      <div className="h-3 bg-muted rounded" />
                      <div className="h-3 bg-muted rounded" />
                      <div className="h-3 bg-muted rounded" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : filteredJobs.length > 0 ? (
            <div className="space-y-4">
              {filteredJobs
                .filter((job) => job.status !== 'running' && job.status !== 'queued')
                .map((job) => (
                  <TrainingProgress
                    key={job.id}
                    job={job}
                    showChart={false}
                    compact={false}
                  />
                ))}
            </div>
          ) : (
            <Card className="border-dashed">
              <CardContent className="p-12 text-center">
                <GraduationCap className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
                {searchQuery || statusFilter !== 'all' ? (
                  <>
                    <h3 className="text-lg font-semibold mb-2">No jobs found</h3>
                    <p className="text-muted-foreground mb-4">
                      Try adjusting your search or filter criteria.
                    </p>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setSearchQuery('');
                        setStatusFilter('all');
                      }}
                    >
                      Clear Filters
                    </Button>
                  </>
                ) : (
                  <>
                    <h3 className="text-lg font-semibold mb-2">No training jobs</h3>
                    <p className="text-muted-foreground mb-4">
                      Start your first training job to train a model.
                    </p>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Training Job
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </section>

        {/* Summary Footer */}
        {!isLoading && filteredJobs.length > 0 && (
          <div className="mt-6 pt-6 border-t flex items-center justify-between text-sm text-muted-foreground">
            <span>
              Showing {filteredJobs.length} of {jobs.length} jobs
            </span>
            <span>
              {statusCounts.running} running | {statusCounts.completed} completed | {statusCounts.failed} failed
            </span>
          </div>
        )}
      </main>
    </div>
  );
}
