'use client';

import useSWR from 'swr';
import { trainingApi } from '@/lib/api';
import type { TrainingJob, TrainingConfig, TrainingStatus } from '@/types';

// Fetcher functions
const fetcher = () => trainingApi.list();
const jobFetcher = (id: string) => trainingApi.get(id);

export function useTraining(filters?: {
  status?: TrainingStatus;
}) {
  const { data, error, isLoading, mutate } = useSWR<TrainingJob[]>(
    '/api/v1/training/jobs',
    fetcher,
    {
      refreshInterval: 5000, // Refresh every 5 seconds for training updates
      revalidateOnFocus: true,
    }
  );

  // Apply client-side filters
  let filteredJobs = data || [];

  if (filters?.status) {
    filteredJobs = filteredJobs.filter((j) => j.status === filters.status);
  }

  // Get active jobs (queued or running)
  const activeJobs = (data || []).filter(
    (j) => j.status === 'queued' || j.status === 'running'
  );

  // Get completed jobs
  const completedJobs = (data || []).filter(
    (j) => j.status === 'completed'
  );

  // Get failed jobs
  const failedJobs = (data || []).filter(
    (j) => j.status === 'failed'
  );

  // Calculate overall progress for running jobs
  const overallProgress =
    activeJobs.length > 0
      ? activeJobs.reduce((acc, job) => {
          return acc + (job.steps_completed / job.total_steps) * 100;
        }, 0) / activeJobs.length
      : 0;

  return {
    jobs: filteredJobs,
    allJobs: data || [],
    activeJobs,
    completedJobs,
    failedJobs,
    overallProgress,
    isLoading,
    isError: !!error,
    error,
    mutate,
  };
}

export function useTrainingJob(id: string) {
  const { data, error, isLoading, mutate } = useSWR<TrainingJob>(
    id ? `/api/v1/training/jobs/${id}` : null,
    () => jobFetcher(id),
    {
      refreshInterval: 2000, // Refresh every 2 seconds for active job detail
      revalidateOnFocus: true,
    }
  );

  const cancel = async () => {
    await trainingApi.cancel(id);
    mutate();
  };

  const getLogs = async () => {
    return trainingApi.getLogs(id);
  };

  // Calculate progress percentage
  const progress = data
    ? (data.steps_completed / data.total_steps) * 100
    : 0;

  // Calculate ETA
  const eta = data?.eta || 'Calculating...';

  return {
    job: data,
    progress,
    eta,
    isLoading,
    isError: !!error,
    error,
    mutate,
    cancel,
    getLogs,
  };
}

export function useCreateTrainingJob() {
  const { mutate } = useSWR<TrainingJob[]>('/api/v1/training/jobs');

  const createJob = async (config: TrainingConfig) => {
    const job = await trainingApi.create(config);
    mutate();
    return job;
  };

  return { createJob };
}

// Hook for training status options
export function useTrainingStatuses(): { value: TrainingStatus; label: string; color: string }[] {
  return [
    { value: 'queued', label: 'Queued', color: 'bg-yellow-500' },
    { value: 'running', label: 'Running', color: 'bg-blue-500' },
    { value: 'completed', label: 'Completed', color: 'bg-green-500' },
    { value: 'failed', label: 'Failed', color: 'bg-red-500' },
    { value: 'cancelled', label: 'Cancelled', color: 'bg-gray-500' },
  ];
}

// Hook to get default training config
export function useDefaultTrainingConfig(): Partial<TrainingConfig> {
  return {
    model_type: 'diffusion_policy',
    learning_rate: 0.0001,
    batch_size: 32,
    epochs: 100,
    augmentation: true,
    early_stopping: true,
  };
}
