'use client';

import useSWR from 'swr';
import { robotsApi } from '@/lib/api';
import type { Robot, RobotCommand } from '@/types';

// Fetcher function for SWR
const fetcher = () => robotsApi.list();
const robotFetcher = (id: string) => robotsApi.get(id);

export function useRobots() {
  const { data, error, isLoading, mutate } = useSWR<Robot[]>(
    '/api/v1/robots',
    fetcher,
    {
      refreshInterval: 10000, // Refresh every 10 seconds
      revalidateOnFocus: true,
    }
  );

  const onlineRobots = data?.filter((r) => r.status === 'online') || [];
  const offlineRobots = data?.filter((r) => r.status === 'offline') || [];
  const trainingRobots = data?.filter((r) => r.status === 'training') || [];

  return {
    robots: data || [],
    onlineRobots,
    offlineRobots,
    trainingRobots,
    isLoading,
    isError: !!error,
    error,
    mutate,
  };
}

export function useRobot(id: string) {
  const { data, error, isLoading, mutate } = useSWR<Robot>(
    id ? `/api/v1/robots/${id}` : null,
    () => robotFetcher(id),
    {
      refreshInterval: 5000, // Refresh every 5 seconds for individual robot
      revalidateOnFocus: true,
    }
  );

  const sendCommand = async (command: RobotCommand) => {
    await robotsApi.sendCommand(id, command);
    mutate();
  };

  const deployModel = async (modelId: string, version: string) => {
    await robotsApi.deployModel(id, modelId, version);
    mutate();
  };

  const updateRobot = async (updates: Partial<Robot>) => {
    const updated = await robotsApi.update(id, updates);
    mutate(updated);
    return updated;
  };

  return {
    robot: data,
    isLoading,
    isError: !!error,
    error,
    mutate,
    sendCommand,
    deployModel,
    updateRobot,
  };
}

export function useCreateRobot() {
  const { mutate } = useSWR<Robot[]>('/api/v1/robots');

  const createRobot = async (data: Partial<Robot>) => {
    const robot = await robotsApi.create(data);
    mutate();
    return robot;
  };

  return { createRobot };
}

export function useDeleteRobot() {
  const { mutate } = useSWR<Robot[]>('/api/v1/robots');

  const deleteRobot = async (id: string) => {
    await robotsApi.delete(id);
    mutate();
  };

  return { deleteRobot };
}
