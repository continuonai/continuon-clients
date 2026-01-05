'use client';

import useSWR from 'swr';
import { modelsApi } from '@/lib/api';
import type { Model, ModelType, ModelStatus } from '@/types';

// Fetcher functions
const fetcher = () => modelsApi.list();
const modelFetcher = (id: string) => modelsApi.get(id);

export function useModels(filters?: {
  type?: ModelType;
  status?: ModelStatus;
  search?: string;
}) {
  const { data, error, isLoading, mutate } = useSWR<Model[]>(
    '/api/v1/models',
    fetcher,
    {
      refreshInterval: 30000, // Refresh every 30 seconds
      revalidateOnFocus: true,
    }
  );

  // Apply client-side filters
  let filteredModels = data || [];

  if (filters?.type) {
    filteredModels = filteredModels.filter((m) => m.type === filters.type);
  }

  if (filters?.status) {
    filteredModels = filteredModels.filter((m) => m.status === filters.status);
  }

  if (filters?.search) {
    const searchLower = filters.search.toLowerCase();
    filteredModels = filteredModels.filter(
      (m) =>
        m.name.toLowerCase().includes(searchLower) ||
        m.description.toLowerCase().includes(searchLower)
    );
  }

  // Group by type
  const modelsByType = filteredModels.reduce(
    (acc, model) => {
      if (!acc[model.type]) {
        acc[model.type] = [];
      }
      acc[model.type].push(model);
      return acc;
    },
    {} as Record<ModelType, Model[]>
  );

  // Get ready models
  const readyModels = filteredModels.filter((m) => m.status === 'ready');

  return {
    models: filteredModels,
    allModels: data || [],
    modelsByType,
    readyModels,
    isLoading,
    isError: !!error,
    error,
    mutate,
  };
}

export function useModel(id: string) {
  const { data, error, isLoading, mutate } = useSWR<Model>(
    id ? `/api/v1/models/${id}` : null,
    () => modelFetcher(id),
    {
      refreshInterval: 10000,
      revalidateOnFocus: true,
    }
  );

  const getDownloadUrl = async (version?: string) => {
    const v = version || data?.version || 'latest';
    return modelsApi.getDownloadUrl(id, v);
  };

  const getVersions = async () => {
    return modelsApi.getVersions(id);
  };

  return {
    model: data,
    isLoading,
    isError: !!error,
    error,
    mutate,
    getDownloadUrl,
    getVersions,
  };
}

export function useDeleteModel() {
  const { mutate } = useSWR<Model[]>('/api/v1/models');

  const deleteModel = async (id: string) => {
    await modelsApi.delete(id);
    mutate();
  };

  return { deleteModel };
}

// Hook for model type options
export function useModelTypes(): { value: ModelType; label: string }[] {
  return [
    { value: 'diffusion_policy', label: 'Diffusion Policy' },
    { value: 'act', label: 'ACT' },
    { value: 'pi0', label: 'Pi0' },
    { value: 'custom', label: 'Custom' },
  ];
}
