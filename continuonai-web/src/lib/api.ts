import type {
  Robot,
  RobotCommand,
  Model,
  TrainingJob,
  TrainingConfig,
  Dataset,
  ApiResponse,
  PaginatedResponse,
} from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.continuonai.com';

// Auth token management
let authToken: string | null = null;

export function setAuthToken(token: string | null): void {
  authToken = token;
}

export function getAuthToken(): string | null {
  return authToken;
}

// Base fetch function with authentication
async function fetchWithAuth<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (authToken) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${authToken}`;
  }

  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(error.error || `HTTP error ${response.status}`);
  }

  return response.json();
}

// Robot API
export const robotsApi = {
  list: async (): Promise<Robot[]> => {
    const response = await fetchWithAuth<ApiResponse<Robot[]>>('/api/v1/robots');
    return response.data;
  },

  get: async (id: string): Promise<Robot> => {
    const response = await fetchWithAuth<ApiResponse<Robot>>(`/api/v1/robots/${id}`);
    return response.data;
  },

  create: async (data: Partial<Robot>): Promise<Robot> => {
    const response = await fetchWithAuth<ApiResponse<Robot>>('/api/v1/robots', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    return response.data;
  },

  update: async (id: string, data: Partial<Robot>): Promise<Robot> => {
    const response = await fetchWithAuth<ApiResponse<Robot>>(`/api/v1/robots/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
    return response.data;
  },

  delete: async (id: string): Promise<void> => {
    await fetchWithAuth(`/api/v1/robots/${id}`, { method: 'DELETE' });
  },

  sendCommand: async (id: string, command: RobotCommand): Promise<void> => {
    await fetchWithAuth(`/api/v1/robots/${id}/command`, {
      method: 'POST',
      body: JSON.stringify(command),
    });
  },

  deployModel: async (robotId: string, modelId: string, version: string): Promise<void> => {
    await fetchWithAuth(`/api/v1/robots/${robotId}/deploy`, {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, version }),
    });
  },
};

// Models API
export const modelsApi = {
  list: async (): Promise<Model[]> => {
    const response = await fetchWithAuth<ApiResponse<Model[]>>('/api/v1/models');
    return response.data;
  },

  get: async (id: string): Promise<Model> => {
    const response = await fetchWithAuth<ApiResponse<Model>>(`/api/v1/models/${id}`);
    return response.data;
  },

  getVersions: async (id: string): Promise<string[]> => {
    const response = await fetchWithAuth<ApiResponse<string[]>>(`/api/v1/models/${id}/versions`);
    return response.data;
  },

  getDownloadUrl: async (id: string, version: string): Promise<string> => {
    const response = await fetchWithAuth<ApiResponse<{ url: string }>>(
      `/api/v1/models/${id}/${version}/download`
    );
    return response.data.url;
  },

  delete: async (id: string): Promise<void> => {
    await fetchWithAuth(`/api/v1/models/${id}`, { method: 'DELETE' });
  },
};

// Training API
export const trainingApi = {
  list: async (): Promise<TrainingJob[]> => {
    const response = await fetchWithAuth<ApiResponse<TrainingJob[]>>('/api/v1/training/jobs');
    return response.data;
  },

  get: async (id: string): Promise<TrainingJob> => {
    const response = await fetchWithAuth<ApiResponse<TrainingJob>>(`/api/v1/training/jobs/${id}`);
    return response.data;
  },

  create: async (config: TrainingConfig): Promise<TrainingJob> => {
    const response = await fetchWithAuth<ApiResponse<TrainingJob>>('/api/v1/training/jobs', {
      method: 'POST',
      body: JSON.stringify(config),
    });
    return response.data;
  },

  cancel: async (id: string): Promise<void> => {
    await fetchWithAuth(`/api/v1/training/jobs/${id}/cancel`, { method: 'POST' });
  },

  getLogs: async (id: string): Promise<string[]> => {
    const response = await fetchWithAuth<ApiResponse<string[]>>(`/api/v1/training/jobs/${id}/logs`);
    return response.data;
  },
};

// Datasets API
export const datasetsApi = {
  list: async (): Promise<Dataset[]> => {
    const response = await fetchWithAuth<ApiResponse<Dataset[]>>('/api/v1/datasets');
    return response.data;
  },

  get: async (id: string): Promise<Dataset> => {
    const response = await fetchWithAuth<ApiResponse<Dataset>>(`/api/v1/datasets/${id}`);
    return response.data;
  },

  upload: async (file: File, metadata: Partial<Dataset>): Promise<Dataset> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));

    const headers: HeadersInit = {};
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }

    const response = await fetch(`${API_BASE}/api/v1/datasets/upload`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    const data: ApiResponse<Dataset> = await response.json();
    return data.data;
  },

  delete: async (id: string): Promise<void> => {
    await fetchWithAuth(`/api/v1/datasets/${id}`, { method: 'DELETE' });
  },
};

// Combined API object for convenience
export const api = {
  robots: robotsApi,
  models: modelsApi,
  training: trainingApi,
  datasets: datasetsApi,
  setAuthToken,
  getAuthToken,
};

export default api;
