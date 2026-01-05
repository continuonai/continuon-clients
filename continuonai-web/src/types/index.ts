// Robot types
export type RobotStatus = 'online' | 'offline' | 'training' | 'error';

export interface Robot {
  id: string;
  name: string;
  device_id: string;
  status: RobotStatus;
  model_version: string;
  last_seen: string;
  battery: number;
  episode_count: number;
  owner_id: string;
  created_at: string;
  updated_at: string;
  telemetry?: RobotTelemetry;
}

export interface RobotTelemetry {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  temperature: number;
  network_status: 'connected' | 'disconnected';
  camera_status: 'active' | 'inactive';
}

export interface RobotCommand {
  type: 'move' | 'rotate' | 'stop' | 'deploy_model' | 'update_config';
  payload?: Record<string, unknown>;
}

// Model types
export type ModelType = 'diffusion_policy' | 'act' | 'pi0' | 'custom';
export type ModelStatus = 'training' | 'ready' | 'deprecated' | 'failed';

export interface Model {
  id: string;
  name: string;
  type: ModelType;
  version: string;
  status: ModelStatus;
  description: string;
  metrics: ModelMetrics;
  created_at: string;
  updated_at: string;
  download_url?: string;
  size_mb: number;
  compatible_robots: string[];
}

export interface ModelMetrics {
  loss: number;
  accuracy?: number;
  inference_time_ms: number;
  episode_count: number;
}

// Training types
export type TrainingStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TrainingJob {
  id: string;
  model_id: string;
  status: TrainingStatus;
  steps_completed: number;
  total_steps: number;
  current_loss?: number;
  best_loss?: number;
  eta?: string;
  loss_history?: LossDataPoint[];
  config: TrainingConfig;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface TrainingConfig {
  model_type: ModelType;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  dataset_ids: string[];
  augmentation: boolean;
  early_stopping: boolean;
}

export interface LossDataPoint {
  step: number;
  loss: number;
  val_loss?: number;
}

// Dataset types
export interface Dataset {
  id: string;
  name: string;
  episode_count: number;
  total_frames: number;
  size_mb: number;
  robot_id: string;
  created_at: string;
  tags: string[];
}

// User types
export interface User {
  id: string;
  email: string;
  display_name: string;
  avatar_url?: string;
  created_at: string;
  subscription_tier: 'free' | 'pro' | 'enterprise';
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  error?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}
