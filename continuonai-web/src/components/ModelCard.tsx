'use client';

import Link from 'next/link';
import { Card, CardHeader, CardContent, CardFooter } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getStatusColor, formatTime, formatBytes } from '@/lib/utils';
import type { Model } from '@/types';
import {
  Download,
  Clock,
  Zap,
  Database,
  TrendingDown,
  ExternalLink,
} from 'lucide-react';

interface ModelCardProps {
  model: Model;
  onDownload?: (id: string) => void;
}

export function ModelCard({ model, onDownload }: ModelCardProps) {
  const statusColor = getStatusColor(model.status);
  const isReady = model.status === 'ready';

  const modelTypeLabels: Record<string, string> = {
    diffusion_policy: 'Diffusion Policy',
    act: 'ACT',
    pi0: 'Pi0',
    custom: 'Custom',
  };

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="font-semibold text-lg">{model.name}</h3>
            <p className="text-sm text-muted-foreground">
              v{model.version} | {modelTypeLabels[model.type] || model.type}
            </p>
          </div>
          <Badge className={statusColor}>
            {model.status.charAt(0).toUpperCase() + model.status.slice(1)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
          {model.description || 'No description available'}
        </p>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex items-center gap-2">
            <TrendingDown className="h-4 w-4 text-blue-500" />
            <div>
              <p className="text-xs text-muted-foreground">Loss</p>
              <p className="font-mono font-medium">{model.metrics.loss.toFixed(4)}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-yellow-500" />
            <div>
              <p className="text-xs text-muted-foreground">Inference</p>
              <p className="font-mono font-medium">{model.metrics.inference_time_ms}ms</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-green-500" />
            <div>
              <p className="text-xs text-muted-foreground">Episodes</p>
              <p className="font-medium">{model.metrics.episode_count.toLocaleString()}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-purple-500" />
            <div>
              <p className="text-xs text-muted-foreground">Updated</p>
              <p className="font-medium">{formatTime(model.updated_at)}</p>
            </div>
          </div>
        </div>

        {/* Size indicator */}
        <div className="mt-4 pt-3 border-t">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Model Size</span>
            <span className="font-medium">{formatBytes(model.size_mb * 1024 * 1024)}</span>
          </div>
          {model.compatible_robots && model.compatible_robots.length > 0 && (
            <div className="mt-2">
              <span className="text-xs text-muted-foreground">Compatible with:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {model.compatible_robots.slice(0, 3).map((robotId) => (
                  <Badge key={robotId} variant="outline" className="text-xs">
                    {robotId}
                  </Badge>
                ))}
                {model.compatible_robots.length > 3 && (
                  <Badge variant="outline" className="text-xs">
                    +{model.compatible_robots.length - 3} more
                  </Badge>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="gap-2">
        {isReady && onDownload && (
          <Button
            variant="default"
            size="sm"
            className="flex-1"
            onClick={() => onDownload(model.id)}
          >
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        )}
        <Link href={`/models/${model.id}`} className="flex-1">
          <Button variant="outline" size="sm" className="w-full">
            <ExternalLink className="h-4 w-4 mr-2" />
            Details
          </Button>
        </Link>
      </CardFooter>
    </Card>
  );
}

export default ModelCard;
