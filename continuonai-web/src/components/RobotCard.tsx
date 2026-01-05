'use client';

import Link from 'next/link';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { formatTime, getStatusColor } from '@/lib/utils';
import type { Robot } from '@/types';
import {
  Cpu,
  Battery,
  Clock,
  Database,
  Wifi,
  WifiOff,
  Activity
} from 'lucide-react';

interface RobotCardProps {
  robot: Robot;
}

export function RobotCard({ robot }: RobotCardProps) {
  const statusColor = getStatusColor(robot.status);
  const isOnline = robot.status === 'online' || robot.status === 'training';

  return (
    <Link href={`/robots/${robot.id}`}>
      <Card className="hover:shadow-lg transition-shadow cursor-pointer border-l-4 hover:border-l-primary">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${statusColor} animate-pulse`} />
            <div>
              <h3 className="font-semibold text-lg">{robot.name}</h3>
              <p className="text-sm text-muted-foreground font-mono">
                {robot.device_id}
              </p>
            </div>
          </div>
          <Badge className={statusColor}>
            {robot.status.charAt(0).toUpperCase() + robot.status.slice(1)}
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Model:</span>
              <span className="font-medium truncate">{robot.model_version || 'None'}</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Last seen:</span>
              <span className="font-medium">{formatTime(robot.last_seen)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Battery className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Battery:</span>
              <span className={`font-medium ${robot.battery < 20 ? 'text-red-500' : ''}`}>
                {robot.battery}%
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Episodes:</span>
              <span className="font-medium">{robot.episode_count.toLocaleString()}</span>
            </div>
          </div>

          {/* Connection status indicator */}
          <div className="mt-4 pt-3 border-t flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm">
              {isOnline ? (
                <>
                  <Wifi className="h-4 w-4 text-green-500" />
                  <span className="text-green-600">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-4 w-4 text-gray-400" />
                  <span className="text-gray-500">Disconnected</span>
                </>
              )}
            </div>
            {robot.status === 'training' && (
              <div className="flex items-center gap-2 text-sm text-blue-600">
                <Activity className="h-4 w-4 animate-pulse" />
                <span>Training in progress</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

export default RobotCard;
