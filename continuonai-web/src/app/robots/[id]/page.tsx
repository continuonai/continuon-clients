'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { useRobot } from '@/hooks/useRobots';
import { Navbar } from '@/components/Navbar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { getStatusColor, formatTime } from '@/lib/utils';
import {
  ArrowLeft,
  Bot,
  Battery,
  Cpu,
  HardDrive,
  Thermometer,
  Wifi,
  WifiOff,
  Camera,
  Play,
  Square,
  RotateCcw,
  Download,
  Settings,
  Activity,
  Database,
  Clock,
  RefreshCw,
} from 'lucide-react';

export default function RobotDetailPage() {
  const params = useParams();
  const robotId = params.id as string;
  const { robot, isLoading, sendCommand, deployModel, mutate } = useRobot(robotId);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-muted rounded w-1/4" />
            <div className="h-4 bg-muted rounded w-1/2" />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
              {[1, 2, 3].map((i) => (
                <Card key={i}>
                  <CardContent className="p-6">
                    <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                    <div className="h-8 bg-muted rounded w-1/2" />
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (!robot) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Card className="max-w-md mx-auto">
            <CardContent className="p-8 text-center">
              <Bot className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h2 className="text-xl font-semibold mb-2">Robot Not Found</h2>
              <p className="text-muted-foreground mb-4">
                The robot you are looking for does not exist or you do not have permission to view it.
              </p>
              <Link href="/robots">
                <Button>
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Robots
                </Button>
              </Link>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  const statusColor = getStatusColor(robot.status);
  const isOnline = robot.status === 'online' || robot.status === 'training';
  const telemetry = robot.telemetry;

  const handleCommand = async (type: string) => {
    await sendCommand({ type: type as 'move' | 'rotate' | 'stop' | 'deploy_model' | 'update_config' });
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Link href="/robots">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div className="flex-1">
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold">{robot.name}</h1>
              <Badge className={statusColor}>
                {robot.status.charAt(0).toUpperCase() + robot.status.slice(1)}
              </Badge>
            </div>
            <p className="text-muted-foreground font-mono">{robot.device_id}</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={() => mutate()}>
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button variant="outline">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>

        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatusCard
            title="Connection"
            value={isOnline ? 'Connected' : 'Disconnected'}
            icon={isOnline ? Wifi : WifiOff}
            iconColor={isOnline ? 'text-green-500' : 'text-gray-400'}
          />
          <StatusCard
            title="Battery"
            value={`${robot.battery}%`}
            icon={Battery}
            iconColor={robot.battery < 20 ? 'text-red-500' : 'text-green-500'}
            progress={robot.battery}
          />
          <StatusCard
            title="Model"
            value={robot.model_version || 'None'}
            icon={Cpu}
            iconColor="text-blue-500"
          />
          <StatusCard
            title="Last Seen"
            value={formatTime(robot.last_seen)}
            icon={Clock}
            iconColor="text-purple-500"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Telemetry */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  System Telemetry
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <TelemetryItem
                    label="CPU Usage"
                    value={telemetry?.cpu_usage ?? 0}
                    unit="%"
                    icon={Cpu}
                    max={100}
                  />
                  <TelemetryItem
                    label="Memory"
                    value={telemetry?.memory_usage ?? 0}
                    unit="%"
                    icon={HardDrive}
                    max={100}
                  />
                  <TelemetryItem
                    label="Temperature"
                    value={telemetry?.temperature ?? 0}
                    unit="C"
                    icon={Thermometer}
                    max={100}
                    warning={70}
                  />
                  <TelemetryItem
                    label="Disk"
                    value={telemetry?.disk_usage ?? 0}
                    unit="%"
                    icon={Database}
                    max={100}
                    warning={80}
                  />
                </div>

                <div className="mt-4 pt-4 border-t flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    {telemetry?.network_status === 'connected' ? (
                      <>
                        <Wifi className="h-4 w-4 text-green-500" />
                        <span className="text-sm">Network: Connected</span>
                      </>
                    ) : (
                      <>
                        <WifiOff className="h-4 w-4 text-gray-400" />
                        <span className="text-sm text-muted-foreground">Network: Disconnected</span>
                      </>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Camera className={`h-4 w-4 ${telemetry?.camera_status === 'active' ? 'text-green-500' : 'text-gray-400'}`} />
                    <span className="text-sm">
                      Camera: {telemetry?.camera_status === 'active' ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Statistics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Training Data
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-muted rounded-lg">
                    <p className="text-3xl font-bold text-primary">{robot.episode_count.toLocaleString()}</p>
                    <p className="text-sm text-muted-foreground">Total Episodes</p>
                  </div>
                  <div className="text-center p-4 bg-muted rounded-lg">
                    <p className="text-3xl font-bold text-primary">N/A</p>
                    <p className="text-sm text-muted-foreground">Success Rate</p>
                  </div>
                  <div className="text-center p-4 bg-muted rounded-lg">
                    <p className="text-3xl font-bold text-primary">N/A</p>
                    <p className="text-sm text-muted-foreground">Avg Episode Length</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Live Feed Placeholder */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Live Feed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  {isOnline && telemetry?.camera_status === 'active' ? (
                    <p className="text-muted-foreground">Live video feed would appear here</p>
                  ) : (
                    <div className="text-center">
                      <Camera className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                      <p className="text-muted-foreground">Camera not available</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button
                  className="w-full justify-start"
                  variant="outline"
                  disabled={!isOnline}
                  onClick={() => handleCommand('stop')}
                >
                  <Square className="h-4 w-4 mr-2" />
                  Emergency Stop
                </Button>
                <Button
                  className="w-full justify-start"
                  variant="outline"
                  disabled={!isOnline}
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Recording
                </Button>
                <Button
                  className="w-full justify-start"
                  variant="outline"
                  disabled={!isOnline}
                  onClick={() => handleCommand('update_config')}
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Restart Robot
                </Button>
                <Button
                  className="w-full justify-start"
                  variant="outline"
                  disabled={!isOnline}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download Episodes
                </Button>
              </CardContent>
            </Card>

            {/* Model Deployment */}
            <Card>
              <CardHeader>
                <CardTitle>Model Deployment</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground">Current Model</p>
                    <p className="font-medium">{robot.model_version || 'No model deployed'}</p>
                  </div>
                  <Button className="w-full" disabled={!isOnline}>
                    Deploy New Model
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Robot Info */}
            <Card>
              <CardHeader>
                <CardTitle>Robot Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Device ID</span>
                  <span className="font-mono">{robot.device_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Created</span>
                  <span>{new Date(robot.created_at).toLocaleDateString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Last Updated</span>
                  <span>{formatTime(robot.updated_at)}</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

function StatusCard({
  title,
  value,
  icon: Icon,
  iconColor,
  progress,
}: {
  title: string;
  value: string;
  icon: React.ElementType;
  iconColor: string;
  progress?: number;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-muted-foreground">{title}</span>
          <Icon className={`h-4 w-4 ${iconColor}`} />
        </div>
        <p className="text-xl font-bold">{value}</p>
        {progress !== undefined && (
          <Progress value={progress} className="h-1 mt-2" />
        )}
      </CardContent>
    </Card>
  );
}

function TelemetryItem({
  label,
  value,
  unit,
  icon: Icon,
  max,
  warning,
}: {
  label: string;
  value: number;
  unit: string;
  icon: React.ElementType;
  max: number;
  warning?: number;
}) {
  const percentage = (value / max) * 100;
  const isWarning = warning && value >= warning;

  return (
    <div className="p-3 bg-muted rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted-foreground">{label}</span>
        <Icon className={`h-4 w-4 ${isWarning ? 'text-orange-500' : 'text-muted-foreground'}`} />
      </div>
      <p className={`text-lg font-bold ${isWarning ? 'text-orange-500' : ''}`}>
        {value}{unit}
      </p>
      <Progress
        value={percentage}
        className={`h-1 mt-2 ${isWarning ? '[&>div]:bg-orange-500' : ''}`}
      />
    </div>
  );
}
