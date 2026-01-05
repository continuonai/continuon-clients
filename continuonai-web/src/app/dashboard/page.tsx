'use client';

import { useRobots } from '@/hooks/useRobots';
import { useTraining } from '@/hooks/useTraining';
import { useModels } from '@/hooks/useModels';
import { RobotCard } from '@/components/RobotCard';
import { TrainingProgress } from '@/components/TrainingProgress';
import { Navbar } from '@/components/Navbar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';
import {
  Bot,
  Upload,
  Play,
  Box,
  TrendingUp,
  Activity,
  AlertCircle,
  Clock,
} from 'lucide-react';

export default function Dashboard() {
  const { robots, onlineRobots, isLoading: robotsLoading } = useRobots();
  const { activeJobs, completedJobs, isLoading: trainingLoading } = useTraining();
  const { models, readyModels, isLoading: modelsLoading } = useModels();

  const isLoading = robotsLoading || trainingLoading || modelsLoading;

  // Calculate total episodes across all robots
  const totalEpisodes = robots.reduce((sum, r) => sum + r.episode_count, 0);

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Dashboard</h1>
            <p className="text-muted-foreground">
              Welcome back! Here is an overview of your robot fleet.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">
              Last updated: {new Date().toLocaleTimeString()}
            </span>
            <Activity className={`h-4 w-4 ${isLoading ? 'animate-pulse text-blue-500' : 'text-green-500'}`} />
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Total Robots"
            value={robots.length}
            subtext={`${onlineRobots.length} online`}
            icon={Bot}
            trend={onlineRobots.length > 0 ? 'up' : 'neutral'}
          />
          <StatCard
            title="Active Training"
            value={activeJobs.length}
            subtext={`${completedJobs.length} completed`}
            icon={Play}
            trend={activeJobs.length > 0 ? 'up' : 'neutral'}
          />
          <StatCard
            title="Models"
            value={models.length}
            subtext={`${readyModels.length} ready`}
            icon={Box}
            trend={readyModels.length > 0 ? 'up' : 'neutral'}
          />
          <StatCard
            title="Total Episodes"
            value={totalEpisodes.toLocaleString()}
            subtext="Training data"
            icon={TrendingUp}
            trend="up"
          />
        </div>

        {/* Fleet Overview */}
        <section className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Your Robots</h2>
            <Link href="/robots">
              <Button variant="outline" size="sm">
                View All
              </Button>
            </Link>
          </div>

          {robotsLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <Card key={i} className="animate-pulse">
                  <CardContent className="p-6">
                    <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                    <div className="h-3 bg-muted rounded w-1/2 mb-4" />
                    <div className="grid grid-cols-2 gap-2">
                      <div className="h-3 bg-muted rounded" />
                      <div className="h-3 bg-muted rounded" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : robots.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {robots.slice(0, 6).map((robot) => (
                <RobotCard key={robot.id} robot={robot} />
              ))}
            </div>
          ) : (
            <Card className="border-dashed">
              <CardContent className="p-8 text-center">
                <Bot className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="font-semibold mb-2">No robots connected</h3>
                <p className="text-muted-foreground mb-4">
                  Connect your first robot to get started with ContinuonAI.
                </p>
                <Button>Add Robot</Button>
              </CardContent>
            </Card>
          )}
        </section>

        {/* Active Training */}
        <section className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Active Training</h2>
            <Link href="/training">
              <Button variant="outline" size="sm">
                View All
              </Button>
            </Link>
          </div>

          {trainingLoading ? (
            <Card className="animate-pulse">
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
          ) : activeJobs.length > 0 ? (
            <div className="space-y-4">
              {activeJobs.map((job) => (
                <TrainingProgress key={job.id} job={job} />
              ))}
            </div>
          ) : (
            <Card className="border-dashed">
              <CardContent className="p-8 text-center">
                <Play className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="font-semibold mb-2">No active training</h3>
                <p className="text-muted-foreground mb-4">
                  Start a new training job to train your robot models.
                </p>
                <Link href="/training">
                  <Button>Start Training</Button>
                </Link>
              </CardContent>
            </Card>
          )}
        </section>

        {/* Quick Actions */}
        <section>
          <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <ActionCard
              title="Upload Episodes"
              description="Upload new training data from your robots"
              icon={Upload}
              href="/training"
            />
            <ActionCard
              title="Start Training"
              description="Begin training a new model"
              icon={Play}
              href="/training"
            />
            <ActionCard
              title="Browse Models"
              description="View and download trained models"
              icon={Box}
              href="/models"
            />
          </div>
        </section>

        {/* Recent Activity (Optional) */}
        <section className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
          <Card>
            <CardContent className="p-4">
              <div className="space-y-4">
                <ActivityItem
                  icon={Play}
                  title="Training started"
                  description="Model diffusion-policy-v3 training started"
                  time="2 hours ago"
                />
                <ActivityItem
                  icon={Bot}
                  title="Robot connected"
                  description="Robot-01 came online"
                  time="3 hours ago"
                />
                <ActivityItem
                  icon={Box}
                  title="Model deployed"
                  description="Model act-v2.1 deployed to Robot-01"
                  time="5 hours ago"
                />
              </div>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}

function StatCard({
  title,
  value,
  subtext,
  icon: Icon,
  trend,
}: {
  title: string;
  value: number | string;
  subtext: string;
  icon: React.ElementType;
  trend: 'up' | 'down' | 'neutral';
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground flex items-center gap-1">
          {trend === 'up' && <TrendingUp className="h-3 w-3 text-green-500" />}
          {subtext}
        </p>
      </CardContent>
    </Card>
  );
}

function ActionCard({
  title,
  description,
  icon: Icon,
  href,
}: {
  title: string;
  description: string;
  icon: React.ElementType;
  href: string;
}) {
  return (
    <Link href={href}>
      <Card className="hover:shadow-md transition-shadow cursor-pointer h-full">
        <CardContent className="p-6 flex items-start gap-4">
          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center shrink-0">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold">{title}</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function ActivityItem({
  icon: Icon,
  title,
  description,
  time,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  time: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center shrink-0">
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium text-sm">{title}</p>
        <p className="text-sm text-muted-foreground truncate">{description}</p>
      </div>
      <span className="text-xs text-muted-foreground whitespace-nowrap">
        {time}
      </span>
    </div>
  );
}
