'use client';

import { useState } from 'react';
import { useRobots } from '@/hooks/useRobots';
import { RobotCard } from '@/components/RobotCard';
import { Navbar } from '@/components/Navbar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import {
  Bot,
  Search,
  Plus,
  Filter,
  Grid,
  List,
  RefreshCw,
} from 'lucide-react';
import type { RobotStatus } from '@/types';

export default function RobotsPage() {
  const { robots, isLoading, mutate } = useRobots();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<RobotStatus | 'all'>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Filter robots
  const filteredRobots = robots.filter((robot) => {
    const matchesSearch =
      robot.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      robot.device_id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || robot.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Count by status
  const statusCounts = {
    all: robots.length,
    online: robots.filter((r) => r.status === 'online').length,
    offline: robots.filter((r) => r.status === 'offline').length,
    training: robots.filter((r) => r.status === 'training').length,
    error: robots.filter((r) => r.status === 'error').length,
  };

  const statusFilters: { value: RobotStatus | 'all'; label: string; color: string }[] = [
    { value: 'all', label: 'All', color: 'bg-gray-500' },
    { value: 'online', label: 'Online', color: 'bg-green-500' },
    { value: 'offline', label: 'Offline', color: 'bg-gray-400' },
    { value: 'training', label: 'Training', color: 'bg-blue-500' },
    { value: 'error', label: 'Error', color: 'bg-red-500' },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold">Robot Fleet</h1>
            <p className="text-muted-foreground">
              Manage and monitor your connected robots
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={() => mutate()}>
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Robot
            </Button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search robots..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Status Filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <Filter className="h-4 w-4 text-muted-foreground" />
            {statusFilters.map((filter) => (
              <Button
                key={filter.value}
                variant={statusFilter === filter.value ? 'default' : 'outline'}
                size="sm"
                onClick={() => setStatusFilter(filter.value)}
                className="gap-2"
              >
                <span className={`w-2 h-2 rounded-full ${filter.color}`} />
                {filter.label}
                <Badge variant="secondary" className="ml-1">
                  {statusCounts[filter.value]}
                </Badge>
              </Button>
            ))}
          </div>

          {/* View Toggle */}
          <div className="flex items-center gap-1 border rounded-md p-1">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'ghost'}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode('grid')}
            >
              <Grid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'ghost'}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode('list')}
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Robot Grid/List */}
        {isLoading ? (
          <div className={viewMode === 'grid'
            ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
            : "space-y-4"
          }>
            {[1, 2, 3, 4, 5, 6].map((i) => (
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
        ) : filteredRobots.length > 0 ? (
          <div className={viewMode === 'grid'
            ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
            : "space-y-4"
          }>
            {filteredRobots.map((robot) => (
              <RobotCard key={robot.id} robot={robot} />
            ))}
          </div>
        ) : (
          <Card className="border-dashed">
            <CardContent className="p-12 text-center">
              <Bot className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              {searchQuery || statusFilter !== 'all' ? (
                <>
                  <h3 className="text-lg font-semibold mb-2">No robots found</h3>
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
                  <h3 className="text-lg font-semibold mb-2">No robots connected</h3>
                  <p className="text-muted-foreground mb-4">
                    Connect your first robot to get started with fleet management.
                  </p>
                  <Button>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Your First Robot
                  </Button>
                </>
              )}
            </CardContent>
          </Card>
        )}

        {/* Summary Footer */}
        {!isLoading && filteredRobots.length > 0 && (
          <div className="mt-6 pt-6 border-t flex items-center justify-between text-sm text-muted-foreground">
            <span>
              Showing {filteredRobots.length} of {robots.length} robots
            </span>
            <span>
              {statusCounts.online} online | {statusCounts.training} training | {statusCounts.error} errors
            </span>
          </div>
        )}
      </main>
    </div>
  );
}
