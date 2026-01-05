'use client';

import { useState } from 'react';
import { useModels, useModelTypes } from '@/hooks/useModels';
import { ModelCard } from '@/components/ModelCard';
import { Navbar } from '@/components/Navbar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import {
  Box,
  Search,
  Filter,
  Grid,
  List,
  RefreshCw,
  Download,
  Upload,
} from 'lucide-react';
import type { ModelType, ModelStatus } from '@/types';

export default function ModelsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState<ModelType | 'all'>('all');
  const [statusFilter, setStatusFilter] = useState<ModelStatus | 'all'>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  const { models, allModels, isLoading, mutate } = useModels({
    type: typeFilter === 'all' ? undefined : typeFilter,
    status: statusFilter === 'all' ? undefined : statusFilter,
    search: searchQuery,
  });

  const modelTypes = useModelTypes();

  // Status filters
  const statusFilters: { value: ModelStatus | 'all'; label: string; color: string }[] = [
    { value: 'all', label: 'All', color: 'bg-gray-500' },
    { value: 'ready', label: 'Ready', color: 'bg-green-500' },
    { value: 'training', label: 'Training', color: 'bg-blue-500' },
    { value: 'failed', label: 'Failed', color: 'bg-red-500' },
    { value: 'deprecated', label: 'Deprecated', color: 'bg-orange-500' },
  ];

  // Count models by status
  const statusCounts = {
    all: allModels.length,
    ready: allModels.filter((m) => m.status === 'ready').length,
    training: allModels.filter((m) => m.status === 'training').length,
    failed: allModels.filter((m) => m.status === 'failed').length,
    deprecated: allModels.filter((m) => m.status === 'deprecated').length,
  };

  const handleDownload = async (modelId: string) => {
    // In a real app, this would trigger the download
    console.log('Downloading model:', modelId);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold">Model Browser</h1>
            <p className="text-muted-foreground">
              Browse, download, and manage your trained models
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={() => mutate()}>
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="outline">
              <Upload className="h-4 w-4 mr-2" />
              Upload Model
            </Button>
          </div>
        </div>

        {/* Filters */}
        <div className="space-y-4 mb-6">
          {/* Search and View Toggle */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search models..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
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

          {/* Type Filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm text-muted-foreground">Type:</span>
            <Button
              variant={typeFilter === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setTypeFilter('all')}
            >
              All Types
            </Button>
            {modelTypes.map((type) => (
              <Button
                key={type.value}
                variant={typeFilter === type.value ? 'default' : 'outline'}
                size="sm"
                onClick={() => setTypeFilter(type.value)}
              >
                {type.label}
              </Button>
            ))}
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
        </div>

        {/* Model Grid/List */}
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
                  <div className="h-16 bg-muted rounded mb-4" />
                  <div className="grid grid-cols-2 gap-2">
                    <div className="h-3 bg-muted rounded" />
                    <div className="h-3 bg-muted rounded" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : models.length > 0 ? (
          <div className={viewMode === 'grid'
            ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
            : "space-y-4"
          }>
            {models.map((model) => (
              <ModelCard key={model.id} model={model} onDownload={handleDownload} />
            ))}
          </div>
        ) : (
          <Card className="border-dashed">
            <CardContent className="p-12 text-center">
              <Box className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              {searchQuery || typeFilter !== 'all' || statusFilter !== 'all' ? (
                <>
                  <h3 className="text-lg font-semibold mb-2">No models found</h3>
                  <p className="text-muted-foreground mb-4">
                    Try adjusting your search or filter criteria.
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setSearchQuery('');
                      setTypeFilter('all');
                      setStatusFilter('all');
                    }}
                  >
                    Clear Filters
                  </Button>
                </>
              ) : (
                <>
                  <h3 className="text-lg font-semibold mb-2">No models available</h3>
                  <p className="text-muted-foreground mb-4">
                    Train your first model to see it here.
                  </p>
                  <Button>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Model
                  </Button>
                </>
              )}
            </CardContent>
          </Card>
        )}

        {/* Summary Footer */}
        {!isLoading && models.length > 0 && (
          <div className="mt-6 pt-6 border-t flex items-center justify-between text-sm text-muted-foreground">
            <span>
              Showing {models.length} of {allModels.length} models
            </span>
            <span>
              {statusCounts.ready} ready | {statusCounts.training} training
            </span>
          </div>
        )}
      </main>
    </div>
  );
}
