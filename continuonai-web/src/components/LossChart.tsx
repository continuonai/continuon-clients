'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { LossDataPoint } from '@/types';

interface LossChartProps {
  data: LossDataPoint[];
  height?: number;
  showValidation?: boolean;
}

export function LossChart({ data, height = 200, showValidation = true }: LossChartProps) {
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-muted rounded-lg"
        style={{ height }}
      >
        <p className="text-muted-foreground text-sm">No loss data available</p>
      </div>
    );
  }

  // Format data for display
  const formattedData = data.map((point) => ({
    step: point.step,
    'Training Loss': point.loss,
    'Validation Loss': point.val_loss,
  }));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={formattedData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => {
              if (value >= 1000) {
                return `${(value / 1000).toFixed(1)}k`;
              }
              return value;
            }}
            className="text-muted-foreground"
          />
          <YAxis
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => value.toFixed(3)}
            className="text-muted-foreground"
            domain={['dataMin', 'dataMax']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            labelStyle={{
              color: 'hsl(var(--foreground))',
              fontWeight: 'bold',
            }}
            formatter={(value: number) => [value.toFixed(4), '']}
            labelFormatter={(label) => `Step ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="Training Loss"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          {showValidation && data.some((d) => d.val_loss !== undefined) && (
            <Line
              type="monotone"
              dataKey="Validation Loss"
              stroke="hsl(var(--destructive))"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              strokeDasharray="5 5"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Mini version for compact displays
export function LossChartMini({ data }: { data: LossDataPoint[] }) {
  if (!data || data.length === 0) {
    return null;
  }

  return (
    <div className="w-full h-16">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line
            type="monotone"
            dataKey="loss"
            stroke="hsl(var(--primary))"
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default LossChart;
