import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';

export default function TrainingChart({ history }) {
  if (!history || history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 bg-white rounded-2xl border border-gray-200">
        <p className="text-gray-500 text-sm">No training history available.</p>
        <p className="text-xs text-gray-400 mt-2">Train the model to see progress.</p>
      </div>
    );
  }

  // Calculate moving averages for smoother lines
  const windowSize = 5;
  const processedData = history.map((point, index, arr) => {
    let sumScore = 0, sumSafety = 0, sumAcc = 0;
    const start = Math.max(0, index - windowSize + 1);
    const count = index - start + 1;
    
    for (let i = start; i <= index; i++) {
      sumScore += arr[i].final_score;
      sumSafety += arr[i].safety;
      sumAcc += arr[i].accuracy;
    }
    
    return {
      ...point,
      avg_score: sumScore / count,
      avg_safety: sumSafety / count,
      avg_accuracy: sumAcc / count,
      label_episode: `Ep ${point.episode}`,
    };
  });

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm overflow-hidden animate-fade-in">
      <div className="mb-6 flex justify-between items-end">
        <div>
          <h3 className="text-lg font-extrabold text-gray-900 tracking-tight">RL Training Progress</h3>
          <p className="text-xs text-gray-500 mt-1">
            Q-Learning Agent convergence over {history[history.length - 1].episode} episodes.
          </p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-black text-emerald-600">
            {(history[history.length - 1].final_score * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] uppercase font-bold text-gray-400 tracking-wider">Final Model Score</p>
        </div>
      </div>

      {/* Main Performance Chart (Smoothed) */}
      <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2">Metrics Progression (Moving Average)</h4>
      <div className="h-72 w-full mb-8">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={processedData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis dataKey="episode" tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} tickMargin={10} minTickGap={30} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
              itemStyle={{ fontSize: '11px', fontWeight: 600 }}
              labelStyle={{ fontSize: '12px', fontWeight: 800, color: '#1e293b', marginBottom: '8px' }}
              labelFormatter={(val) => `Episode ${val}`}
            />
            <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
            <Line type="monotone" name="Overall Score" dataKey="avg_score" stroke="#10b981" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
            <Line type="monotone" name="Safety" dataKey="avg_safety" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" dot={false} />
            <Line type="monotone" name="Accuracy" dataKey="avg_accuracy" stroke="#3b82f6" strokeWidth={2} strokeDasharray="3 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {/* Step Efficiency Chart */}
      <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2">Learning Efficiency (Steps to completion)</h4>
      <div className="h-32 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis dataKey="episode" hide />
            <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} tickCount={4} />
            <Tooltip
              cursor={{ fill: '#f8fafc' }}
              contentStyle={{ borderRadius: '8px', border: 'none', padding: '8px 12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
              itemStyle={{ fontSize: '11px', fontWeight: 600, color: '#8b5cf6' }}
              labelStyle={{ display: 'none' }}
              formatter={(val, name, props) => [val + ' steps', `Episode ${props.payload.episode}`]}
            />
            <Bar dataKey="steps" fill="#c4b5fd" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
