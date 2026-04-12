const API = '/api';

export async function fetchTasks() {
  const res = await fetch(`${API}/tasks`);
  if (!res.ok) throw new Error(`Failed to fetch tasks: ${res.status}`);
  return res.json();
}

export async function resetEnv(taskName) {
  const res = await fetch(`${API}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_name: taskName }),
  });
  if (!res.ok) throw new Error(`Reset failed: ${res.status}`);
  return res.json();
}

export async function stepEnv(actionType, parameters) {
  const res = await fetch(`${API}/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_type: actionType, parameters }),
  });
  if (!res.ok) throw new Error(`Step failed: ${res.status}`);
  return res.json();
}

export async function getState() {
  const res = await fetch(`${API}/state`);
  if (!res.ok) throw new Error(`State fetch failed: ${res.status}`);
  return res.json();
}

export async function autoRun(taskName) {
  const res = await fetch(`${API}/auto-run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_name: taskName, agent: 'rl' }),
  });
  if (!res.ok) throw new Error(`Auto-run failed: ${res.status}`);
  return res.json();
}

export async function customRun(payload) {
  const res = await fetch(`${API}/custom-run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Custom-run failed: ${res.status}`);
  return res.json();
}

export async function trainAgent(episodes = 30, taskName = 'all') {
  const res = await fetch(`${API}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ episodes, task_name: taskName }),
  });
  if (!res.ok) throw new Error(`Training failed: ${res.status}`);
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${API}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
