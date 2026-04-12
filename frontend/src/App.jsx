import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchTasks, resetEnv, autoRun, healthCheck, customRun } from './api';
import TrainingChart from './TrainingChart';

/* ================================================================
   PharmacistEnv -- Clinical Decision Intelligence Dashboard
   
   Frontend displays results from a TRAINED RL model.
   The model is trained offline via `python train.py`.
   The backend loads the trained Q-table and runs inference.
   ================================================================ */

const SEV_STYLES = {
  critical: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', dot: 'bg-red-500' },
  high: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', dot: 'bg-amber-500' },
  moderate: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200', dot: 'bg-blue-500' },
  low: { bg: 'bg-gray-50', text: 'text-gray-600', border: 'border-gray-200', dot: 'bg-gray-400' },
};

const DIFF_STYLES = {
  easy: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  medium: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  hard: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
};

const ACTION_LABELS = {
  extract_medicine: { label: 'Extract Medicine', color: 'bg-violet-500' },
  check_interaction: { label: 'Check Interaction', color: 'bg-red-500' },
  ask_patient_info: { label: 'Patient Info', color: 'bg-blue-500' },
  search_inventory: { label: 'Search Inventory', color: 'bg-amber-500' },
  suggest_alternative: { label: 'Suggest Alternative', color: 'bg-emerald-500' },
  risk_assessment: { label: 'Risk Assessment', color: 'bg-orange-500' },
  finalize: { label: 'Finalize Decision', color: 'bg-primary-600' },
};

const Icons = {
  shield: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>,
  beaker: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>,
  chart: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
  alert: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>,
  check: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>,
  arrow: <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>,
  play: <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>,
  back: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>,
};


// ================================================================
// TopBar
// ================================================================
function TopBar({ connected, modelInfo, view, onBack, onViewChange }) {
  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="flex items-center justify-between px-6 h-14">
        <div className="flex items-center gap-4">
          {view !== 'dashboard' && (
            <button onClick={onBack} className="flex items-center gap-1.5 text-gray-500 hover:text-gray-800 text-sm font-medium transition-colors">
              {Icons.back}
            </button>
          )}
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-primary-600 flex items-center justify-center text-white">
              {Icons.shield}
            </div>
            <span className="text-sm font-bold text-gray-900 tracking-tight">PharmacistEnv</span>
            <span className="text-[10px] text-gray-400 font-medium hidden md:block">Clinical Decision Intelligence</span>
          </div>
        </div>

        <div className="flex items-center gap-5">
          {/* Model status */}
          {modelInfo && (
            <div className="hidden md:flex items-center gap-3 text-[10px]">
              <button 
                onClick={() => onViewChange?.('training')}
                className={`flex items-center gap-2 font-semibold px-2 py-0.5 rounded-md border hover:bg-opacity-80 transition-opacity ${
                modelInfo.model_loaded
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : 'bg-amber-50 text-amber-700 border-amber-200'
              }`}>
                {Icons.chart}
                {modelInfo.model_loaded ? 'Trained Model' : 'Untrained'}
              </button>
              {modelInfo.model_loaded && (
                <span className="text-gray-400">{modelInfo.episodes_trained} episodes | {modelInfo.q_table_size} Q-values</span>
              )}
            </div>
          )}
          {/* Connection */}
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-500 animate-pulse-dot' : 'bg-red-400'}`} />
            <span className={`text-xs font-medium ${connected ? 'text-emerald-600' : 'text-red-500'}`}>
              {connected ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}


// ================================================================
// Dashboard
// ================================================================
function Dashboard({ tasks, onStart, onCustomStart, modelInfo }) {
  const [tab, setTab] = useState('demo');

  return (
    <div className="max-w-4xl mx-auto px-6 py-14 animate-fade-in">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-extrabold text-gray-900 tracking-tight mb-3">
          Clinical Decision Simulation
        </h2>
        <p className="text-gray-500 text-sm max-w-md mx-auto leading-relaxed">
          Select a scenario. The trained RL agent will analyze the prescription
          and make safety-critical decisions based on learned Q-values.
        </p>
        {modelInfo && !modelInfo.model_loaded && (
          <div className="mt-4 inline-block bg-amber-50 border border-amber-200 rounded-xl px-4 py-2.5">
            <p className="text-xs text-amber-700 font-medium">No trained model found. Run training first:</p>
            <code className="text-[11px] text-amber-800 font-mono mt-1 block">python train.py --episodes 100</code>
          </div>
        )}

        <div className="mt-8 flex justify-center">
          <div className="inline-flex bg-gray-100/80 p-1 rounded-xl shadow-inner my-4 backdrop-blur-sm border border-gray-200/50">
            <button onClick={() => setTab('demo')} className={`px-5 py-2 text-xs font-bold rounded-lg transition-all ${tab === 'demo' ? 'bg-white text-gray-900 shadow-sm ring-1 ring-gray-900/5' : 'text-gray-500 hover:text-gray-700'}`}>Demo Scenarios</button>
            <button onClick={() => setTab('custom')} className={`px-5 py-2 text-xs font-bold rounded-lg transition-all ${tab === 'custom' ? 'bg-white text-gray-900 shadow-sm ring-1 ring-gray-900/5' : 'text-gray-500 hover:text-gray-700'}`}>+ Custom Input</button>
          </div>
        </div>
        </div>

        {tab === 'demo' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
            {tasks.map((task, idx) => {
              const ds = DIFF_STYLES[task.difficulty] || DIFF_STYLES.easy;
              return (
                <button
                  key={task.name}
                  onClick={() => onStart(task.name)}
                  className="card-lift group bg-white border border-gray-200 rounded-2xl p-5 text-left
                             hover:border-primary-300 transition-all duration-200 animate-fade-in"
                  style={{ animationDelay: `${idx * 100}ms` }}
                >
                  <div className="flex justify-between items-start mb-4">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md border ${ds.bg} ${ds.text} ${ds.border}`}>
                      {task.difficulty.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-gray-400 font-medium">{task.optimal_steps} steps</span>
                  </div>
                  <h3 className="text-sm font-bold text-gray-900 mb-1.5 group-hover:text-primary-600 transition-colors">
                    {task.name.charAt(0).toUpperCase() + task.name.slice(1)} Scenario
                  </h3>
                  <p className="text-xs text-gray-500 leading-relaxed mb-4">{task.description}</p>
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-primary-600 opacity-0 group-hover:opacity-100 transition-opacity">
                    Run Trained Agent {Icons.arrow}
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {tab === 'custom' && (
          <form className="bg-white border border-gray-200 rounded-2xl p-6 mb-10 text-left animate-fade-in"
            onSubmit={(e) => {
              e.preventDefault();
              const fd = new FormData(e.target);
              onCustomStart({
                prescription_text: fd.get('prescription_text'),
                age: parseInt(fd.get('age') || '0', 10),
                gender: fd.get('gender') || 'unknown',
                weight_kg: parseFloat(fd.get('weight_kg') || '0'),
                renal_function: fd.get('renal_function') || 'normal',
                allergies: (fd.get('allergies') || '').split(',').map(s=>s.trim()).filter(Boolean),
                conditions: (fd.get('conditions') || '').split(',').map(s=>s.trim()).filter(Boolean)
              });
            }}>
            <h3 className="text-sm font-bold text-gray-900 mb-4">Patient Profile & Prescription Input</h3>
            
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-semibold text-gray-700 mb-1">Age</label>
                <input name="age" type="number" required placeholder="e.g. 35" className="w-full text-sm border-gray-300 rounded-lg p-2 border" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-700 mb-1">Gender</label>
                <select name="gender" className="w-full text-sm border-gray-300 rounded-lg p-2 border">
                  <option value="unknown">Unknown</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-700 mb-1">Weight (kg) Optional</label>
                <input name="weight_kg" type="number" step="0.1" placeholder="e.g. 60.5" className="w-full text-sm border-gray-300 rounded-lg p-2 border" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-700 mb-1">Renal Function</label>
                <select name="renal_function" className="w-full text-sm border-gray-300 rounded-lg p-2 border">
                  <option value="normal">Normal</option>
                  <option value="impaired">Impaired</option>
                  <option value="severe">Severe</option>
                </select>
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-xs font-semibold text-gray-700 mb-1">Allergies (comma separated)</label>
              <input name="allergies" type="text" placeholder="e.g. penicillin, sulfa" className="w-full text-sm border-gray-300 rounded-lg p-2 border" />
            </div>

            <div className="mb-4">
              <label className="block text-xs font-semibold text-gray-700 mb-1">Conditions (comma separated)</label>
              <input name="conditions" type="text" placeholder="e.g. hypertension, diabetes" className="w-full text-sm border-gray-300 rounded-lg p-2 border" />
            </div>

            <div className="mb-5">
              <label className="block text-xs font-semibold text-gray-700 mb-1">Prescription Text</label>
              <textarea name="prescription_text" required rows={4} placeholder="Type or paste the raw prescription text here..." className="w-full font-mono text-xs border-gray-300 rounded-lg p-3 border leading-relaxed" />
            </div>

            <button type="submit" className="w-full py-3 bg-primary-600 hover:bg-primary-700 text-white text-sm font-bold rounded-xl transition-colors shadow-sm focus:ring-4 focus:ring-primary-100">
              Run AI Simulation on Custom Patient
            </button>
          </form>
        )}

      {/* Pipeline */}
      <div className="bg-white border border-gray-200 rounded-2xl p-6">
        <h3 className="text-xs font-bold text-gray-900 mb-5 uppercase tracking-wider">Agent Decision Pipeline</h3>
        <div className="grid grid-cols-4 gap-4">
          {[
            { icon: Icons.beaker, title: 'Extract', desc: 'Parse medicines from noisy text' },
            { icon: Icons.alert, title: 'Detect', desc: 'Find interactions and allergy conflicts' },
            { icon: Icons.chart, title: 'Assess', desc: 'Evaluate patient-specific risks' },
            { icon: Icons.shield, title: 'Decide', desc: 'Make a safe clinical decision' },
          ].map((step, i) => (
            <div key={step.title} className="flex flex-col items-center text-center">
              <div className="w-9 h-9 rounded-xl bg-primary-50 border border-primary-100 flex items-center justify-center text-primary-600 mb-2">
                {step.icon}
              </div>
              <div className="text-[10px] text-gray-400 mb-0.5">Step {i + 1}</div>
              <div className="text-xs font-bold text-gray-800">{step.title}</div>
              <div className="text-[10px] text-gray-400 mt-0.5">{step.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


// ================================================================
// Simulation View
// ================================================================
function SimulationView({ obs, logs, running, onRunAgent, taskName, totalReward, grades, modelInfo, clinicalReport }) {
  const timelineRef = useRef(null);
  useEffect(() => {
    if (timelineRef.current) timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
  }, [logs]);

  if (!obs && !running) return null;

  if (!obs && running) return (
    <div className="flex flex-col items-center justify-center h-[calc(100vh-56px)] animate-fade-in text-center px-6">
       <div className="w-16 h-16 border-4 border-primary-100 border-t-primary-600 rounded-full animate-spin mb-6" />
       <h3 className="text-xl font-bold text-gray-900 mb-2">Simulating Patient Interaction...</h3>
       <p className="text-sm text-gray-500 max-w-sm">
         The Qwen AI is analyzing the prescription, assessing risks, and making clinical decisions step-by-step. This may take up to 60 seconds...
       </p>
    </div>
  );
  const progress = (obs.step_count / obs.max_steps) * 100;

  return (
    <div className="flex flex-col h-[calc(100vh-56px)] overflow-hidden">
      {/* Control bar */}
      <div className="flex items-center justify-between px-5 py-2.5 bg-white border-b border-gray-200">
        <div className="flex items-center gap-4">
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md border ${DIFF_STYLES[taskName]?.bg} ${DIFF_STYLES[taskName]?.text} ${DIFF_STYLES[taskName]?.border}`}>
            {taskName.toUpperCase()}
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Step</span>
            <span className="text-sm font-bold text-gray-900">{obs.step_count}</span>
            <span className="text-xs text-gray-400">/ {obs.max_steps}</span>
          </div>
          <div className="w-28 step-progress">
            <div className="step-progress-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Reward</span>
            <span className={`text-sm font-bold ${totalReward >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
              {totalReward >= 0 ? '+' : ''}{totalReward.toFixed(2)}
            </span>
          </div>
          {!obs.done && !running && (
            <button onClick={onRunAgent}
              className="flex items-center gap-2 px-4 py-1.5 bg-primary-600 hover:bg-primary-700 text-white text-xs font-semibold rounded-xl transition-all shadow-sm">
              {Icons.play} Run Trained Agent
            </button>
          )}
          {running && (
            <div className="flex items-center gap-2 px-4 py-1.5 bg-primary-50 text-primary-600 text-xs font-semibold rounded-xl border border-primary-200">
              <div className="w-3 h-3 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
              Agent Running...
            </div>
          )}
          {obs.done && !running && (
            <div className="flex items-center gap-2 px-4 py-1.5 bg-emerald-50 text-emerald-700 text-xs font-semibold rounded-xl border border-emerald-200">
              {Icons.check} Complete
            </div>
          )}
        </div>
      </div>

      {/* 3 Columns */}
      <div className="flex-1 grid grid-cols-12 gap-0 overflow-hidden bg-gray-50">
        {/* LEFT */}
        <div className="col-span-3 bg-white border-r border-gray-200 overflow-y-auto p-4 space-y-5 animate-slide-left">
          <PanelSection title="Prescription" icon={Icons.beaker}>
            <pre className="text-[12px] text-gray-700 whitespace-pre-wrap leading-relaxed bg-gray-50 rounded-xl p-3 border border-gray-200 font-mono">
              {obs.prescription_text}
            </pre>
          </PanelSection>

          <PanelSection title="Patient Profile" icon={Icons.shield}>
            <div className="bg-gray-50 rounded-xl p-3 border border-gray-200 space-y-2">
              <DetailRow label="Age" value={`${obs.patient_profile.age} yrs`} />
              <DetailRow label="Gender" value={obs.patient_profile.gender} />
              <DetailRow label="Weight" value={obs.patient_profile.weight_kg ? `${obs.patient_profile.weight_kg} kg` : '--'} />
              <DetailRow label="Renal" value={obs.patient_profile.renal_function} />
            </div>
            <TagGroup label="Allergies" items={obs.patient_profile.allergies} style="red" />
            <TagGroup label="Conditions" items={obs.patient_profile.conditions} style="blue" />
            <TagGroup label="Current Meds" items={obs.patient_profile.current_medications} style="violet" />
          </PanelSection>

          {obs.extracted_medicines.length > 0 && (
            <PanelSection title="Extracted Medicines" count={obs.extracted_medicines.length}>
              <div className="space-y-1.5">
                {obs.extracted_medicines.map((m, i) => (
                  <div key={i} className="bg-gray-50 rounded-lg p-2.5 border border-gray-200 animate-scale-in" style={{ animationDelay: `${i * 60}ms` }}>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-gray-800">{m.name}</span>
                      {m.confidence > 0 && <span className="text-[9px] font-bold text-gray-400">{(m.confidence * 100).toFixed(0)}%</span>}
                    </div>
                    <div className="text-[10px] text-gray-500 mt-0.5">{[m.dosage, m.frequency].filter(Boolean).join(' | ') || '...'}</div>
                  </div>
                ))}
              </div>
            </PanelSection>
          )}
        </div>

        {/* CENTER */}
        <div className="col-span-6 overflow-y-auto p-5" ref={timelineRef}>
          {logs.length === 0 && !running && (
            <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
              <div className="w-14 h-14 rounded-2xl bg-primary-50 border border-primary-100 flex items-center justify-center text-primary-500 mb-3">
                {Icons.play}
              </div>
              <h3 className="text-base font-bold text-gray-800 mb-1">Ready to Simulate</h3>
              <p className="text-xs text-gray-400 max-w-xs">
                Click "Run Trained Agent" to see the model's decisions.
                {modelInfo?.model_loaded
                  ? ` Trained on ${modelInfo.episodes_trained} episodes.`
                  : ' Note: No trained model loaded.'}
              </p>
            </div>
          )}

          {logs.length > 0 && (
            <div className="space-y-0">
              {logs.map((log, i) => {
                const meta = ACTION_LABELS[log.action_type] || { label: log.action_type, color: 'bg-gray-500' };
                const isLast = i === logs.length - 1;
                return (
                  <div key={i} className="relative animate-fade-in" style={{ animationDelay: `${i * 40}ms` }}>
                    <div className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className={`w-9 h-9 rounded-full flex items-center justify-center text-xs font-bold text-white ${meta.color} shadow-sm`}>
                          {log.step}
                        </div>
                        {!isLast && <div className="w-0.5 h-6 bg-gray-200 my-1" />}
                      </div>
                      <div className="flex-1 pb-3">
                        <div className="bg-white rounded-xl p-3.5 border border-gray-200 shadow-sm">
                          <div className="flex justify-between items-center mb-1.5">
                            <span className="text-xs font-bold text-gray-800">{meta.label}</span>
                            <span className={`text-xs font-bold ${log.reward >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                              {log.reward >= 0 ? '+' : ''}{log.reward.toFixed(2)}
                            </span>
                          </div>
                          {log.reasoning && <p className="text-[10px] text-gray-500 leading-relaxed">{log.reasoning}</p>}
                          {log.reward_components && Object.keys(log.reward_components).length > 0 && (
                            <div className="mt-2 pt-2 border-t border-gray-100 flex flex-wrap gap-1">
                              {Object.entries(log.reward_components).map(([k, v]) => (
                                <span key={k} className={`text-[9px] font-medium px-1.5 py-0.5 rounded ${v >= 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'}`}>
                                  {k.replace(/_/g, ' ')}: {v >= 0 ? '+' : ''}{v.toFixed(2)}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Final Decision */}
          {obs.final_decision && (
            <div className="mt-4 animate-scale-in">
              <div className="bg-white rounded-2xl p-5 border-2 border-primary-200 shadow-md shadow-primary-100/30">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-7 h-7 rounded-lg bg-primary-600 flex items-center justify-center text-white">{Icons.shield}</div>
                  <div>
                    <h3 className="text-sm font-bold text-gray-900">Clinical Decision</h3>
                    <span className={`text-xs font-semibold ${
                      obs.final_decision.decision === 'dispense' ? 'text-emerald-600'
                      : obs.final_decision.decision === 'modify' ? 'text-primary-600' : 'text-red-600'
                    }`}>
                      {obs.final_decision.decision?.toUpperCase()} | Confidence: {((obs.final_decision.confidence || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {obs.final_decision.medications && (
                  <div className="mb-3">
                    <span className="text-[9px] font-bold text-gray-500 uppercase tracking-wider block mb-1.5">Final Medications</span>
                    <div className="grid grid-cols-2 gap-1.5">
                      {obs.final_decision.medications.map((m, i) => (
                        <div key={i} className="flex items-center gap-2 bg-gray-50 rounded-lg p-2 border border-gray-200">
                          <div className="w-1.5 h-1.5 rounded-full bg-primary-500" />
                          <div>
                            <div className="text-[11px] font-semibold text-gray-800">{m.name}</div>
                            <div className="text-[9px] text-gray-500">{m.dosage} {m.frequency}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {obs.final_decision.reasoning && (
                  <div className="bg-primary-50 rounded-xl p-3 border border-primary-100">
                    <span className="text-[9px] font-bold text-primary-600 uppercase tracking-wider block mb-1">Clinical Reasoning</span>
                    <p className="text-[11px] text-primary-800 leading-relaxed">{obs.final_decision.reasoning}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Clinical Report / Professional Clinical Intelligence Dashboard */}
          {clinicalReport && (
            <div className="mt-6 animate-scale-in">
              {typeof clinicalReport === 'object' ? (
                <div className="bg-white rounded-2xl border-2 border-primary-100 shadow-xl overflow-hidden mb-8">
                  {/* Deep Diagnostic Header */}
                  <div className="bg-primary-600 p-4 flex justify-between items-center text-white">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center text-xl">
                        {Icons.shield}
                      </div>
                      <div>
                        <h3 className="text-lg font-bold leading-none mb-1">Senior Pharmacist Review</h3>
                        <p className="text-[10px] text-primary-100 uppercase tracking-widest font-bold">Clinical Intelligence v2.0</p>
                      </div>
                    </div>
                    <div className={`px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-widest shadow-inner ${
                      clinicalReport.verdict === 'Safe' ? 'bg-emerald-500' :
                      clinicalReport.verdict === 'Warning' ? 'bg-amber-500' : 'bg-red-500'
                    }`}>
                      {clinicalReport.verdict}
                    </div>
                  </div>

                  <div className="p-6">
                    {/* Executive Summary Section */}
                    <div className="mb-6">
                      <div className="text-sm font-bold text-gray-900 mb-2 flex items-center gap-2">
                        <span className="w-1 h-4 bg-primary-600 rounded-full" />
                        Executive Summary
                      </div>
                      <div className="bg-gray-50 rounded-xl p-4 text-sm text-gray-700 leading-relaxed italic border-l-4 border-primary-200">
                        "{clinicalReport.summary || clinicalReport}"
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                      {/* Patient Analysis Card */}
                      <div className="flex flex-col">
                        <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-2 ml-1">
                          Patient Assessment
                        </div>
                        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className={`text-[9px] font-black px-2 py-0.5 rounded-full uppercase tracking-widest ${
                              clinicalReport.patient_analysis?.risk_level === 'Low' ? 'bg-emerald-100 text-emerald-700' :
                              clinicalReport.patient_analysis?.risk_level === 'Moderate' ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                            }`}>
                              Risk: {clinicalReport.patient_analysis?.risk_level}
                            </span>
                          </div>
                          <p className="text-xs text-gray-600 leading-relaxed font-medium">
                            {clinicalReport.patient_analysis?.details}
                          </p>
                        </div>
                      </div>

                      {/* Recommendation Card */}
                      <div className="flex flex-col">
                        <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-2 ml-1">
                          Final Recommendation
                        </div>
                        <div className="bg-primary-50 border border-primary-100 rounded-xl p-4 shadow-sm flex-1">
                          <p className="text-xs text-primary-900 leading-relaxed font-bold italic">
                            {clinicalReport.final_recommendation}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Reasoning Visualization */}
                    {clinicalReport.reasoning_steps && clinicalReport.reasoning_steps.length > 0 && (
                      <div className="mt-8">
                        <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4 ml-1">Diagnostic Workflow Validation</div>
                        <div className="grid grid-cols-1 gap-3">
                          {clinicalReport.reasoning_steps.map((s, i) => (
                            <div key={i} className="flex gap-4 items-center bg-gray-50/50 p-3 rounded-xl border border-gray-100">
                              <div className="w-6 h-6 rounded-full bg-white border border-gray-200 flex items-center justify-center text-[10px] font-extrabold text-primary-600 shadow-sm">
                                0{i + 1}
                              </div>
                              <div className="flex-1">
                                <div className="text-[11px] font-bold text-gray-800">{s.step}</div>
                                <div className="text-[10px] text-gray-500 italic">{s.observation}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                  </div>
                </div>
              ) : (
                /* Fallback for simple string reports */
                <div className="bg-primary-50 rounded-2xl p-6 border-2 border-primary-100 shadow-sm border-l-4 border-l-primary-600 mb-8 font-medium text-sm text-primary-900 whitespace-pre-wrap">
                  {clinicalReport}
                </div>
              )}
            </div>
          )}

          {/* Grades */}
          {grades && obs.done && (
            <div className="mt-4 bg-white border border-gray-200 rounded-2xl p-4 animate-fade-in">
              <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-3">Grading Results</h4>
              <div className="grid grid-cols-4 gap-4">
                {[
                  { label: 'Accuracy', value: grades.accuracy, weight: '35%' },
                  { label: 'Safety', value: grades.safety, weight: '45%' },
                  { label: 'Efficiency', value: grades.efficiency, weight: '20%' },
                  { label: 'Final Score', value: grades.final_score, weight: '' },
                ].map(g => (
                  <div key={g.label} className="text-center">
                    <div className={`text-lg font-bold ${g.value >= 0.9 ? 'text-emerald-600' : g.value >= 0.5 ? 'text-amber-600' : 'text-red-600'}`}>
                      {(g.value * 100).toFixed(0)}%
                    </div>
                    <div className="text-[10px] text-gray-500 font-semibold">{g.label}</div>
                    {g.weight && <div className="text-[9px] text-gray-400">weight: {g.weight}</div>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT */}
        <div className="col-span-3 bg-white border-l border-gray-200 overflow-y-auto p-4 space-y-5 animate-slide-right">
          <PanelSection title="Pharmacy Inventory" count={obs.inventory.length}>
            <div className="space-y-0 divide-y divide-gray-100">
              {obs.inventory.map((item, i) => (
                <div key={i} className="flex justify-between items-center py-2">
                  <div>
                    <div className="text-xs font-semibold text-gray-800">{item.name}</div>
                    <div className="text-[9px] text-gray-400">{item.dosage_form} | ${item.price.toFixed(2)}</div>
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md border ${
                    item.stock > 0 ? 'text-emerald-600 bg-emerald-50 border-emerald-200' : 'text-red-600 bg-red-50 border-red-200'
                  }`}>
                    {item.stock > 0 ? item.stock : 'OUT'}
                  </span>
                </div>
              ))}
            </div>
          </PanelSection>

          {obs.detected_interactions.length > 0 && (
            <PanelSection title="Drug Interactions" count={obs.detected_interactions.length}>
              <div className="space-y-1.5">
                {obs.detected_interactions.map((intr, i) => {
                  const s = SEV_STYLES[intr.severity] || SEV_STYLES.moderate;
                  return (
                    <div key={i} className={`rounded-xl p-2.5 border animate-scale-in ${s.bg} ${s.border}`} style={{ animationDelay: `${i * 80}ms` }}>
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
                        <span className={`text-[10px] font-bold ${s.text}`}>{intr.severity.toUpperCase()}</span>
                      </div>
                      <div className={`text-[10px] font-semibold ${s.text}`}>{intr.drug_a} + {intr.drug_b}</div>
                      <p className="text-[9px] text-gray-600 mt-0.5">{intr.description}</p>
                    </div>
                  );
                })}
              </div>
            </PanelSection>
          )}

          {obs.risk_flags.length > 0 && (
            <PanelSection title="Safety Alerts" count={obs.risk_flags.length}>
              <div className="space-y-1.5">
                {obs.risk_flags.map((flag, i) => {
                  const s = SEV_STYLES[flag.severity] || SEV_STYLES.moderate;
                  return (
                    <div key={i} className={`rounded-xl p-2.5 border animate-scale-in ${s.bg} ${s.border}`} style={{ animationDelay: `${i * 80}ms` }}>
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
                        <span className={`text-[9px] font-bold uppercase ${s.text}`}>{flag.category}</span>
                        <span className="text-[9px] text-gray-400">{flag.affected_drug}</span>
                      </div>
                      <p className="text-[9px] text-gray-600">{flag.description}</p>
                    </div>
                  );
                })}
              </div>
            </PanelSection>
          )}

          {obs.clinical_notes.length > 0 && (
            <PanelSection title="Clinical Notes" count={obs.clinical_notes.length}>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {obs.clinical_notes.map((note, i) => (
                  <div key={i} className="text-[9px] text-gray-500 border-l-2 border-gray-200 pl-2 py-0.5">{note}</div>
                ))}
              </div>
            </PanelSection>
          )}
        </div>
      </div>
    </div>
  );
}


// ================================================================
// Shared Components
// ================================================================
function PanelSection({ title, icon, count, children }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          {icon && <span className="text-gray-400">{icon}</span>}
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{title}</h3>
        </div>
        {count !== undefined && <span className="text-[9px] font-bold text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">{count}</span>}
      </div>
      {children}
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[11px] text-gray-500">{label}</span>
      <span className="text-[11px] font-semibold text-gray-800">{value}</span>
    </div>
  );
}

function TagGroup({ label, items, style }) {
  if (!items || items.length === 0) return null;
  const styles = {
    red: 'bg-red-50 text-red-700 border-red-200',
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    violet: 'bg-violet-50 text-violet-700 border-violet-200',
  };
  return (
    <div className="mt-2">
      <span className="text-[9px] font-bold text-gray-500 uppercase tracking-wider block mb-1">{label}</span>
      <div className="flex flex-wrap gap-1">
        {items.map((item, i) => (
          <span key={i} className={`text-[10px] font-medium px-2 py-0.5 rounded-md border ${styles[style]}`}>{item}</span>
        ))}
      </div>
    </div>
  );
}

// ================================================================
// Training Progress View
// ================================================================
function TrainingProgressView({ history }) {
  return (
    <div className="max-w-4xl mx-auto px-6 py-14 animate-fade-in w-full h-full">
      <TrainingChart history={history} />
    </div>
  );
}


// ================================================================
// Main App
// ================================================================
export default function App() {
  const [view, setView] = useState('dashboard');
  const [connected, setConnected] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [obs, setObs] = useState(null);
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const [taskName, setTaskName] = useState('');
  const [totalReward, setTotalReward] = useState(0);
  const [grades, setGrades] = useState(null);
  const [clinicalReport, setClinicalReport] = useState('');
  const [trainingHistory, setTrainingHistory] = useState([]);

  useEffect(() => {
    const check = async () => {
      const ok = await healthCheck();
      setConnected(ok);
      if (ok) {
        try {
          // Get model status from health
          const res = await fetch('/api/health');
          const health = await res.json();
          setModelInfo(health);

          const histRes = await fetch('/api/training-history');
          const histData = await histRes.json();
          setTrainingHistory(histData.history || []);

          const data = await fetchTasks();
          setTasks(data.tasks || []);
        } catch { /* ignore */ }
      }
    };
    check();
    const iv = setInterval(check, 5000);
    return () => clearInterval(iv);
  }, []);

  const handleStart = useCallback(async (name) => {
    try {
      const data = await resetEnv(name);
      setObs(data.observation);
      setLogs([]);
      setTaskName(name);
      setTotalReward(0);
      setGrades(null);
      setView('simulation');
    } catch (err) {
      console.error('Reset failed:', err);
    }
  }, []);

  const handleCustomStart = useCallback(async (payload) => {
    try {
      // Clear up the view
      setObs(null);
      setLogs([]);
      setTaskName("custom_user_task");
      setTotalReward(0);
      setGrades(null);
      setClinicalReport('');
      setView('simulation');
      setRunning(true);

      const data = await customRun(payload);
      
      // Animate steps visually
      for (let i = 0; i < data.steps.length; i++) {
        const step = data.steps[i];
        if (step.observation) setObs(step.observation);
        const logEntry = step.observation?.action_history?.[step.observation.action_history.length - 1] || {
          step: step.step, action_type: step.action_type, reward: step.reward,
        };
        setLogs(prev => [...prev, logEntry]);
        // Cumulative reward manually tracked if not provided
        setTotalReward(prev => prev + (step.reward || logEntry.reward || 0));
        await new Promise(r => setTimeout(r, 800));
      }

      if (data.clinical_report) {
        setClinicalReport(data.clinical_report);
      }
      
    } catch (err) {
      console.error('Custom run failed:', err);
    } finally {
      setRunning(false);
    }
  }, []);

  // Run the TRAINED agent via /auto-run
  const handleRunAgent = useCallback(async () => {
    if (!taskName) return;
    setRunning(true);
    setClinicalReport('');

    try {
      const data = await autoRun(taskName);

      // Animate steps
      for (let i = 0; i < data.steps.length; i++) {
        const step = data.steps[i];
        if (step.observation) setObs(step.observation);
        const logEntry = step.observation?.action_history?.[step.observation.action_history.length - 1] || {
          step: step.step, action_type: step.action_type, reward: step.reward,
        };
        setLogs(prev => [...prev, logEntry]);
        setTotalReward(step.cumulative_reward || 0);
        await new Promise(r => setTimeout(r, 800));
      }

      if (data.grades) setGrades(data.grades);
      if (data.clinical_report) setClinicalReport(data.clinical_report);
    } catch (err) {
      console.error('Auto-run failed:', err);
    }
    setRunning(false);
  }, [taskName]);

  const handleBack = useCallback(() => {
    setView('dashboard');
    setObs(null); setLogs([]); setTaskName(''); setTotalReward(0); setGrades(null); setClinicalReport('');
  }, []);

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <TopBar connected={connected} modelInfo={modelInfo} view={view} onBack={handleBack} onViewChange={setView} />

      {!connected && view === 'dashboard' && (
        <div className="max-w-md mx-auto px-6 py-20 text-center animate-fade-in">
          <h2 className="text-xl font-bold text-gray-900 mb-3">Backend Not Connected</h2>
          <p className="text-sm text-gray-500 mb-6">Start the server:</p>
          <div className="bg-gray-900 rounded-xl p-4 text-left">
            <pre className="text-xs text-gray-300 font-mono">
              <span className="text-gray-500">$</span> python train.py --episodes 100{'\n'}
              <span className="text-gray-500">$</span> uvicorn server.app:app --port 8000
            </pre>
          </div>
        </div>
      )}

      {connected && view === 'dashboard' && <Dashboard tasks={tasks} onStart={handleStart} onCustomStart={handleCustomStart} modelInfo={modelInfo} />}
      {connected && view === 'training' && <TrainingProgressView history={trainingHistory} />}
      {view === 'simulation' && (
        <SimulationView 
          obs={obs} logs={logs} running={running} 
          onRunAgent={handleRunAgent} taskName={taskName} totalReward={totalReward} 
          grades={grades} modelInfo={modelInfo}
          clinicalReport={clinicalReport}
        />
      )}
    </div>
  );
}
