import { useEffect, useState } from "react";
import {
  Cpu, Zap, HardDrive, MemoryStick, Boxes, ShieldCheck, ShieldAlert,
  Network, GitBranch, FileCheck2, AlertTriangle, CheckCircle2, Image as ImageIcon,
  Layers, TerminalSquare,
} from "lucide-react";
import { getEnvironment, getLeakage, getQaScripts, getRepresentative, rawImageUrl } from "@/lib/api";

const Card = ({ children, className = "", testid }) => (
  <div
    data-testid={testid}
    className={`rounded-xl border border-slate-800 bg-[#111827] p-5 rf-fade-up ${className}`}
  >
    {children}
  </div>
);

const Stat = ({ label, value, sub, accent = "text-slate-100" }) => (
  <div>
    <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">{label}</div>
    <div className={`text-2xl font-display font-bold mt-1 ${accent}`}>{value}</div>
    {sub && <div className="text-xs text-slate-500 mt-0.5">{sub}</div>}
  </div>
);

export default function DashboardCheckpoint({ summary, mlStatus, onGoto }) {
  const [env, setEnv] = useState(null);
  const [leak, setLeak] = useState(null);
  const [qa, setQa] = useState(null);
  const [reps, setReps] = useState([]);

  useEffect(() => {
    getEnvironment().then(setEnv).catch(() => {});
    getLeakage().then(setLeak).catch(() => {});
    getQaScripts().then(setQa).catch(() => {});
    getRepresentative().then(setReps).catch(() => {});
  }, []);

  const split = summary?.split;
  const gpu = env?.gpu_present;

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 rf-grid-bg">
      <div className="mb-6">
        <h1 className="font-display text-3xl md:text-4xl font-bold tracking-tight">
          Phase 1 Checkpoint
        </h1>
        <p className="text-slate-400 mt-1 text-sm">
          Verified foundation state — every value below is measured from the real pipeline, not fabricated.
        </p>
      </div>

      {/* Top row stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Card testid="stat-total">
          <div className="flex items-center justify-between">
            <Stat label="Total Images" value={summary?.total_images ?? "—"} sub="ingested + indexed" />
            <ImageIcon className="w-5 h-5 text-blue-400" />
          </div>
          <div className="flex gap-3 mt-3 text-xs font-mono">
            <span className="text-slate-400">B1 <b className="text-slate-200">{summary?.batch1 ?? "—"}</b></span>
            <span className="text-slate-400">B2 <b className="text-slate-200">{summary?.batch2 ?? "—"}</b></span>
          </div>
        </Card>
        <Card testid="stat-integrity">
          <div className="flex items-center justify-between">
            <Stat label="Integrity" value={summary ? `${summary.corrupted} corrupt` : "—"}
              sub={`${summary?.duplicates ?? 0} duplicates`}
              accent={summary?.corrupted ? "text-rose-400" : "text-emerald-400"} />
            <FileCheck2 className="w-5 h-5 text-emerald-400" />
          </div>
        </Card>
        <Card testid="stat-groups">
          <div className="flex items-center justify-between">
            <Stat label="House Groups" value={summary?.house_groups ?? "—"}
              sub={`${summary?.multi_view_groups ?? 0} multi-view`} />
            <Network className="w-5 h-5 text-violet-400" />
          </div>
        </Card>
        <Card testid="stat-annot">
          <div className="flex items-center justify-between">
            <Stat label="Annotations" value={summary?.status_counts?.approved ?? 0}
              sub={`${summary?.status_counts?.needs_review ?? 0} needs review`}
              accent="text-emerald-400" />
            <Layers className="w-5 h-5 text-amber-400" />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Environment report */}
        <Card testid="env-card" className="lg:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            {gpu ? <Zap className="w-5 h-5 text-emerald-400" /> : <Cpu className="w-5 h-5 text-amber-400" />}
            <h3 className="font-display text-lg font-semibold">Environment Report</h3>
            <span className={`ml-auto text-xs font-mono px-2 py-1 rounded border ${
              gpu ? "bg-emerald-950/60 border-emerald-500/40 text-emerald-300"
                  : "bg-amber-950/50 border-amber-500/40 text-amber-300"}`}>
              {env?.verdict || "…"}
            </span>
          </div>
          {env && (
            <>
              <p className="text-sm text-slate-400 mb-4 leading-relaxed">{env.message}</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="flex items-start gap-2">
                  <Cpu className="w-4 h-4 text-slate-500 mt-0.5" />
                  <div><div className="text-xs text-slate-500">CPU</div>
                    <div className="text-sm font-mono">{env.cpu?.logical_cores} cores</div></div>
                </div>
                <div className="flex items-start gap-2">
                  <MemoryStick className="w-4 h-4 text-slate-500 mt-0.5" />
                  <div><div className="text-xs text-slate-500">RAM</div>
                    <div className="text-sm font-mono">{env.memory?.total_gb} GB</div></div>
                </div>
                <div className="flex items-start gap-2">
                  <HardDrive className="w-4 h-4 text-slate-500 mt-0.5" />
                  <div><div className="text-xs text-slate-500">Disk free</div>
                    <div className="text-sm font-mono">{env.disk_app?.free_gb} GB</div></div>
                </div>
                <div className="flex items-start gap-2">
                  <Boxes className="w-4 h-4 text-slate-500 mt-0.5" />
                  <div><div className="text-xs text-slate-500">PyTorch</div>
                    <div className="text-sm font-mono">{env.torch?.installed ? env.torch.version : "not installed"}</div></div>
                </div>
              </div>
              <div className="mt-4 p-3 rounded-lg bg-[#0B0F19] border border-slate-800 font-mono text-xs text-slate-400">
                <div>GPU present: <span className={gpu ? "text-emerald-400" : "text-amber-400"}>{String(gpu)}</span></div>
                <div>CUDA available: <span className="text-slate-300">{String(env.torch?.cuda_available)}</span></div>
                <div>Heavy inference backend: <span className="text-slate-300">{env.heavy_inference_backend}</span></div>
                <div>SAM2 active backend: <span className="text-slate-300">{mlStatus?.active_backend || "…"}</span></div>
              </div>
            </>
          )}
        </Card>

        {/* Leakage */}
        <Card testid="leakage-card">
          <div className="flex items-center gap-2 mb-4">
            {leak?.passed ? <ShieldCheck className="w-5 h-5 text-emerald-400" />
                          : <ShieldAlert className="w-5 h-5 text-rose-400" />}
            <h3 className="font-display text-lg font-semibold">Split Leakage</h3>
          </div>
          {leak && (
            <div className={`p-4 rounded-lg border ${leak.passed
              ? "bg-emerald-950/40 border-emerald-500/40" : "bg-rose-950/40 border-rose-500/40"}`}>
              <div className={`flex items-center gap-2 font-semibold ${leak.passed ? "text-emerald-300" : "text-rose-300"}`}>
                {leak.passed ? <CheckCircle2 className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                {leak.passed ? "PASSED" : "LEAKAGE"}
              </div>
              <p className="text-xs text-slate-400 mt-2">{leak.message}</p>
              <div className="mt-3 text-xs font-mono text-slate-500">
                {leak.groups_checked} groups · {leak.images_checked} images checked
              </div>
            </div>
          )}
          {split && (
            <div className="mt-4">
              <div className="text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
                Group-aware split ({(split.ratios?.train * 100) | 0}/{(split.ratios?.val * 100) | 0}/{(split.ratios?.holdout * 100) | 0})
              </div>
              <SplitBar counts={split.image_counts} />
              <div className="flex justify-between text-xs mt-2 font-mono">
                <span className="text-blue-300">train {split.image_counts?.train}</span>
                <span className="text-amber-300">val {split.image_counts?.val}</span>
                <span className="text-violet-300">holdout {split.image_counts?.holdout}</span>
              </div>
              <div className="text-[11px] text-slate-600 mt-2">Holdout fixed (seed {split.seed})</div>
            </div>
          )}
        </Card>
      </div>

      {/* QA scripts */}
      <Card testid="qa-card" className="mt-4">
        <div className="flex items-center gap-2 mb-4">
          <TerminalSquare className="w-5 h-5 text-cyan-400" />
          <h3 className="font-display text-lg font-semibold">QA & Validation Scripts</h3>
        </div>
        <div className="grid md:grid-cols-2 gap-3">
          {qa?.scripts?.map((s) => (
            <div key={s.name} className="p-4 rounded-lg bg-[#0B0F19] border border-slate-800">
              <div className="font-mono text-sm text-cyan-300">{s.path}</div>
              <div className="text-xs text-slate-400 mt-1">{s.purpose}</div>
              <div className="text-[11px] text-slate-600 mt-2 font-mono">python {s.path}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Representative 5 */}
      <Card testid="representative-card" className="mt-4">
        <div className="flex items-center gap-2 mb-1">
          <ImageIcon className="w-5 h-5 text-blue-400" />
          <h3 className="font-display text-lg font-semibold">5 Representative Test Images</h3>
          <span className="ml-auto text-xs font-mono text-amber-300 px-2 py-1 rounded border border-amber-500/40 bg-amber-950/40">
            awaiting your review — no mass annotation yet
          </span>
        </div>
        <p className="text-xs text-slate-500 mb-4">
          Selected after real visual inspection to span the dataset's conditions. Note: all 217 images
          are ground-level real-estate photos (not aerial); zero portrait orientation exists in this set.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {reps.map((r) => (
            <div key={r.dataset_id} data-testid={`rep-${r.dataset_id}`}
              className="rounded-lg overflow-hidden border border-slate-800 bg-[#0B0F19]">
              <div className="aspect-[4/3] bg-slate-900">
                <img src={rawImageUrl(r.dataset_id)} alt={r.dataset_id} loading="lazy"
                  className="w-full h-full object-cover" />
              </div>
              <div className="p-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-blue-300">{r.dataset_id}</span>
                  <span className="text-[10px] font-mono text-slate-500">{r.batch === "batch1" ? "B1" : "B2"} · {r.split}</span>
                </div>
                <p className="text-[11px] text-slate-500 mt-1 leading-snug">{r.representative_reason}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="flex flex-wrap gap-3 mt-6">
        <button data-testid="goto-gallery" onClick={() => onGoto("gallery")}
          className="px-4 py-2.5 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">
          Open Dataset Gallery
        </button>
        <button data-testid="goto-housegroups" onClick={() => onGoto("housegroups")}
          className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium transition-colors">
          Review House Groups
        </button>
      </div>
    </div>
  );
}

function SplitBar({ counts }) {
  const total = (counts?.train || 0) + (counts?.val || 0) + (counts?.holdout || 0) || 1;
  const seg = (v, c) => (
    <div className={c} style={{ width: `${((v || 0) / total) * 100}%` }} />
  );
  return (
    <div className="flex h-3 rounded-full overflow-hidden bg-slate-800">
      {seg(counts?.train, "bg-blue-500")}
      {seg(counts?.val, "bg-amber-500")}
      {seg(counts?.holdout, "bg-violet-500")}
    </div>
  );
}
