import { useEffect, useState } from "react";
import { ShieldCheck, ShieldAlert, Download } from "lucide-react";
import { getSplit, getLeakage, API } from "@/lib/api";

export default function LeakageReport() {
  const [split, setSplit] = useState(null);
  const [leak, setLeak] = useState(null);

  useEffect(() => {
    getSplit().then(setSplit).catch(() => {});
    getLeakage().then(setLeak).catch(() => {});
  }, []);

  const downloadJson = () => {
    const blob = new Blob([JSON.stringify({ split, leakage: leak }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "roofline_split_leakage.json"; a.click();
    URL.revokeObjectURL(url);
  };

  const bySplit = { train: [], val: [], holdout: [] };
  (split?.groups || []).forEach((g) => { if (bySplit[g.split]) bySplit[g.split].push(g); });

  return (
    <div className="max-w-6xl mx-auto px-4 md:px-6 py-6">
      <div className="flex items-center gap-3 mb-6">
        <h2 className="font-display text-2xl font-semibold">Split & Leakage Report</h2>
        <button data-testid="download-split" onClick={downloadJson}
          className="ml-auto flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm">
          <Download className="w-4 h-4" /> Export JSON
        </button>
      </div>

      {leak && (
        <div data-testid="leakage-banner" className={`p-5 rounded-xl border mb-6 flex items-center gap-4 ${
          leak.passed ? "bg-emerald-950/40 border-emerald-500/40" : "bg-rose-950/40 border-rose-500/40"}`}>
          {leak.passed ? <ShieldCheck className="w-8 h-8 text-emerald-400" />
                       : <ShieldAlert className="w-8 h-8 text-rose-400" />}
          <div>
            <div className={`font-display text-xl font-bold ${leak.passed ? "text-emerald-300" : "text-rose-300"}`}>
              {leak.passed ? "LEAKAGE CHECK PASSED" : "LEAKAGE DETECTED"}
            </div>
            <div className="text-sm text-slate-400 mt-0.5">{leak.message}</div>
            <div className="text-xs font-mono text-slate-500 mt-1">
              {leak.groups_checked} house_groups · {leak.images_checked} images · split at house_group level
            </div>
          </div>
        </div>
      )}

      {split && (
        <div className="grid md:grid-cols-3 gap-4">
          {["train", "val", "holdout"].map((s) => (
            <div key={s} data-testid={`split-col-${s}`} className="rounded-xl border border-slate-800 bg-[#111827] p-4">
              <div className="flex items-center justify-between mb-3">
                <span className={`font-display font-semibold capitalize ${
                  s === "train" ? "text-blue-300" : s === "val" ? "text-amber-300" : "text-violet-300"}`}>{s}</span>
                <span className="text-xs font-mono text-slate-500">
                  {split.group_counts?.[s]} grp · {split.image_counts?.[s]} img
                </span>
              </div>
              <div className="max-h-80 overflow-y-auto space-y-1 pr-1">
                {bySplit[s].map((g) => (
                  <div key={g.group_id} className="flex items-center justify-between text-xs font-mono px-2 py-1 rounded bg-[#0B0F19] border border-slate-800">
                    <span className="text-slate-400">{g.group_id}</span>
                    <span className="text-slate-600">{g.size}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {leak && Object.keys(leak.leaks || {}).length > 0 && (
        <div className="mt-6 p-4 rounded-xl border border-rose-500/40 bg-rose-950/30">
          <h3 className="font-semibold text-rose-300 mb-2">Leaking groups</h3>
          <pre className="text-xs font-mono text-rose-200 overflow-x-auto">{JSON.stringify(leak.leaks, null, 2)}</pre>
        </div>
      )}

      <div className="mt-6 text-xs text-slate-600 font-mono">
        CLI equivalent: <span className="text-slate-400">python scripts/check_split_leakage.py</span>
      </div>
    </div>
  );
}
