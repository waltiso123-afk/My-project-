import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Network, AlertTriangle, Layers, Info, Check, X, HelpCircle } from "lucide-react";
import { getHouseGroups, getPairs, rulePair, rawImageUrl } from "@/lib/api";

const RULING_STYLE = {
  same: "bg-emerald-500/20 border-emerald-500/50 text-emerald-300",
  different: "bg-slate-700/50 border-slate-600 text-slate-300",
  unsure: "bg-amber-500/20 border-amber-500/50 text-amber-300",
};

export default function HouseGroupManager() {
  const [groups, setGroups] = useState([]);
  const [pairs, setPairs] = useState([]);
  const [busy, setBusy] = useState(false);

  const load = () => {
    getHouseGroups(false).then(setGroups).catch(() => {});
    getPairs().then((d) => setPairs(d.pairs || [])).catch(() => {});
  };
  useEffect(() => { load(); }, []);

  const multi = groups.filter((g) => g.size > 1);

  const rule = async (p, ruling) => {
    setBusy(true);
    try {
      const res = await rulePair(p.image_a, p.image_b, ruling);
      if (ruling === "same" && res.merged) toast.success(`Merged → ${res.group?.group_id}. Split recomputed.`);
      else toast.success(`Ruling saved: ${ruling.toUpperCase()}`);
      load();
    } catch (e) { toast.error("Failed to save ruling"); }
    finally { setBusy(false); }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-6 py-6">
      <div className="flex items-center gap-3 mb-2">
        <Network className="w-6 h-6 text-violet-400" />
        <h2 className="font-display text-2xl font-semibold">House Group Manager</h2>
      </div>

      <div className="p-4 rounded-lg bg-blue-950/30 border border-blue-500/30 text-sm text-slate-300 flex gap-3 mb-6">
        <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
        <div>
          Groups are detected with <b>ORB local features + RANSAC homography</b> geometric verification.
          This is <b>candidate evidence, not ground truth</b> — ORB can miss the same property across large
          viewpoint/lighting/crop changes. Rule each pair below: <b>SAME</b> merges the pair and recomputes
          the anti-leakage split; <b>DIFFERENT</b>/<b>UNSURE</b> are recorded without merging. Nothing is
          auto-merged. Goal: minimise both false merges and false splits.
        </div>
      </div>

      {/* Confirmed multi-view groups */}
      <div className="mb-8">
        <h3 className="font-display text-lg font-semibold mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-400" /> Confirmed multi-view groups
          <span className="text-sm font-mono text-slate-500">({multi.length})</span>
        </h3>
        {multi.length === 0 ? (
          <div className="text-sm text-slate-500 p-4 rounded-lg border border-slate-800 bg-[#0B0F19]">
            None yet — all {groups.length} images are singleton house_groups. Rule a pair SAME to create one.
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {multi.map((g) => (
              <div key={g.group_id} data-testid={`group-${g.group_id}`} className="p-3 rounded-lg border border-slate-800 bg-[#111827]">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-sm text-violet-300">{g.group_id}</span>
                  <span className="text-xs font-mono text-slate-500">{g.split} · {g.size} imgs</span>
                </div>
                <div className="flex gap-1.5 overflow-x-auto">
                  {g.image_ids.map((id) => (
                    <img key={id} src={rawImageUrl(id)} alt={id} className="w-16 h-16 object-cover rounded border border-slate-700" />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Candidate pairs with full metrics + rulings */}
      <h3 className="font-display text-lg font-semibold mb-3 flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-amber-400" /> Candidate pairs — human review
        <span className="text-sm font-mono text-slate-500">({pairs.length})</span>
      </h3>
      <div className="grid md:grid-cols-2 gap-4">
        {pairs.map((p) => {
          const key = `${p.image_a}-${p.image_b}`;
          return (
            <div key={key} data-testid={`candidate-${key}`} className="p-4 rounded-lg border border-slate-800 bg-[#111827]">
              <div className="grid grid-cols-2 gap-2 mb-3">
                {[[p.image_a, p.file_a, p.batch_a, p.orientation_a], [p.image_b, p.file_b, p.batch_b, p.orientation_b]].map(([id, file, batch, orient]) => (
                  <div key={id} className="relative rounded overflow-hidden border border-slate-700">
                    <img src={rawImageUrl(id)} alt={id} className="w-full h-32 object-cover" />
                    <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-black/75 text-[10px] font-mono text-slate-200">
                      {id} · {file}
                    </span>
                    <span className="absolute top-1 right-1 px-1.5 py-0.5 rounded bg-black/75 text-[9px] font-mono text-slate-400">
                      {batch === "batch1" ? "B1" : "B2"} · {orient?.slice(0, 4)}
                    </span>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-4 gap-2 text-center mb-3">
                <Metric label="inliers" value={p.inliers} accent={p.strong ? "text-emerald-300" : "text-amber-300"} />
                <Metric label="good match" value={p.raw_good_matches} />
                <Metric label="ratio" value={p.inlier_ratio ?? "—"} />
                <Metric label="kp a/b" value={`${p.keypoints_a ?? "?"}/${p.keypoints_b ?? "?"}`} />
              </div>
              <div className="flex items-center gap-2">
                {p.ruling && (
                  <span className={`px-2 py-1 rounded text-[11px] font-mono border ${RULING_STYLE[p.ruling]}`}>
                    ruled: {p.ruling}
                  </span>
                )}
                <div className="ml-auto flex gap-1.5">
                  <button data-testid={`rule-same-${key}`} disabled={busy} onClick={() => rule(p, "same")}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 text-xs hover:bg-emerald-500/25">
                    <Check className="w-3.5 h-3.5" /> Same
                  </button>
                  <button data-testid={`rule-different-${key}`} disabled={busy} onClick={() => rule(p, "different")}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-slate-700/60 border border-slate-600 text-slate-300 text-xs hover:bg-slate-700">
                    <X className="w-3.5 h-3.5" /> Different
                  </button>
                  <button data-testid={`rule-unsure-${key}`} disabled={busy} onClick={() => rule(p, "unsure")}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-amber-500/15 border border-amber-500/40 text-amber-300 text-xs hover:bg-amber-500/25">
                    <HelpCircle className="w-3.5 h-3.5" /> Unsure
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Metric({ label, value, accent = "text-slate-200" }) {
  return (
    <div className="rounded bg-[#0B0F19] border border-slate-800 py-1.5">
      <div className={`font-mono text-sm ${accent}`}>{value}</div>
      <div className="text-[9px] font-mono uppercase text-slate-600">{label}</div>
    </div>
  );
}
