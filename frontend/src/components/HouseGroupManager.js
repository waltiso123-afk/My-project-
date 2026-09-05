import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Network, Merge, AlertTriangle, Layers, Info } from "lucide-react";
import { getHouseGroups, getCandidates, mergeGroups, rawImageUrl } from "@/lib/api";

export default function HouseGroupManager() {
  const [groups, setGroups] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [minInliers, setMinInliers] = useState(12);
  const [selected, setSelected] = useState(new Set());
  const [merging, setMerging] = useState(false);

  const load = () => {
    getHouseGroups(false).then(setGroups).catch(() => {});
    getCandidates(minInliers).then((d) => setCandidates(d.candidates || [])).catch(() => {});
  };
  useEffect(() => { load(); /* eslint-disable-next-line */ }, [minInliers]);

  const multi = groups.filter((g) => g.size > 1);
  const toggle = (id) => {
    const s = new Set(selected);
    s.has(id) ? s.delete(id) : s.add(id);
    setSelected(s);
  };

  const doMerge = async (ids) => {
    if (ids.length < 2) return;
    setMerging(true);
    try {
      const res = await mergeGroups(ids);
      toast.success(`Merged ${ids.length} images → ${res.group_id}. Split recomputed.`);
      setSelected(new Set());
      load();
    } catch (e) {
      toast.error("Merge failed");
    } finally {
      setMerging(false);
    }
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
          House groups are detected with a <b>real duplicate-property analysis</b>: ORB local
          features + RANSAC homography geometric verification (robust to viewpoint/framing changes,
          unlike perceptual hashing). Only pairs with <b>strong geometric evidence</b> (≥ 22 inliers)
          are auto-grouped; borderline pairs are surfaced below as <b>candidates for your review</b>.
          On this dataset the candidates turned out to be look-alike Florida homes (shared white stucco /
          dark window frames / tile roofs), not the same property — so nothing was auto-merged.
          Confirm a genuine multi-view match and the anti-leakage split recomputes automatically
          (holdout stays fixed). Uncertainties are never resolved silently.
        </div>
      </div>

      {/* Confirmed multi-view groups */}
      <div className="mb-8">
        <h3 className="font-display text-lg font-semibold mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-400" /> Multi-view groups
          <span className="text-sm font-mono text-slate-500">({multi.length})</span>
        </h3>
        {multi.length === 0 ? (
          <div className="text-sm text-slate-500 p-4 rounded-lg border border-slate-800 bg-[#0B0F19]">
            No multi-view groups yet — all {groups.length} images are singleton house_groups. Merge candidates below to create one.
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {multi.map((g) => (
              <div key={g.group_id} data-testid={`group-${g.group_id}`}
                className="p-3 rounded-lg border border-slate-800 bg-[#111827]">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-sm text-violet-300">{g.group_id}</span>
                  <span className="text-xs font-mono text-slate-500">{g.split} · {g.size} imgs</span>
                </div>
                <div className="flex gap-1.5 overflow-x-auto">
                  {g.image_ids.map((id) => (
                    <img key={id} src={rawImageUrl(id)} alt={id}
                      className="w-16 h-16 object-cover rounded border border-slate-700" />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Candidate pairs */}
      <div>
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <h3 className="font-display text-lg font-semibold flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" /> Review candidates
            <span className="text-sm font-mono text-slate-500">({candidates.length})</span>
          </h3>
          <div className="ml-auto flex items-center gap-2 text-sm">
            <span className="text-slate-500 font-mono text-xs">min inliers</span>
            <input type="range" min="8" max="30" value={minInliers} data-testid="maxdist-slider"
              onChange={(e) => setMinInliers(Number(e.target.value))} className="accent-blue-500" />
            <span className="font-mono text-slate-300 w-6">{minInliers}</span>
          </div>
          {selected.size >= 2 && (
            <button data-testid="merge-selected" disabled={merging}
              onClick={() => doMerge([...selected])}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-violet-500 hover:bg-violet-600 text-white text-sm font-medium">
              <Merge className="w-4 h-4" /> Merge {selected.size} selected
            </button>
          )}
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
          {candidates.map((c) => {
            const key = `${c.image_a}-${c.image_b}`;
            return (
              <div key={key} data-testid={`candidate-${key}`}
                className="p-3 rounded-lg border border-slate-800 bg-[#111827]">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-slate-400">
                    inliers <b className={c.strong ? "text-emerald-300" : "text-amber-300"}>{c.inliers}</b>
                    <span className="text-slate-600"> · good {c.good_matches}</span>
                    <span className="text-slate-600"> · {c.batch_a === "batch1" ? "B1" : "B2"}↔{c.batch_b === "batch1" ? "B1" : "B2"}</span>
                  </span>
                  <button
                    data-testid={`merge-pair-${key}`}
                    onClick={() => doMerge([c.image_a, c.image_b])}
                    disabled={merging}
                    className="text-xs px-2 py-1 rounded bg-violet-500/15 border border-violet-500/30 text-violet-300 hover:bg-violet-500/25">
                    Merge pair
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {[c.image_a, c.image_b].map((id) => (
                    <button key={id} onClick={() => toggle(id)}
                      className={`relative rounded overflow-hidden border-2 transition-colors ${
                        selected.has(id) ? "border-blue-400" : "border-transparent"}`}>
                      <img src={rawImageUrl(id)} alt={id} className="w-full h-24 object-cover" />
                      <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-black/70 text-[10px] font-mono text-slate-200">
                        {id}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
