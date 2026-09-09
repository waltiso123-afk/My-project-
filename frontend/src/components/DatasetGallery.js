import { useEffect, useState } from "react";
import { Search, PenTool, CircleDot } from "lucide-react";
import { listDataset, thumbUrl } from "@/lib/api";

const STATUS = {
  pending: { dot: "bg-amber-400", text: "text-amber-300", bg: "bg-amber-950/60 border-amber-500/40" },
  approved: { dot: "bg-emerald-400", text: "text-emerald-300", bg: "bg-emerald-950/60 border-emerald-500/40" },
  needs_review: { dot: "bg-rose-400", text: "text-rose-300", bg: "bg-rose-950/60 border-rose-500/40" },
};

export default function DatasetGallery({ onAnnotate }) {
  const [rows, setRows] = useState([]);
  const [status, setStatus] = useState("");
  const [batch, setBatch] = useState("");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    listDataset({ status: status || undefined, batch: batch || undefined, search: search || undefined })
      .then(setRows)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    const t = setTimeout(load, 200);
    return () => clearTimeout(t);
    // eslint-disable-next-line
  }, [status, batch, search]);

  const ids = rows.map((r) => r.dataset_id);

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-6 py-6">
      <div className="flex flex-wrap items-center gap-3 mb-5">
        <h2 className="font-display text-2xl font-semibold">Dataset Gallery</h2>
        <span className="text-sm text-slate-500 font-mono">{rows.length} images</span>
        <div className="ml-auto flex flex-wrap gap-2">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              data-testid="gallery-search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search id / file / group"
              className="pl-9 pr-3 py-2 rounded-lg bg-[#111827] border border-slate-800 text-sm w-56 focus:outline-none focus:border-blue-500/60"
            />
          </div>
          <select data-testid="gallery-status" value={status} onChange={(e) => setStatus(e.target.value)}
            className="px-3 py-2 rounded-lg bg-[#111827] border border-slate-800 text-sm focus:outline-none focus:border-blue-500/60">
            <option value="">All statuses</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="needs_review">Needs review</option>
          </select>
          <select data-testid="gallery-batch" value={batch} onChange={(e) => setBatch(e.target.value)}
            className="px-3 py-2 rounded-lg bg-[#111827] border border-slate-800 text-sm focus:outline-none focus:border-blue-500/60">
            <option value="">All batches</option>
            <option value="batch1">Batch 1</option>
            <option value="batch2">Batch 2</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="text-slate-500 text-sm py-20 text-center">Loading images…</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {rows.map((r) => {
            const st = STATUS[r.status] || STATUS.pending;
            return (
              <div
                key={r.dataset_id}
                data-testid={`gallery-card-${r.dataset_id}`}
                className="group relative rounded-lg overflow-hidden border border-slate-800 bg-[#0B0F19] hover:border-blue-500/50 transition-colors"
              >
                <div className="aspect-[4/3] bg-slate-900 overflow-hidden">
                  <img
                    src={thumbUrl(r.dataset_id)}
                    alt={r.dataset_id}
                    loading="lazy"
                    decoding="async"
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                </div>
                <div className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-mono border bg-black/60 backdrop-blur-sm border-slate-700">
                  <CircleDot className="w-3 h-3 text-slate-400" />
                  {r.dataset_id}
                </div>
                <div className={`absolute top-2 right-2 flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-mono border ${st.bg} ${st.text}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${st.dot}`} />
                  {r.status}
                </div>
                <div className="p-2.5 flex items-center justify-between">
                  <div className="text-[11px] font-mono text-slate-500">
                    <div>{r.batch === "batch1" ? "B1" : "B2"} · {r.orientation?.slice(0, 4)}</div>
                    <div className="text-slate-600">{r.house_group_id} · {r.split}</div>
                  </div>
                  <button
                    data-testid={`annotate-${r.dataset_id}`}
                    onClick={() => onAnnotate(r.dataset_id, ids)}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-blue-500/15 border border-blue-500/30 text-blue-300 text-xs hover:bg-blue-500/25 transition-colors"
                  >
                    <PenTool className="w-3.5 h-3.5" />
                    Annotate
                  </button>
                </div>
                {r.plane_count > 0 && (
                  <div className="absolute bottom-14 right-2 px-1.5 py-0.5 rounded bg-black/70 text-[10px] font-mono text-cyan-300 border border-cyan-500/30">
                    {r.plane_count} planes
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
