import { LayoutDashboard, Images, PenTool, Network, ShieldCheck, Cpu, Zap } from "lucide-react";

const TABS = [
  { id: "dashboard", label: "Checkpoint", icon: LayoutDashboard },
  { id: "gallery", label: "Dataset Gallery", icon: Images },
  { id: "workspace", label: "Annotation Studio", icon: PenTool },
  { id: "housegroups", label: "House Groups", icon: Network },
  { id: "leakage", label: "Leakage Report", icon: ShieldCheck },
];

export default function HeaderNav({ tab, setTab, summary, mlStatus }) {
  const gpu = mlStatus?.active_backend === "sam2_cuda";
  return (
    <header
      className="sticky top-0 z-40 border-b border-slate-800 bg-[#0B0F19]/95 backdrop-blur-xl"
      data-testid="header-nav"
    >
      <div className="flex items-center gap-4 px-4 md:px-6 h-16">
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <PenTool className="w-5 h-5 text-white" />
          </div>
          <div className="leading-tight">
            <div className="font-display font-bold text-[15px] tracking-tight">Roofline</div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500">
              Phase 1 · Foundation
            </div>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-1 ml-2">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                data-testid={`nav-${t.id}`}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-md text-sm font-medium transition-colors ${
                  active
                    ? "bg-blue-500/15 text-blue-300 border border-blue-500/30"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 border border-transparent"
                }`}
              >
                <Icon className="w-4 h-4" />
                {t.label}
              </button>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          {summary && (
            <div className="hidden lg:flex items-center gap-3 text-xs font-mono text-slate-400">
              <span data-testid="ticker-total">{summary.total_images} imgs</span>
              <span className="text-slate-700">|</span>
              <span className="text-emerald-400" data-testid="ticker-approved">
                {summary.status_counts?.approved ?? 0} approved
              </span>
            </div>
          )}
          <div
            data-testid="gpu-badge"
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono border ${
              gpu
                ? "bg-emerald-950/60 border-emerald-500/40 text-emerald-300"
                : mlStatus?.active_backend === "sam2_cpu"
                ? "bg-sky-950/60 border-sky-500/40 text-sky-300"
                : "bg-amber-950/50 border-amber-500/30 text-amber-300"
            }`}
            title={mlStatus?.message || ""}
          >
            {gpu ? <Zap className="w-3.5 h-3.5" /> : <Cpu className="w-3.5 h-3.5" />}
            {mlStatus?.active_backend === "sam2_cuda" ? "SAM2 · GPU"
              : mlStatus?.active_backend === "sam2_cpu" ? "SAM2 · CPU"
              : "SAM2 n/a"}
          </div>
        </div>
      </div>

      <nav className="md:hidden flex items-center gap-1 px-3 pb-2 overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`whitespace-nowrap px-3 py-1.5 rounded-md text-xs font-medium ${
              tab === t.id ? "bg-blue-500/15 text-blue-300" : "text-slate-400"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
