import { useEffect, useLayoutEffect, useRef, useState, useCallback } from "react";
import { toast } from "sonner";
import {
  MousePointerClick, PenLine, SquareDashed, Hand, Maximize2, Eye, EyeOff,
  Plus, Trash2, ChevronLeft, ChevronRight, Check, AlertTriangle, Keyboard,
  Loader2, Eraser, Save,
} from "lucide-react";
import {
  getImage, getAnnotation, saveAnnotation, approveAnnotation, markNeedsReview,
  propose, rawImageUrl, PLANE_COLORS,
} from "@/lib/api";

const OCCLUSION_TYPES = ["tree", "chimney", "shadow", "solar_panel", "antenna", "other"];

function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const uid = () => Math.random().toString(36).slice(2, 9);

export default function AnnotationWorkspace({
  activeId, setActiveId, orderedIds, setOrderedIds, mlStatus, onStatusChange,
}) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const imgRef = useRef(null);
  const viewRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
  const panRef = useRef(null);
  const spaceRef = useRef(false);
  const skipSaveRef = useRef(true);

  const [meta, setMeta] = useState(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [tool, setTool] = useState("point");
  const [planes, setPlanes] = useState([]);
  const [activePlane, setActivePlane] = useState(null);
  const [draft, setDraft] = useState([]);
  const [occlusions, setOcclusions] = useState([]);
  const [pendingBox, setPendingBox] = useState(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.45);
  const [showOverlay, setShowOverlay] = useState(true);
  const [notes, setNotes] = useState("");
  const [checklist, setChecklist] = useState({});
  const [status, setStatus] = useState("pending");
  const [proposing, setProposing] = useState(false);
  const [saveState, setSaveState] = useState("idle");
  const [tick, setTick] = useState(0); // force redraw
  const [showShortcuts, setShowShortcuts] = useState(false);

  const idx = orderedIds.indexOf(activeId);

  // ---------- load ----------
  useEffect(() => {
    if (!activeId) return;
    skipSaveRef.current = true;
    setImgLoaded(false);
    Promise.all([getImage(activeId), getAnnotation(activeId)]).then(([im, ann]) => {
      setMeta(im);
      if (!orderedIds.length) setOrderedIds([activeId]);
      setPlanes((ann.planes || []).map((p) => ({ ...p, id: p.id || uid() })));
      setOcclusions(ann.occlusions || []);
      setNotes(ann.notes || "");
      setChecklist(ann.checklist || {});
      setStatus(ann.status || im.status || "pending");
      setActivePlane((ann.planes && ann.planes[0]?.id) || null);
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => { imgRef.current = image; setImgLoaded(true); fitView(); };
      image.src = rawImageUrl(activeId);
    });
    // eslint-disable-next-line
  }, [activeId]);

  // ---------- autosave ----------
  useEffect(() => {
    if (skipSaveRef.current) { skipSaveRef.current = false; return; }
    if (!activeId) return;
    setSaveState("saving");
    const t = setTimeout(() => {
      saveAnnotation(activeId, { planes, occlusions, notes, checklist, status })
        .then(() => setSaveState("saved"))
        .catch(() => setSaveState("idle"));
    }, 700);
    return () => clearTimeout(t);
    // eslint-disable-next-line
  }, [planes, occlusions, notes, checklist]);

  // ---------- canvas sizing ----------
  useLayoutEffect(() => {
    const resize = () => {
      const c = canvasRef.current, w = wrapRef.current;
      if (!c || !w) return;
      c.width = w.clientWidth; c.height = w.clientHeight;
      setTick((t) => t + 1);
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  const fitView = useCallback(() => {
    const img = imgRef.current, w = wrapRef.current;
    if (!img || !w) return;
    const s = Math.min(w.clientWidth / img.width, w.clientHeight / img.height) * 0.94;
    viewRef.current = {
      scale: s,
      offsetX: (w.clientWidth - img.width * s) / 2,
      offsetY: (w.clientHeight - img.height * s) / 2,
    };
    setTick((t) => t + 1);
  }, []);

  // ---------- draw ----------
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, c.width, c.height);
    const img = imgRef.current;
    if (!img || !imgLoaded) return;
    const { scale, offsetX, offsetY } = viewRef.current;
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);

    if (showOverlay) {
      planes.forEach((p) => {
        (p.polygons || []).forEach((poly) => {
          if (poly.length < 2) return;
          ctx.beginPath();
          ctx.moveTo(poly[0][0], poly[0][1]);
          poly.forEach((pt) => ctx.lineTo(pt[0], pt[1]));
          ctx.closePath();
          ctx.fillStyle = hexToRgba(p.color, overlayOpacity);
          ctx.fill();
          ctx.lineWidth = (p.id === activePlane ? 2.5 : 1.5) / scale;
          ctx.strokeStyle = p.color;
          ctx.stroke();
        });
      });
    }

    // draft polygon
    if (draft.length) {
      ctx.beginPath();
      ctx.moveTo(draft[0][0], draft[0][1]);
      draft.forEach((pt) => ctx.lineTo(pt[0], pt[1]));
      ctx.strokeStyle = "#06B6D4";
      ctx.lineWidth = 2 / scale;
      ctx.stroke();
      draft.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 4 / scale, 0, Math.PI * 2);
        ctx.fillStyle = "#06B6D4";
        ctx.fill();
      });
    }

    // occlusions
    occlusions.forEach((o) => {
      const [x, y, w, h] = o.bbox;
      ctx.strokeStyle = "#EAB308";
      ctx.setLineDash([6 / scale, 4 / scale]);
      ctx.lineWidth = 2 / scale;
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    });
    if (pendingBox) {
      const { x, y, w, h } = pendingBox;
      ctx.strokeStyle = "#EC4899";
      ctx.lineWidth = 2 / scale;
      ctx.strokeRect(x, y, w, h);
    }
    ctx.restore();
    // eslint-disable-next-line
  }, [tick, planes, draft, occlusions, pendingBox, overlayOpacity, showOverlay, activePlane, imgLoaded]);

  // ---------- coords ----------
  const toImg = (e) => {
    const c = canvasRef.current, r = c.getBoundingClientRect();
    const { scale, offsetX, offsetY } = viewRef.current;
    return [
      clamp((e.clientX - r.left - offsetX) / scale, 0, imgRef.current?.width || 0),
      clamp((e.clientY - r.top - offsetY) / scale, 0, imgRef.current?.height || 0),
    ];
  };

  // ---------- plane helpers ----------
  const ensureActivePlane = () => {
    let ap = planes.find((p) => p.id === activePlane);
    if (ap) return ap;
    const np = newPlaneObj();
    setPlanes((prev) => [...prev, np]);
    setActivePlane(np.id);
    return np;
  };
  const newPlaneObj = () => ({
    id: uid(), name: `Plane ${planes.length + 1}`,
    color: PLANE_COLORS[planes.length % PLANE_COLORS.length],
    is_parapet: false, polygons: [],
  });
  const addPlane = () => {
    const np = newPlaneObj();
    setPlanes((p) => [...p, np]);
    setActivePlane(np.id);
  };
  const updatePlane = (id, patch) =>
    setPlanes((prev) => prev.map((p) => (p.id === id ? { ...p, ...patch } : p)));
  const deletePlane = (id) => {
    setPlanes((prev) => prev.filter((p) => p.id !== id));
    if (activePlane === id) setActivePlane(null);
  };
  const setPlanePolys = (id, polys) =>
    setPlanes((prev) => prev.map((p) => (p.id === id ? { ...p, polygons: polys } : p)));

  // ---------- point-assist propose ----------
  const runPropose = async (pt) => {
    setProposing(true);
    const ap = ensureActivePlane();
    try {
      const res = await propose({ dataset_id: activeId, positive_points: [[Math.round(pt[0]), Math.round(pt[1])]] });
      if (!res.polygons?.length) { toast.warning("No region found — try another point or adjust."); return; }
      setPlanePolys(ap.id, [...(ap.polygons || []), ...res.polygons]);
      toast.success(res.backend === "sam2_cuda" ? "SAM2 proposal added" : "CPU-assist proposal added (not SAM2)");
    } catch (e) {
      toast.error("Proposal failed");
    } finally {
      setProposing(false);
    }
  };

  // ---------- mouse ----------
  const onMouseDown = (e) => {
    if (spaceRef.current || tool === "pan") {
      panRef.current = { x: e.clientX, y: e.clientY, ...viewRef.current };
      return;
    }
    const p = toImg(e);
    if (tool === "point") runPropose(p);
    else if (tool === "polygon") setDraft((d) => [...d, p]);
    else if (tool === "occlusion") panRef.current = { box: true, start: p };
  };
  const onMouseMove = (e) => {
    if (panRef.current && panRef.current.box) {
      const p = toImg(e);
      const s = panRef.current.start;
      setPendingBox({ x: Math.min(s[0], p[0]), y: Math.min(s[1], p[1]), w: Math.abs(p[0] - s[0]), h: Math.abs(p[1] - s[1]) });
      return;
    }
    if (panRef.current) {
      const dx = e.clientX - panRef.current.x, dy = e.clientY - panRef.current.y;
      viewRef.current = { ...viewRef.current, offsetX: panRef.current.offsetX + dx, offsetY: panRef.current.offsetY + dy };
      setTick((t) => t + 1);
    }
  };
  const onMouseUp = () => {
    if (panRef.current && panRef.current.box) {
      if (pendingBox && pendingBox.w > 4 && pendingBox.h > 4) {
        // keep pendingBox for form
      } else setPendingBox(null);
    }
    panRef.current = null;
  };
  const onWheel = (e) => {
    e.preventDefault();
    const c = canvasRef.current, r = c.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    const { scale, offsetX, offsetY } = viewRef.current;
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    const ns = clamp(scale * factor, 0.05, 12);
    viewRef.current = {
      scale: ns,
      offsetX: cx - ((cx - offsetX) / scale) * ns,
      offsetY: cy - ((cy - offsetY) / scale) * ns,
    };
    setTick((t) => t + 1);
  };
  const commitDraft = () => {
    if (draft.length >= 3) {
      const ap = ensureActivePlane();
      setPlanePolys(ap.id, [...(ap.polygons || []), draft]);
    }
    setDraft([]);
  };
  const addOcclusion = (type, hides) => {
    if (!pendingBox) return;
    setOcclusions((o) => [...o, { type, hides: hides || "", bbox: [Math.round(pendingBox.x), Math.round(pendingBox.y), Math.round(pendingBox.w), Math.round(pendingBox.h)] }]);
    setPendingBox(null);
  };

  // ---------- nav + actions ----------
  const go = (delta) => {
    const ni = idx + delta;
    if (ni >= 0 && ni < orderedIds.length) setActiveId(orderedIds[ni]);
  };
  const doApprove = async () => {
    if (!planes.length) { toast.error("Add at least one roof plane before approving."); return; }
    try {
      await approveAnnotation(activeId, { planes, occlusions, notes, checklist });
      setStatus("approved");
      toast.success("Approved · merged mask + per-plane masks + metadata generated");
      onStatusChange?.();
    } catch (e) { toast.error("Approve failed"); }
  };
  const doNeedsReview = async () => {
    const reason = notes || "flagged ambiguous";
    await markNeedsReview(activeId, reason);
    setStatus("needs_review");
    toast.info("Marked needs_review");
    onStatusChange?.();
  };

  // ---------- keyboard ----------
  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (e.code === "Space") { spaceRef.current = true; return; }
      const k = e.key.toLowerCase();
      if (k === "o") setShowOverlay((s) => !s);
      else if (k === "p") setTool("point");
      else if (k === "g") setTool("polygon");
      else if (k === "x") setTool("occlusion");
      else if (k === "h") setTool("pan");
      else if (k === "f") fitView();
      else if (k === "a" || e.key === "ArrowLeft") go(-1);
      else if (k === "d" || e.key === "ArrowRight") go(1);
      else if (k === "n") doNeedsReview();
      else if (k === "s" || e.key === "Enter") { if (tool === "polygon" && draft.length) commitDraft(); else doApprove(); }
      else if (e.key === "Escape") setDraft([]);
    };
    const onUp = (e) => { if (e.code === "Space") spaceRef.current = false; };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onUp);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onUp); };
    // eslint-disable-next-line
  }, [idx, orderedIds, planes, draft, tool, activeId, occlusions, notes, checklist]);

  if (!activeId) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] text-slate-500">
        <PenLine className="w-10 h-10 mb-3 opacity-40" />
        <p>Select an image from the Dataset Gallery to start annotating.</p>
      </div>
    );
  }

  const TOOLS = [
    { id: "point", icon: MousePointerClick, label: "Point assist (P)" },
    { id: "polygon", icon: PenLine, label: "Polygon (G)" },
    { id: "occlusion", icon: SquareDashed, label: "Occlusion box (X)" },
    { id: "pan", icon: Hand, label: "Pan (H / Space)" },
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] overflow-hidden bg-[#030712]">
      {/* top action bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800 bg-[#0B0F19]">
        <button data-testid="prev-image" onClick={() => go(-1)} disabled={idx <= 0}
          className="p-2 rounded-md hover:bg-slate-800 disabled:opacity-30"><ChevronLeft className="w-4 h-4" /></button>
        <span className="font-mono text-sm text-slate-300" data-testid="current-image">
          {activeId} <span className="text-slate-600">· {idx + 1}/{orderedIds.length || 1}</span>
        </span>
        <button data-testid="next-image" onClick={() => go(1)} disabled={idx >= orderedIds.length - 1}
          className="p-2 rounded-md hover:bg-slate-800 disabled:opacity-30"><ChevronRight className="w-4 h-4" /></button>

        <StatusPill status={status} />
        <span className="text-xs font-mono text-slate-500 flex items-center gap-1" data-testid="save-indicator">
          {saveState === "saving" ? <><Loader2 className="w-3 h-3 animate-spin" /> saving…</>
            : saveState === "saved" ? <><Save className="w-3 h-3 text-emerald-400" /> saved</> : ""}
        </span>

        <div className="ml-auto flex items-center gap-2">
          <button data-testid="shortcuts-btn" onClick={() => setShowShortcuts(true)}
            className="p-2 rounded-md hover:bg-slate-800 text-slate-400"><Keyboard className="w-4 h-4" /></button>
          <button data-testid="needs-review-btn" onClick={doNeedsReview}
            className="flex items-center gap-1.5 px-3 py-2 rounded-md bg-rose-500/15 border border-rose-500/30 text-rose-300 text-sm hover:bg-rose-500/25">
            <AlertTriangle className="w-4 h-4" /> Needs Review
          </button>
          <button data-testid="approve-btn" onClick={doApprove}
            className="flex items-center gap-1.5 px-4 py-2 rounded-md bg-emerald-500 hover:bg-emerald-600 text-white text-sm font-medium">
            <Check className="w-4 h-4" /> Approve & Generate
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* tools */}
        <div className="w-14 border-r border-slate-800 bg-[#0B0F19] flex flex-col items-center gap-1.5 py-3">
          {TOOLS.map((t) => (
            <button key={t.id} data-testid={`tool-${t.id}`} title={t.label} onClick={() => setTool(t.id)}
              className={`p-2.5 rounded-lg transition-colors ${tool === t.id ? "bg-blue-500/20 text-blue-300 border border-blue-500/40" : "text-slate-400 hover:bg-slate-800"}`}>
              <t.icon className="w-5 h-5" />
            </button>
          ))}
          <div className="w-8 h-px bg-slate-800 my-1" />
          <button data-testid="fit-btn" title="Fit (F)" onClick={fitView} className="p-2.5 rounded-lg text-slate-400 hover:bg-slate-800"><Maximize2 className="w-5 h-5" /></button>
          <button data-testid="overlay-toggle" title="Toggle overlay (O)" onClick={() => setShowOverlay((s) => !s)}
            className={`p-2.5 rounded-lg ${showOverlay ? "text-cyan-300" : "text-slate-500"} hover:bg-slate-800`}>
            {showOverlay ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
          </button>
        </div>

        {/* canvas */}
        <div ref={wrapRef} className="flex-1 relative overflow-hidden select-none"
          style={{ cursor: tool === "pan" ? "grab" : "crosshair" }}>
          <canvas ref={canvasRef} data-testid="annotation-canvas"
            onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp} onWheel={onWheel} onDoubleClick={commitDraft} />
          {proposing && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/70 border border-slate-700 text-xs text-cyan-300">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> generating proposal…
            </div>
          )}
          {tool === "polygon" && draft.length > 0 && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full bg-black/70 border border-slate-700 text-xs text-slate-300">
              {draft.length} pts · double-click / Enter to close · Esc to cancel
            </div>
          )}
          <div className="absolute bottom-3 right-3 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/70 border border-slate-700 text-xs">
            <span className="text-slate-400">overlay</span>
            <input data-testid="opacity-slider" type="range" min="0" max="100" value={overlayOpacity * 100}
              onChange={(e) => setOverlayOpacity(Number(e.target.value) / 100)} className="accent-cyan-400 w-24" />
          </div>
        </div>

        {/* inspector */}
        <div className="w-80 border-l border-slate-800 bg-[#0B0F19] flex flex-col overflow-y-auto">
          <div className="p-3 border-b border-slate-800 flex items-center justify-between">
            <span className="font-display font-semibold text-sm flex items-center gap-2">Roof Planes
              <span className="text-xs font-mono text-slate-500">{planes.length}</span></span>
            <button data-testid="add-plane" onClick={addPlane}
              className="flex items-center gap-1 px-2 py-1 rounded-md bg-blue-500/15 border border-blue-500/30 text-blue-300 text-xs hover:bg-blue-500/25">
              <Plus className="w-3.5 h-3.5" /> Plane
            </button>
          </div>

          <div className="p-2 space-y-1.5">
            {planes.length === 0 && <p className="text-xs text-slate-500 p-3">No planes yet. Use Point-assist (P) or Polygon (G) to add a roof plane.</p>}
            {planes.map((p, i) => (
              <div key={p.id} data-testid={`plane-row-${i}`}
                onClick={() => setActivePlane(p.id)}
                className={`p-2.5 rounded-lg border cursor-pointer ${p.id === activePlane ? "border-blue-500/50 bg-blue-500/10" : "border-slate-800 bg-[#111827]"}`}>
                <div className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 rounded-sm shrink-0" style={{ background: p.color }} />
                  <input value={p.name} onClick={(e) => e.stopPropagation()}
                    onChange={(e) => updatePlane(p.id, { name: e.target.value })}
                    className="bg-transparent text-sm flex-1 min-w-0 focus:outline-none" />
                  <span className="text-[10px] font-mono text-slate-500">{(p.polygons || []).length}p</span>
                  <button data-testid={`delete-plane-${i}`} onClick={(e) => { e.stopPropagation(); deletePlane(p.id); }}
                    className="text-slate-500 hover:text-rose-400"><Trash2 className="w-3.5 h-3.5" /></button>
                </div>
                <div className="flex items-center gap-3 mt-2 pl-5.5">
                  <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer" onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" data-testid={`parapet-${i}`} checked={!!p.is_parapet}
                      onChange={(e) => updatePlane(p.id, { is_parapet: e.target.checked })} className="accent-cyan-500" />
                    -parapet
                  </label>
                  {(p.polygons || []).length > 0 && (
                    <button onClick={(e) => { e.stopPropagation(); setPlanePolys(p.id, []); }}
                      className="flex items-center gap-1 text-xs text-slate-500 hover:text-amber-400">
                      <Eraser className="w-3 h-3" /> clear
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* occlusion pending form */}
          {pendingBox && (
            <div className="p-3 border-t border-slate-800" data-testid="occlusion-form">
              <div className="text-xs font-mono uppercase text-slate-500 mb-2">New occlusion</div>
              <OcclusionForm onAdd={addOcclusion} onCancel={() => setPendingBox(null)} />
            </div>
          )}

          {occlusions.length > 0 && (
            <div className="p-3 border-t border-slate-800">
              <div className="text-xs font-mono uppercase text-slate-500 mb-2">Occlusions ({occlusions.length})</div>
              {occlusions.map((o, i) => (
                <div key={i} className="flex items-center justify-between text-xs py-1">
                  <span className="text-amber-300">{o.type}</span>
                  <span className="text-slate-500 font-mono">{o.hides}</span>
                  <button onClick={() => setOcclusions((arr) => arr.filter((_, j) => j !== i))}
                    className="text-slate-500 hover:text-rose-400"><Trash2 className="w-3 h-3" /></button>
                </div>
              ))}
            </div>
          )}

          {/* checklist */}
          <div className="p-3 border-t border-slate-800">
            <div className="text-xs font-mono uppercase text-slate-500 mb-2">QA checklist (spec)</div>
            {[
              ["eave_verified", "Eave = gutter lip / drip edge (not fascia/soffit/shingle)"],
              ["gable_excluded", "Gable wall excluded"],
              ["planes_separated", "Planes properly separated"],
              ["occlusion_under_15", "Occlusion < 15% rule applied"],
            ].map(([k, label]) => (
              <label key={k} className="flex items-start gap-2 text-xs text-slate-400 py-1 cursor-pointer">
                <input type="checkbox" data-testid={`check-${k}`} checked={!!checklist[k]}
                  onChange={(e) => setChecklist((c) => ({ ...c, [k]: e.target.checked }))}
                  className="accent-emerald-500 mt-0.5" />
                {label}
              </label>
            ))}
          </div>

          <div className="p-3 border-t border-slate-800">
            <div className="text-xs font-mono uppercase text-slate-500 mb-2">Notes</div>
            <textarea data-testid="notes-field" value={notes} onChange={(e) => setNotes(e.target.value)}
              rows={3} placeholder="Ambiguities, decisions, edge cases…"
              className="w-full bg-[#111827] border border-slate-800 rounded-lg p-2 text-sm focus:outline-none focus:border-blue-500/60 resize-none" />
          </div>
        </div>
      </div>

      {showShortcuts && <ShortcutsModal onClose={() => setShowShortcuts(false)} mlStatus={mlStatus} />}
    </div>
  );
}

function StatusPill({ status }) {
  const map = {
    pending: "bg-amber-950/60 border-amber-500/40 text-amber-300",
    approved: "bg-emerald-950/60 border-emerald-500/40 text-emerald-300",
    needs_review: "bg-rose-950/60 border-rose-500/40 text-rose-300",
  };
  return (
    <span data-testid="status-pill" className={`ml-2 px-2 py-1 rounded-md text-[11px] font-mono border ${map[status] || map.pending}`}>
      {status}
    </span>
  );
}

function OcclusionForm({ onAdd, onCancel }) {
  const [type, setType] = useState(OCCLUSION_TYPES[0]);
  const [hides, setHides] = useState("");
  return (
    <div className="space-y-2">
      <select value={type} onChange={(e) => setType(e.target.value)} data-testid="occlusion-type"
        className="w-full bg-[#111827] border border-slate-800 rounded-md p-1.5 text-sm">
        {OCCLUSION_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
      <input value={hides} onChange={(e) => setHides(e.target.value)} placeholder="hides (e.g. plane-02 edge)"
        data-testid="occlusion-hides" className="w-full bg-[#111827] border border-slate-800 rounded-md p-1.5 text-sm" />
      <div className="flex gap-2">
        <button data-testid="occlusion-add" onClick={() => onAdd(type, hides)}
          className="flex-1 px-2 py-1.5 rounded-md bg-amber-500/20 border border-amber-500/40 text-amber-300 text-xs">Add</button>
        <button onClick={onCancel} className="px-3 py-1.5 rounded-md bg-slate-800 text-slate-400 text-xs">Cancel</button>
      </div>
    </div>
  );
}

function ShortcutsModal({ onClose, mlStatus }) {
  const rows = [
    ["P", "Point-assist tool"], ["G", "Polygon tool"], ["X", "Occlusion box"], ["H / Space", "Pan"],
    ["O", "Toggle overlay"], ["F", "Fit to screen"], ["A / ←", "Previous image"], ["D / →", "Next image"],
    ["Enter / S", "Close polygon / Approve"], ["N", "Mark needs review"], ["Esc", "Cancel polygon draft"],
  ];
  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-[#111827] border border-slate-700 rounded-xl p-6 max-w-md w-full" onClick={(e) => e.stopPropagation()}>
        <h3 className="font-display text-lg font-semibold mb-1">Keyboard shortcuts</h3>
        <p className="text-xs text-slate-500 mb-4">{mlStatus?.message}</p>
        <div className="grid grid-cols-1 gap-1.5">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between text-sm">
              <span className="text-slate-400">{v}</span>
              <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono text-xs text-slate-200">{k}</kbd>
            </div>
          ))}
        </div>
        <button onClick={onClose} className="mt-5 w-full py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm">Close</button>
      </div>
    </div>
  );
}
