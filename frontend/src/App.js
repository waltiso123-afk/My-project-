import { useEffect, useState } from "react";
import "@/App.css";
import { Toaster } from "sonner";
import HeaderNav from "@/components/HeaderNav";
import DashboardCheckpoint from "@/components/DashboardCheckpoint";
import DatasetGallery from "@/components/DatasetGallery";
import AnnotationWorkspace from "@/components/AnnotationWorkspace";
import HouseGroupManager from "@/components/HouseGroupManager";
import LeakageReport from "@/components/LeakageReport";
import { getSummary, getMlStatus } from "@/lib/api";

function App() {
  const [tab, setTab] = useState("dashboard");
  const [summary, setSummary] = useState(null);
  const [mlStatus, setMlStatus] = useState(null);
  const [activeId, setActiveId] = useState(null);
  const [orderedIds, setOrderedIds] = useState([]);

  const refreshSummary = () => getSummary().then(setSummary).catch(() => {});

  useEffect(() => {
    refreshSummary();
    getMlStatus().then(setMlStatus).catch(() => {});
  }, []);

  const openAnnotator = (id, ids) => {
    setActiveId(id);
    if (ids) setOrderedIds(ids);
    setTab("workspace");
  };

  return (
    <div className="dark min-h-screen bg-[#090D16] text-slate-100">
      <Toaster theme="dark" position="bottom-right" richColors />
      <HeaderNav tab={tab} setTab={setTab} summary={summary} mlStatus={mlStatus} />
      <main className="pt-2">
        {tab === "dashboard" && (
          <DashboardCheckpoint summary={summary} mlStatus={mlStatus} onGoto={setTab} />
        )}
        {tab === "gallery" && (
          <DatasetGallery onAnnotate={openAnnotator} />
        )}
        {tab === "workspace" && (
          <AnnotationWorkspace
            activeId={activeId}
            setActiveId={setActiveId}
            orderedIds={orderedIds}
            setOrderedIds={setOrderedIds}
            mlStatus={mlStatus}
            onStatusChange={refreshSummary}
          />
        )}
        {tab === "housegroups" && <HouseGroupManager />}
        {tab === "leakage" && <LeakageReport />}
      </main>
    </div>
  );
}

export default App;
