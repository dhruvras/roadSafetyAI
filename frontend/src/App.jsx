import { useCallback, useEffect, useState } from "react";
import { deleteJob, getJob, listJobs } from "./api.js";
import History from "./components/History.jsx";
import Results from "./components/Results.jsx";
import Upload from "./components/Upload.jsx";

const busy = (job) => job && (job.status === "queued" || job.status === "processing");
const idFromHash = () => window.location.hash.replace("#/job/", "") || null;

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selected, setSelected] = useState(idFromHash());

  const refresh = useCallback(() => listJobs().then(setJobs).catch(() => {}), []);
  useEffect(() => { refresh(); }, [refresh]);

  // the address bar (#/job/<id>) is the source of truth, so a refresh or a shared link opens the same result
  useEffect(() => {
    const onHash = () => setSelected(idFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const open = (id) => { window.location.hash = id ? `/job/${id}` : ""; };

  // while any job is running, ask the server for news every second
  const running = jobs.some(busy);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(async () => {
      const live = jobs.filter(busy);
      const fresh = await Promise.all(live.map((j) => getJob(j.id).catch(() => j)));
      setJobs((all) => all.map((j) => fresh.find((f) => f.id === j.id) || j));
    }, 1000);
    return () => clearInterval(timer);
  }, [running, jobs]);

  const job = jobs.find((j) => j.id === selected);

  async function remove(id) {
    await deleteJob(id);
    if (selected === id) open(null);
    refresh();
  }

  return (
    <div className="shell">
      <header className="top">
        <a className="brand" href="#/" onClick={() => open(null)}>
          <span className="logo">🏍️</span>
          <span>RoadWatch</span>
        </a>
        <span className="tagline">Helmet and triple-riding detection from traffic video</span>
      </header>

      <main className="layout">
        <section className="main">
          {!job && (
            <Upload onUploaded={(j) => { setJobs((all) => [j, ...all]); open(j.id); }} />
          )}
          {job && <Results job={job} onBack={() => open(null)} />}
        </section>
        <aside className="side">
          <History jobs={jobs} selected={selected} onOpen={open} onDelete={remove} />
        </aside>
      </main>
    </div>
  );
}
