const STATUS = { queued: "queued", processing: "running", error: "failed" };

function summary(job) {
  if (job.status !== "done") return STATUS[job.status];
  const { no_helmet, triple_riding } = job.result.counts;
  return no_helmet + triple_riding === 0 ? "clean" : `${no_helmet} no helmet · ${triple_riding} triple`;
}

export default function History({ jobs, selected, onOpen, onDelete }) {
  return (
    <div className="card">
      <h2>History</h2>
      {jobs.length === 0 && <p className="muted small">Analysed videos will show up here.</p>}
      <ul className="history">
        {jobs.map((j) => (
          <li key={j.id} className={j.id === selected ? "active" : ""} onClick={() => onOpen(j.id)}>
            <div className="h-name">{j.filename}</div>
            <div className="h-meta">
              <span className={`dot ${j.status}`} />
              {summary(j)}
              {j.status === "processing" && ` ${Math.round(j.progress * 100)}%`}
            </div>
            {j.status !== "queued" && j.status !== "processing" && (
              <button
                className="x"
                title="Delete"
                onClick={(e) => { e.stopPropagation(); onDelete(j.id); }}
              >×</button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
