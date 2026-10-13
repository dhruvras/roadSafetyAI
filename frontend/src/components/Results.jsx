import { useRef } from "react";

const LABEL = { no_helmet: "No helmet", triple_riding: "Triple riding" };

function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone || ""}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export default function Results({ job, onBack }) {
  const video = useRef(null);

  // clicking a violation jumps the video to a second before it happened
  function seek(t) {
    if (!video.current) return;
    video.current.currentTime = Math.max(0, t - 1);
    video.current.play();
    video.current.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const head = (
    <div className="results-head">
      <button className="link" onClick={onBack}>← New analysis</button>
      <h1>{job.filename}</h1>
    </div>
  );

  if (job.status === "error") {
    return (
      <div className="card">
        {head}
        <div className="error">Analysis failed: {job.error}</div>
      </div>
    );
  }

  if (job.status !== "done") {
    const pct = Math.round(job.progress * 100);
    return (
      <div className="card">
        {head}
        <p className="muted">
          {job.status === "queued" ? "Waiting for the model..." : "Analysing frame by frame. Keep this tab open."}
        </p>
        <div className="bar"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
        <div className="muted right">{job.status === "queued" ? "queued" : `${pct}%`}</div>
      </div>
    );
  }

  const { result } = job;
  const total = result.violations.length;
  return (
    <div className="stack">
      <div className="card">{head}</div>

      <div className="stats">
        <Stat label="No helmet" value={result.counts.no_helmet} tone={result.counts.no_helmet ? "bad" : ""} />
        <Stat label="Triple riding" value={result.counts.triple_riding} tone={result.counts.triple_riding ? "bad" : ""} />
        <Stat label="Bikes checked" value={result.bikes_checked} />
      </div>

      <div className="grid2">
        <div className="card">
          <h2>Annotated video</h2>
          <video ref={video} src={result.video_url} controls playsInline />
          <p className="muted small">
            Green box = helmet, red box = bare head. A red bike label means it was flagged; grey means parked or not
            a motorcycle, so not judged.
          </p>
        </div>

        <div className="card">
          <h2>Violations <span className="muted">({total})</span></h2>
          {total === 0 ? (
            <div className="ok">No violations found in this video.</div>
          ) : (
            <table>
              <thead>
                <tr><th>Time</th><th>Bike</th><th>Violation</th><th>Conf.</th></tr>
              </thead>
              <tbody>
                {result.violations.map((v) => (
                  <tr key={`${v.bike}-${v.type}`} onClick={() => seek(v.time_s)}>
                    <td>{v.time}</td>
                    <td>#{v.bike}</td>
                    <td><span className={`badge ${v.type}`}>{LABEL[v.type]}</span></td>
                    <td>{v.confidence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {total > 0 && <p className="muted small">Click a row to jump to it in the video.</p>}
        </div>
      </div>

      {total > 0 && (
        <div className="card">
          <h2>Evidence</h2>
          <div className="evidence">
            {result.violations.map((v) => (
              <figure key={`${v.bike}-${v.type}`} onClick={() => seek(v.time_s)}>
                <img src={v.image_url} alt={`${LABEL[v.type]} bike ${v.bike}`} loading="lazy" />
                <figcaption>
                  <span className={`badge ${v.type}`}>{LABEL[v.type]}</span>
                  <div>Bike #{v.bike} · {v.time}</div>
                  <div className="muted small">{v.riders} rider(s) · confidence {v.confidence}</div>
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
