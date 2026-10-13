import { useRef, useState } from "react";
import { uploadVideo } from "../api.js";

export default function Upload({ onUploaded }) {
  const input = useRef(null);
  const [file, setFile] = useState(null);
  const [useYoloe, setUseYoloe] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  const pick = (f) => { if (f) { setFile(f); setError(""); } };

  async function submit() {
    setSending(true);
    setError("");
    try {
      onUploaded(await uploadVideo(file, useYoloe));
    } catch (e) {
      setError(e.message);
      setSending(false);
    }
  }

  return (
    <div className="card">
      <h1>Analyse a traffic video</h1>
      <p className="muted">
        Upload a clip. The server finds every motorcycle, counts its riders, checks their helmets and
        reports <b>no-helmet</b> and <b>triple-riding</b> violations with photo evidence.
      </p>

      <div
        className={`drop ${dragging ? "over" : ""}`}
        onClick={() => input.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]); }}
      >
        <input ref={input} type="file" accept="video/*" hidden onChange={(e) => pick(e.target.files[0])} />
        {file ? (
          <>
            <div className="file-name">{file.name}</div>
            <div className="muted">{(file.size / 1e6).toFixed(1)} MB · click to choose another</div>
          </>
        ) : (
          <>
            <div className="drop-icon">⬆</div>
            <div>Drop a video here, or click to browse</div>
            <div className="muted">mp4, mov, avi, mkv or webm</div>
          </>
        )}
      </div>

      <label className="check">
        <input type="checkbox" checked={useYoloe} onChange={(e) => setUseYoloe(e.target.checked)} />
        <span>
          Second helmet check with YOLOE <span className="muted">(slower, fewer false alarms)</span>
        </span>
      </label>

      {error && <div className="error">{error}</div>}
      <button className="btn" disabled={!file || sending} onClick={submit}>
        {sending ? "Uploading..." : "Analyse video"}
      </button>
    </div>
  );
}
