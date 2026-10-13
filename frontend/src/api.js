// Thin wrapper over the FastAPI endpoints (see backend/main.py).
async function request(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.status === 204 ? null : res.json();
}

export const listJobs = () => request("/api/jobs");
export const getJob = (id) => request(`/api/jobs/${id}`);
export const deleteJob = (id) => request(`/api/jobs/${id}`, { method: "DELETE" });

export function uploadVideo(file, useYoloe) {
  const form = new FormData();
  form.append("file", file);
  form.append("use_yoloe", useYoloe);
  return request("/api/jobs", { method: "POST", body: form });
}
