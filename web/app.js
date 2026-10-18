const fileInput = document.querySelector("#file");
const filename = document.querySelector("#filename");
const go = document.querySelector("#go");
const progress = document.querySelector("#progress");
const fill = document.querySelector("#fill");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const player = document.querySelector("#player");
const download = document.querySelector("#download");
const table = document.querySelector("#table");
const rows = document.querySelector("#rows");
const apiInput = document.querySelector("#api");

apiInput.value = localStorage.getItem("touchduel-api") || "";
apiInput.addEventListener("change", () => {
  localStorage.setItem("touchduel-api", apiInput.value.trim());
});

function api(path) {
  const base = apiInput.value.trim().replace(/\/$/, "");
  return base + path;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  filename.textContent = file ? file.name : "Choose a video";
  go.disabled = !file;
});

go.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  errorBox.hidden = true;
  player.hidden = true;
  player.removeAttribute("src");
  download.hidden = true;
  table.hidden = true;
  rows.innerHTML = "";
  progress.hidden = false;
  fill.style.width = "0%";
  status.textContent = "0%";
  go.disabled = true;

  const body = new FormData();
  body.append("file", file);

  let created;
  try {
    const res = await fetch(api("/api/jobs"), { method: "POST", body });
    if (!res.ok) throw new Error("Upload failed.");
    created = await res.json();
  } catch (err) {
    errorBox.hidden = false;
    errorBox.textContent = "Could not reach the processing server.";
    go.disabled = false;
    return;
  }

  const timer = setInterval(async () => {
    let job;
    try {
      const res = await fetch(api("/api/jobs/" + created.id));
      job = await res.json();
    } catch (err) {
      return;
    }

    const pct = job.percent || 0;
    fill.style.width = pct + "%";
    const done = Math.floor((job.frames_done || 0) / 2);
    const half = Math.floor((job.frames_total || 0) / 2);
    status.textContent = half
      ? pct + "%  ·  " + done + " / " + half + " frames"
      : pct + "%";

    if (job.status === "done") {
      clearInterval(timer);
      go.disabled = false;
      const videoUrl = api("/api/jobs/" + created.id + "/video");
      download.href = videoUrl;
      download.hidden = false;
      drawTable(job.juggles || []);
      if (job.has_video) {
        player.hidden = false;
        player.src = videoUrl;
        player.load();
        player.onerror = () => {
          errorBox.hidden = false;
          errorBox.textContent = "Overlay is ready but this browser could not play it. Use the download link.";
        };
      }
    } else if (job.status === "error") {
      clearInterval(timer);
      go.disabled = false;
      errorBox.hidden = false;
      errorBox.textContent = job.error || "Something went wrong.";
    }
  }, 500);
});

function drawTable(juggles) {
  rows.innerHTML = "";
  for (const row of juggles) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + row.n + "</td>" +
      "<td>" + Number(row.time).toFixed(2) + "s</td>" +
      "<td>" + row.label.replaceAll("_", " ") + "</td>" +
      "<td>" + row.streak + "</td>";
    rows.appendChild(tr);
  }
  table.hidden = false;
}