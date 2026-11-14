const DURATION = 30;
const IDLE_BREAK = 1.5; // seconds without a juggle before the HUD shows the streak as broken
const SECTIONS = ["menu", "lobby", "work", "arena"];
const ROOM_KEY = "touchduel-room";

const $ = (id) => document.getElementById(id);
const apiInput = $("api");
const seedMode = new URLSearchParams(location.search).has("seed");
const canRecord =
  window.isSecureContext &&
  !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) &&
  typeof MediaRecorder !== "undefined";

apiInput.value = localStorage.getItem("touchduel-api") || "";
apiInput.addEventListener("change", () => {
  localStorage.setItem("touchduel-api", apiInput.value.trim());
});
if (seedMode) $("seed-note").hidden = false;

function api(path) {
  return apiInput.value.trim().replace(/\/$/, "") + path;
}

async function call(path, options) {
  let res;
  try {
    res = await fetch(api(path), options);
  } catch (err) {
    const e = new Error("Could not reach the processing server.");
    e.status = 0;
    throw e;
  }
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail || "";
    } catch (err) {}
    const e = new Error(detail || "Request failed.");
    e.status = res.status;
    throw e;
  }
  return res.json();
}

function show(name) {
  for (const s of SECTIONS) $(s).hidden = s !== name;
}

function fail(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function clearError() {
  $("error").hidden = true;
}

let session = null; // { mode: "room", code, player } or { mode: "solo" }
let pollTimer = null;
let polling = false;
let clock = { rtt: Infinity, offset: 0 }; // offset = server time minus this phone's time
const countdowns = new Set();
let stream = null;
let recorder = null;

/* ---------- menu ---------- */

$("create").addEventListener("click", async () => {
  clearError();
  try {
    const room = await call("/api/rooms", { method: "POST" });
    enterRoom(room.code, room.player);
  } catch (err) {
    fail(err.message);
  }
});

$("join").addEventListener("click", async () => {
  clearError();
  const code = $("code").value.trim().toUpperCase();
  if (code.length !== 4) {
    fail("Enter the 4 letter room code.");
    return;
  }
  try {
    const room = await call("/api/rooms/" + code + "/join", { method: "POST" });
    enterRoom(room.code, room.player);
  } catch (err) {
    fail(err.message);
  }
});

$("solo-file").addEventListener("change", () => {
  const file = $("solo-file").files[0];
  $("solo-name").textContent = file ? file.name : "Solo: pick a video";
  $("solo-go").disabled = !file;
});

$("solo-go").addEventListener("click", runSolo);

for (const k of ["a", "b"]) {
  $("cmp-" + k).addEventListener("change", () => {
    const file = $("cmp-" + k).files[0];
    $("cmp-" + k + "-name").textContent = file ? file.name : "Clip " + k.toUpperCase();
    $("cmp-go").disabled = !($("cmp-a").files[0] && $("cmp-b").files[0]);
  });
}

$("cmp-go").addEventListener("click", runCompare);

$("again").addEventListener("click", () => {
  sessionStorage.removeItem(ROOM_KEY);
  location.reload();
});

/* ---------- room: lobby, countdown, recording ---------- */

function enterRoom(code, player) {
  session = { mode: "room", code, player };
  sessionStorage.setItem(ROOM_KEY, JSON.stringify(session));
  $("room-code").textContent = code;
  $("lobby-status").textContent = "";
  $("cam-open").hidden = !canRecord;
  $("fallback").hidden = canRecord;
  $("lobby-note").hidden = canRecord;
  show("lobby");
  startPolling();
}

function leaveRoom() {
  stopPolling();
  sessionStorage.removeItem(ROOM_KEY);
  session = null;
  show("menu");
}

function startPolling() {
  stopPolling();
  pollRoom();
  pollTimer = setInterval(pollRoom, 700);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

async function pollRoom() {
  if (polling || !session || session.mode !== "room") return;
  polling = true;
  try {
    const t0 = Date.now();
    const room = await call("/api/rooms/" + session.code);
    const t1 = Date.now();
    if (t1 - t0 < clock.rtt) {
      clock = { rtt: t1 - t0, offset: room.server_ms - (t0 + t1) / 2 };
    }
    handleRoom(room);
  } catch (err) {
    if (err.status === 404) {
      leaveRoom();
      fail("That room no longer exists.");
    }
  } finally {
    polling = false;
  }
}

function sideFrom(p) {
  return { id: p.job, stats: p.stats, juggles: p.juggles };
}

function handleRoom(room) {
  const me = room.players[session.player];
  const opp = room.players[session.player === "a" ? "b" : "a"];

  if (room.state === "error") {
    stopPolling();
    show("menu");
    fail(me.error || opp.error || "A clip could not be scored.");
    return;
  }
  if (room.state === "done") {
    showArena(sideFrom(me), sideFrom(opp), "Opponent");
    return;
  }
  if (me.job) {
    show("work");
    $("label-me").textContent = "You";
    $("label-opp").textContent = "Opponent";
    setBar("me", me);
    setBar("opp", opp);
    return;
  }

  if (!opp.joined) {
    $("lobby-status").textContent = "Waiting for your opponent to join. Share the code.";
  } else if (canRecord && !stream) {
    $("lobby-status").textContent = "Opponent joined. Open your camera.";
  } else if (canRecord) {
    $("lobby-status").textContent = opp.ready ? "Opponent is ready." : "Waiting for your opponent to get ready.";
  } else {
    $("lobby-status").textContent = "Opponent joined. Record your 30 seconds.";
  }

  if (room.start_at && stream && !countdowns.has(room.start_at)) {
    countdowns.add(room.start_at);
    runCountdown(room.start_at);
  }
}

$("cam-open").addEventListener("click", async () => {
  clearError();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    fail("Camera permission was blocked.");
    return;
  }
  $("preview").srcObject = stream;
  try {
    await $("preview").play();
  } catch (err) {}
  $("cam").hidden = false;
  $("cam-open").hidden = true;
  $("ready").hidden = false;
});

$("ready").addEventListener("click", async () => {
  clearError();
  $("ready").disabled = true;
  try {
    await call("/api/rooms/" + session.code + "/ready?player=" + session.player, { method: "POST" });
    $("lobby-status").textContent = "Waiting for your opponent to get ready.";
  } catch (err) {
    $("ready").disabled = false;
    fail(err.message);
  }
});

function runCountdown(startAt) {
  const target = startAt - clock.offset; // start time on this phone's clock
  const box = $("countdown");
  $("ready").hidden = true;
  const tick = () => {
    const left = target - Date.now();
    if (left > 0) {
      box.textContent = Math.ceil(left / 1000);
      setTimeout(tick, 50);
    } else {
      beginRecording();
    }
  };
  tick();
}

function pickMime() {
  const options = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm", "video/mp4"];
  for (const type of options) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function beginRecording() {
  const chunks = [];
  const mime = pickMime();
  try {
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  } catch (err) {
    fail("This browser cannot record video.");
    return;
  }
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size) chunks.push(e.data);
  };
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    $("cam").hidden = true;
    const type = recorder.mimeType || mime || "video/webm";
    const ext = type.includes("mp4") ? "mp4" : "webm";
    const file = new File([new Blob(chunks, { type })], "clip." + ext, { type });
    $("lobby-status").textContent = "Uploading…";
    try {
      await uploadClip(file);
    } catch (err) {
      fail(err.message);
    }
  };

  const began = Date.now();
  $("countdown").textContent = "GO";
  recorder.start(1000);
  const timer = setInterval(() => {
    const left = Math.max(0, DURATION - (Date.now() - began) / 1000);
    $("countdown").textContent = Math.ceil(left);
  }, 250);
  setTimeout(() => {
    clearInterval(timer);
    $("countdown").textContent = "";
    if (recorder.state !== "inactive") recorder.stop();
  }, DURATION * 1000);
}

$("cap").addEventListener("change", async () => {
  const file = $("cap").files[0];
  if (!file) return;
  clearError();
  $("lobby-status").textContent = "Uploading…";
  try {
    await uploadClip(file);
  } catch (err) {
    fail(err.message);
  }
});

async function uploadClip(file) {
  const body = new FormData();
  body.append("file", file);
  if (session && session.mode === "room") {
    body.append("room", session.code);
    body.append("player", session.player);
  }
  if (seedMode) body.append("seed", "true");
  const created = await call("/api/duel/jobs", { method: "POST", body });
  return created.id;
}

/* ---------- progress bars ---------- */

function setBar(prefix, p) {
  let pct = 0;
  let text = "Waiting for clip…";
  if (p.job) {
    if (p.status === "done") {
      pct = 100;
      text = "Done";
    } else if (p.stage === "queued") {
      text = "Queued…";
    } else if (p.stage === "trimming") {
      text = "Preparing clip…";
    } else {
      pct = p.percent || 0;
      text = pct + "%";
    }
  }
  $("fill-" + prefix).style.width = pct + "%";
  $("text-" + prefix).textContent = text;
}

/* ---------- solo vs random opponent ---------- */

async function runSolo() {
  const file = $("solo-file").files[0];
  if (!file) return;
  clearError();
  session = { mode: "solo" };
  show("work");
  $("label-me").textContent = "You";
  $("label-opp").textContent = "Random opponent";
  setBar("me", {});
  setBar("opp", {});
  $("text-opp").textContent = "Picked when yours finishes";

  let id;
  try {
    id = await uploadClip(file);
  } catch (err) {
    show("menu");
    fail(err.message);
    return;
  }

  const timer = setInterval(async () => {
    let job;
    try {
      job = await call("/api/jobs/" + id);
    } catch (err) {
      return;
    }
    setBar("me", Object.assign({ job: id }, job));
    if (job.status === "error") {
      clearInterval(timer);
      show("menu");
      fail(job.error || "Scoring failed.");
    } else if (job.status === "done") {
      clearInterval(timer);
      try {
        const opp = await call("/api/duel/opponent?exclude=" + id);
        showArena(
          { id, stats: job.stats, juggles: job.juggles },
          { id: opp.id, stats: opp.stats, juggles: opp.juggles },
          "Random opponent"
        );
      } catch (err) {
        show("menu");
        fail(
          err.status === 404
            ? "No opponents yet. Open duel.html?seed=1 and upload a clip to add one."
            : err.message
        );
      }
    }
  }, 500);
}

/* ---------- compare two clips ---------- */

async function runCompare() {
  clearError();
  session = { mode: "compare" };
  show("work");
  $("label-me").textContent = "Clip A";
  $("label-opp").textContent = "Clip B";
  setBar("me", {});
  setBar("opp", {});

  let ids;
  try {
    ids = await Promise.all([uploadClip($("cmp-a").files[0]), uploadClip($("cmp-b").files[0])]);
  } catch (err) {
    show("menu");
    fail(err.message);
    return;
  }

  const timer = setInterval(async () => {
    let found;
    try {
      found = await Promise.all(ids.map((id) => call("/api/jobs/" + id)));
    } catch (err) {
      return;
    }
    setBar("me", Object.assign({ job: ids[0] }, found[0]));
    setBar("opp", Object.assign({ job: ids[1] }, found[1]));
    const bad = found.find((j) => j.status === "error");
    if (bad) {
      clearInterval(timer);
      show("menu");
      fail(bad.error || "Scoring failed.");
    } else if (found.every((j) => j.status === "done")) {
      clearInterval(timer);
      showArena(
        { id: ids[0], stats: found[0].stats, juggles: found[0].juggles },
        { id: ids[1], stats: found[1].stats, juggles: found[1].juggles },
        "Clip B",
        "Clip A"
      );
    }
  }, 500);
}

/* ---------- arena ---------- */

const sides = {};
let raf = 0;

function emptyStats() {
  return { total: 0, longest_streak: 0, avg_gap: null, parts: {} };
}

function setupSide(key, data) {
  const video = $("vid-" + key);
  video.src = api("/api/jobs/" + data.id + "/video");
  video.load();
  video.onerror = () => fail("Could not play one of the overlay videos.");
  video.onended = checkEnd;
  return {
    video,
    juggles: (data.juggles || []).slice().sort((a, b) => a.time - b.time),
    stats: data.stats || emptyStats(),
    shown: "",
  };
}

function showArena(me, opp, oppName, meName = "You") {
  stopPolling();
  clearError();
  show("arena");
  $("name-me").textContent = meName;
  $("name-opp").textContent = oppName;
  $("banner").hidden = true;
  $("final-me").hidden = true;
  $("final-opp").hidden = true;
  $("play").textContent = "Start duel";
  sides.me = setupSide("me", me);
  sides.opp = setupSide("opp", opp);
  sides.me.name = meName;
  sides.opp.name = oppName;
  cancelAnimationFrame(raf);
  loop();
}

function loop() {
  updateHud("me");
  updateHud("opp");
  raf = requestAnimationFrame(loop);
}

function updateHud(key) {
  const s = sides[key];
  const t = s.video.currentTime;
  let count = 0;
  let streak = 0;
  let best = 0;
  let last = -1;
  for (const j of s.juggles) {
    if (j.time > t) break;
    count = j.n;
    streak = j.streak;
    best = Math.max(best, j.streak);
    last = j.time;
  }
  if (last >= 0 && t - last > IDLE_BREAK) streak = 0;
  const signature = count + "/" + streak + "/" + best;
  if (signature === s.shown) return;
  s.shown = signature;
  $("hud-count-" + key).textContent = count;
  $("hud-streak-" + key).textContent = streak;
  $("hud-best-" + key).textContent = best;
}

function whenReady(video) {
  return new Promise((resolve) => {
    if (video.readyState >= 3) resolve();
    else video.addEventListener("canplay", resolve, { once: true });
  });
}

async function startDuel() {
  $("play").disabled = true;
  $("banner").hidden = true;
  $("final-me").hidden = true;
  $("final-opp").hidden = true;
  for (const s of [sides.me, sides.opp]) {
    s.video.pause();
    s.video.currentTime = 0;
  }
  await Promise.all([whenReady(sides.me.video), whenReady(sides.opp.video)]);
  try {
    await Promise.all([sides.me.video.play(), sides.opp.video.play()]);
  } catch (err) {
    fail("Playback was blocked. Tap Start duel again.");
  }
  $("play").disabled = false;
}

$("play").addEventListener("click", startDuel);

function checkEnd() {
  if (!(sides.me.video.ended && sides.opp.video.ended)) return;
  const a = sides.me.stats;
  const b = sides.opp.stats;
  const wins = (side) => (side.name === "You" ? "You win" : side.name + " wins");
  let verdict = "Draw";
  let note = "";
  if (a.total !== b.total) {
    verdict = wins(a.total > b.total ? sides.me : sides.opp);
  } else if (a.longest_streak !== b.longest_streak) {
    verdict = wins(a.longest_streak > b.longest_streak ? sides.me : sides.opp);
    note = " (longest streak)";
  }
  $("banner").textContent = verdict + "  ·  " + a.total + " – " + b.total + note;
  $("banner").hidden = false;
  renderFinal("me");
  renderFinal("opp");
  $("play").textContent = "Replay";
}

function renderFinal(key) {
  const s = sides[key].stats;
  const p = s.parts || {};
  const rows = [
    ["Juggles", s.total],
    ["Longest streak", s.longest_streak],
    ["Avg gap", s.avg_gap == null ? "–" : s.avg_gap + "s"],
    ["Left foot", p.left_foot || 0],
    ["Right foot", p.right_foot || 0],
    ["Left knee", p.left_knee || 0],
    ["Right knee", p.right_knee || 0],
  ];
  $("final-" + key).innerHTML = rows
    .map(([k, v]) => "<div><dt>" + k + "</dt><dd>" + v + "</dd></div>")
    .join("");
  $("final-" + key).hidden = false;
}

/* ---------- resume after a reload ---------- */

(async function resume() {
  const saved = sessionStorage.getItem(ROOM_KEY);
  if (!saved) return;
  try {
    const s = JSON.parse(saved);
    await call("/api/rooms/" + s.code);
    enterRoom(s.code, s.player);
  } catch (err) {
    sessionStorage.removeItem(ROOM_KEY);
  }
})();