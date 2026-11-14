
/* Touchduel live camera client.
   Captures each presented camera frame, gives it a timestamp, and sends frames
   in order. Server-side FIFO processing uses LiveJuggle.push(frame, t).
*/
const $ = (id) => document.getElementById(id);
const preview = $("preview");
const apiInput = $("api");
const startButton = $("start");
const stopButton = $("stop");

// Preserve the requested 1280px camera width for small-ball detection.
// Camera capture targets 30 FPS; queued frames are not replaced if inference lags.
const MAX_FRAME_WIDTH = 1280;
const JPEG_QUALITY = 0.84;
const MAX_SOCKET_BUFFERED_BYTES = 3 * 1024 * 1024;
const PACKET_MAGIC = new Uint8Array([0x54, 0x44, 0x4c, 0x31]); // "TDL1"
const PACKET_HEADER_BYTES = 16;

let cameraStream = null;
let socket = null;
let active = false;
let stopRequested = false;
let stopCommandSent = false;
let videoFrameCallbackId = null;
let fallbackCaptureTimer = null;
let sendPumpTimer = null;
let firstMediaTime = null;
let startPerformanceTime = 0;
let lastPresentedFrames = null;
let capturedFrames = 0;
let nextEncodedIndex = 0;
let nextSendIndex = 0;
let encodingPending = 0;
let encodedFrames = new Map();
let sendQueue = [];
let captureSkipped = 0;

apiInput.value = localStorage.getItem("touchduel-api") || "";
apiInput.addEventListener("change", () => {
  apiInput.value = apiInput.value.trim();
  localStorage.setItem("touchduel-api", apiInput.value);
});

function serverBase() {
  return apiInput.value.trim().replace(/\/$/, "") || location.origin;
}

function websocketUrl() {
  const url = new URL("/api/live", serverBase());
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function setStatus(message) {
  $("status").textContent = message;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function clearError() {
  $("error").hidden = true;
  $("error").textContent = "";
}

function resetDisplay() {
  $("count").textContent = "0";
  $("streak").textContent = "0";
  $("best").textContent = "0";
  $("last-part").textContent = "Last detected part: —";
}

function setControls(running) {
  startButton.disabled = running;
  stopButton.disabled = !running;
}

function resetQueues() {
  capturedFrames = 0;
  nextEncodedIndex = 0;
  nextSendIndex = 0;
  encodingPending = 0;
  encodedFrames = new Map();
  sendQueue = [];
  captureSkipped = 0;
  firstMediaTime = null;
  lastPresentedFrames = null;
  stopRequested = false;
  stopCommandSent = false;
}

async function startLive() {
  clearError();
  resetDisplay();
  resetQueues();
  startButton.disabled = true;
  setStatus("Requesting camera access…");

  if (!navigator.mediaDevices?.getUserMedia) {
    startButton.disabled = false;
    showError("Camera access is unavailable. Use HTTPS or localhost.");
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
    });
    preview.srcObject = cameraStream;
    await preview.play();
  } catch (err) {
    stopCamera();
    startButton.disabled = false;
    showError("Could not open the camera. Check camera permission and use HTTPS (or localhost).");
    setStatus("Camera unavailable.");
    return;
  }

  const settings = cameraStream.getVideoTracks()[0]?.getSettings?.() || {};
  const cameraFps = settings.frameRate || "unknown";
  setStatus(`Camera ready (${cameraFps} FPS reported). Connecting to analyzer…`);

  let ws;
  try {
    ws = new WebSocket(websocketUrl());
  } catch (err) {
    stopCamera();
    startButton.disabled = false;
    showError("The processing server address is invalid.");
    setStatus("Could not connect.");
    return;
  }

  socket = ws;
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    active = true;
    startPerformanceTime = performance.now();
    setControls(true);
    setStatus("Connected. Initializing LiveJuggle…");
    startFrameCapture();
  };

  ws.onmessage = (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (err) {
      return;
    }

    if (message.type === "ready") {
      setStatus("LiveJuggle ready. Capturing camera frames…");
      return;
    }

    if (message.type === "update" || message.type === "finished") {
      updateStats(message);
      if (message.type === "update") {
        const processFps = Number(message.processing_fps || 0).toFixed(1);
        const captureFps = Number(message.capture_fps || 0).toFixed(1);
        // capturedFrames - analyzed frames includes browser, network, server-queue,
        // and currently-processing frames, so it is a better end-to-end lag estimate.
        const lagFrames = Math.max(0, capturedFrames - Number(message.frames || 0));
        const lagSeconds = (lagFrames / Math.max(Number(message.capture_fps || 30), 1)).toFixed(1);
        setStatus(
          `Analyzed ${message.frames || 0} frames · processing ${processFps} FPS · ` +
          `camera ${captureFps} FPS · queued ${lagFrames} (~${lagSeconds}s)` +
          (captureSkipped ? ` · camera skipped ${captureSkipped}` : "")
        );
      } else {
        setStatus("Finished. All queued frames processed and lookahead flushed.");
        closeSocket();
        stopCamera();
        active = false;
        setControls(false);
      }
      return;
    }

    if (message.type === "error") {
      showError(message.message || "The live analyzer reported an error.");
      setStatus("Live tracking stopped.");
      active = false;
      stopCamera();
      closeSocket();
      setControls(false);
    }
  };

  ws.onerror = () => {
    showError("Could not communicate with the processing server. Check server.py and the server address.");
  };

  ws.onclose = () => {
    if (socket === ws) socket = null;
    active = false;
    stopFrameCapture();
    stopCamera();
    setControls(false);
    if (!stopCommandSent && !$("error").hidden) setStatus("Connection closed.");
    else if (!stopCommandSent) setStatus("Connection closed before the session finished.");
  };
}

function startFrameCapture() {
  stopFrameCapture();
  active = true;
  if (typeof preview.requestVideoFrameCallback === "function") {
    videoFrameCallbackId = preview.requestVideoFrameCallback(onVideoFrame);
  } else {
    // Fallback for browsers without requestVideoFrameCallback.
    fallbackCaptureTimer = setInterval(() => {
      captureFrame(performance.now(), null);
    }, 1000 / 30);
  }
}

function stopFrameCapture() {
  active = false;
  if (videoFrameCallbackId !== null && typeof preview.cancelVideoFrameCallback === "function") {
    preview.cancelVideoFrameCallback(videoFrameCallbackId);
  }
  videoFrameCallbackId = null;
  if (fallbackCaptureTimer !== null) clearInterval(fallbackCaptureTimer);
  fallbackCaptureTimer = null;
}

function onVideoFrame(now, metadata) {
  if (!active) return;
  videoFrameCallbackId = preview.requestVideoFrameCallback(onVideoFrame);
  captureFrame(now, metadata);
}

function captureFrame(now, metadata) {
  if (!active || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (!preview.videoWidth || !preview.videoHeight) return;

  let timestamp;
  if (metadata && Number.isFinite(metadata.mediaTime)) {
    if (firstMediaTime === null) firstMediaTime = metadata.mediaTime;
    timestamp = Math.max(0, metadata.mediaTime - firstMediaTime);

    if (Number.isFinite(metadata.presentedFrames)) {
      if (lastPresentedFrames !== null && metadata.presentedFrames > lastPresentedFrames + 1) {
        captureSkipped += metadata.presentedFrames - lastPresentedFrames - 1;
      }
      lastPresentedFrames = metadata.presentedFrames;
    }
  } else {
    timestamp = Math.max(0, (now - startPerformanceTime) / 1000);
  }

  const index = capturedFrames++;
  const scale = Math.min(1, MAX_FRAME_WIDTH / preview.videoWidth);
  const width = Math.max(1, Math.round(preview.videoWidth * scale));
  const height = Math.max(1, Math.round(preview.videoHeight * scale));

  // Create a per-frame canvas so later camera frames cannot overwrite this one
  // while JPEG encoding is still in progress.
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = width;
  frameCanvas.height = height;
  const context = frameCanvas.getContext("2d", { alpha: false });
  if (!context) {
    showError("Could not create a camera frame canvas.");
    stopLive();
    return;
  }
  context.drawImage(preview, 0, 0, width, height);
  encodingPending++;

  frameCanvas.toBlob((blob) => {
    encodingPending--;
    if (blob) encodedFrames.set(index, { blob, timestamp });
    else {
      encodedFrames.set(index, null);
      showError("The browser failed to encode a camera frame.");
    }
    flushEncodedInOrder();
    flushSendQueue();
    maybeSendStopCommand();
  }, "image/jpeg", JPEG_QUALITY);
}

function makePacket(timestamp, index, jpegBlob) {
  const header = new ArrayBuffer(PACKET_HEADER_BYTES);
  const bytes = new Uint8Array(header);
  bytes.set(PACKET_MAGIC, 0);
  const view = new DataView(header);
  view.setFloat64(4, timestamp, false); // network byte order
  view.setUint32(12, index >>> 0, false);
  return new Blob([header, jpegBlob], { type: "application/octet-stream" });
}

function flushEncodedInOrder() {
  while (encodedFrames.has(nextEncodedIndex)) {
    const item = encodedFrames.get(nextEncodedIndex);
    encodedFrames.delete(nextEncodedIndex);
    if (item) sendQueue.push(makePacket(item.timestamp, nextEncodedIndex, item.blob));
    nextEncodedIndex++;
  }
}

function flushSendQueue() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  while (sendQueue.length && socket.bufferedAmount < MAX_SOCKET_BUFFERED_BYTES) {
    const packet = sendQueue.shift();
    try {
      socket.send(packet);
      nextSendIndex++;
    } catch (err) {
      // Keep the current packet at the front; retry when the socket is writable.
      sendQueue.unshift(packet);
      break;
    }
  }

  if (sendQueue.length || (stopRequested && socket.bufferedAmount > 0)) {
    if (sendPumpTimer === null) {
      sendPumpTimer = setTimeout(() => {
        sendPumpTimer = null;
        flushSendQueue();
        maybeSendStopCommand();
      }, 20);
    }
  }
  maybeSendStopCommand();
}

function maybeSendStopCommand() {
  if (!stopRequested || stopCommandSent || !socket || socket.readyState !== WebSocket.OPEN) return;
  const allEncoded = encodingPending === 0 && nextEncodedIndex === capturedFrames;
  const allSent = sendQueue.length === 0 && socket.bufferedAmount === 0 && nextSendIndex === nextEncodedIndex;
  if (allEncoded && allSent) {
    stopCommandSent = true;
    setStatus("All captured frames sent. Waiting for the analyzer to drain its queue and flush lookahead…");
    socket.send(JSON.stringify({ cmd: "stop" }));
  }
}

function updateStats(message) {
  if (message.count !== null && message.count !== undefined) $("count").textContent = message.count;
  if (message.streak !== null && message.streak !== undefined) $("streak").textContent = message.streak;
  if (message.best_streak !== null && message.best_streak !== undefined) $("best").textContent = message.best_streak;
  if (message.label) $("last-part").textContent = "Last detected part: " + String(message.label).replaceAll("_", " ");
}

function stopLive() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    stopFrameCapture();
    stopCamera();
    setControls(false);
    setStatus("Stopped.");
    return;
  }
  if (stopRequested) return;
  stopRequested = true;
  stopFrameCapture();
  setControls(false);
  setStatus("Stopping capture. Sending every captured frame before finalizing…");
  flushEncodedInOrder();
  flushSendQueue();
  maybeSendStopCommand();
}

function closeSocket() {
  if (sendPumpTimer !== null) clearTimeout(sendPumpTimer);
  sendPumpTimer = null;
  if (!socket) return;
  const old = socket;
  socket = null;
  if (old.readyState === WebSocket.OPEN || old.readyState === WebSocket.CONNECTING) old.close();
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  preview.pause();
  preview.srcObject = null;
}

startButton.addEventListener("click", startLive);
stopButton.addEventListener("click", stopLive);
$("reset").addEventListener("click", () => {
  if (active || stopRequested) return;
  resetDisplay();
});

window.addEventListener("beforeunload", () => {
  stopFrameCapture();
  if (socket) socket.close();
  stopCamera();
});
