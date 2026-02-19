
(function () {
'use strict';



const CONFIG = {
  MAX_SNAPS:           12,
  IDEAL_WIDTH:         1280,
  IDEAL_HEIGHT:        720,
  STATS_FLUSH_EVERY:   8,      // frames between stats DOM writes (~4 Hz at 30fps)
  FRAME_COUNTER_EVERY: 5,      // frames between framesEl DOM writes
  STREAM_TIMEOUT_MS:   8000,   // waitForVideoDimensions timeout
  SNAP_REVOKE_MS:      10000,  // delay before revoking a Blob URL after download
  MEDIAPIPE_CDN:       'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/',
  MIN_DETECTION_CONF:  0.5,
};



// classList.replace() is absent in some older browsers — use these helpers everywhere.
function clsAdd(el, cls)         { el.classList.add(cls); }
function clsRemove(el, cls)      { el.classList.remove(cls); }
function clsSet(el, cls, active) { active ? clsAdd(el, cls) : clsRemove(el, cls); }

// String.prototype.padStart polyfill (needed in some older WebViews)
if (!String.prototype.padStart) {
  String.prototype.padStart = function (len, fill) {
    let s = String(this);
    fill = fill || ' ';
    while (s.length < len) s = fill + s;
    return s;
  };
}

// Monotonic high-res timer with Date.now() fallback
const perf = (typeof performance !== 'undefined' && performance.now)
  ? () => performance.now()
  : () => Date.now();


const DOM = {
  // Header
  dotCam:       document.getElementById('dot-cam'),
  dotModel:     document.getElementById('dot-model'),
  dotDetect:    document.getElementById('dot-detect'),
  clock:        document.getElementById('clock'),

  // Viewport
  video:        document.getElementById('video'),
  overlay:      document.getElementById('overlay'),
  flash:        document.getElementById('flash'),
  idleMsg:      document.getElementById('idle-msg'),
  idleText:     document.getElementById('idle-text'),
  resLabel:     document.getElementById('res-label'),
  detBadge:     document.getElementById('det-badge'),
  banner:       document.getElementById('banner'),

  // Controls
  startBtn:     document.getElementById('start-btn'),
  toggleBtn:    document.getElementById('toggle-btn'),
  snapBtn:      document.getElementById('snap-btn'),
  loadBar:      document.getElementById('load-bar'),
  loadFill:     document.getElementById('load-fill'),

  // Gallery
  gallery:          document.getElementById('gallery'),
  galleryWrapper:   document.getElementById('gallery-wrapper'),
  galleryEmpty:     document.getElementById('gallery-empty'),
  snapCountLbl:     document.getElementById('snap-count-lbl'),
  galleryScrollUp:  document.getElementById('gallery-scroll-up'),
  galleryScrollDown:document.getElementById('gallery-scroll-down'),

  // Metrics
  faceCount:    document.getElementById('face-count'),
  fpsVal:       document.getElementById('fps-val'),
  latencyVal:   document.getElementById('latency-val'),
  confVal:      document.getElementById('conf-val'),
  framesVal:    document.getElementById('frames-val'),

  // Stats
  statTotal:    document.getElementById('stat-total'),
  statMax:      document.getElementById('stat-max'),
  statAvg:      document.getElementById('stat-avg'),
  statConf:     document.getElementById('stat-conf'),
  statTime:     document.getElementById('stat-time'),
  statSnaps:    document.getElementById('stat-snaps'),

  // Frequency chart — cached to avoid repeated getElementById in hot path
  barFills: [0, 1, 2, 3].map(i => document.getElementById(`bar-${i}`)),
  barVals:  [0, 1, 2, 3].map(i => document.getElementById(`bv-${i}`)),

  // Lightbox
  lightbox:     document.getElementById('lightbox'),
  lbImg:        document.getElementById('lb-img'),
  lbMeta:       document.getElementById('lb-meta'),
  lbDl:         document.getElementById('lb-dl'),
  lbClose:      document.getElementById('lb-close'),

  // Canvas 2D context (derived, not an element)
  ctx:          document.getElementById('overlay').getContext('2d'),

  // System log
  log:          document.getElementById('log'),
};


const State = {
  // Session lifecycle
  running:       false,
  initializing:  false,   // prevents double-click race on start button
  detectionOn:   true,

  // MediaPipe handles
  mpCamera:      null,
  faceDetector:  null,
  modelReady:    false,
  inFlight:      false,   // prevents queuing multiple faceDetector.send() calls
  lastDetections: [],

  // Stream
  activeStream:  null,
  permStatus:    null,    // PermissionStatus handle — kept for .onchange listener

  // Canvas size cache — resizing canvas is expensive; only do it when dims change
  overlayW: 0,
  overlayH: 0,

  // FPS measurement
  frameCount:     0,
  fpsWindowStart: perf(),

  // Session statistics
  sessionStart:    null,
  sessionInterval: null,
  totalFaceSum:    0,
  detFrames:       0,
  maxSimul:        0,
  confSum:         0,
  confSamples:     0,
  freqBuckets:     [0, 0, 0, 0],

  // Stats DOM throttle
  statsDirty:        false,
  statFlushCounter:  0,

  // Snapshots (in-memory only — never transmitted anywhere)
  snapshots: [],
};

/** Reset all session-level counters for a fresh start. */
function resetSessionState() {
  State.totalFaceSum   = 0;
  State.detFrames      = 0;
  State.maxSimul       = 0;
  State.confSum        = 0;
  State.confSamples    = 0;
  State.freqBuckets    = [0, 0, 0, 0];
  State.statsDirty     = false;
  State.statFlushCounter = 0;
  State.frameCount     = 0;
  State.fpsWindowStart = perf();
}


function log(msg, type = '') {
  const ts    = new Date().toTimeString().slice(0, 8);
  const entry = document.createElement('div');
  entry.className = `le${type ? ' ' + type : ''}`;

  // Use textContent to prevent XSS from error message strings
  const tsSpan = document.createElement('span');
  tsSpan.className   = 'ts';
  tsSpan.textContent = `[${ts}] `;

  entry.appendChild(tsSpan);
  entry.appendChild(document.createTextNode(msg));
  DOM.log.insertBefore(entry, DOM.log.firstChild);

  // Keep the log from growing unbounded
  if (DOM.log.children.length > 60) DOM.log.removeChild(DOM.log.lastChild);
}

/** Show the permission/error banner. */
function showBanner(msg, isError = false) {
  DOM.banner.textContent = msg;
  clsSet(DOM.banner, 'error', isError);
  clsAdd(DOM.banner, 'visible');
}

/** Hide the banner. */
function hideBanner() { clsRemove(DOM.banner, 'visible'); }


/** Called once per detection frame to accumulate raw numbers. */
function accumulateStats(detections) {
  const n = detections.length;
  State.totalFaceSum += n;
  State.detFrames++;
  if (n > State.maxSimul) State.maxSimul = n;

  for (const det of detections) {
    State.confSum += (det.score && det.score[0] != null) ? det.score[0] : 0.5;
    State.confSamples++;
  }

  State.freqBuckets[Math.min(n, 3)]++;
  State.statsDirty = true;

  State.statFlushCounter++;
  if (State.statFlushCounter >= CONFIG.STATS_FLUSH_EVERY) {
    State.statFlushCounter = 0;
    flushStatsDom();
  }
}

/** Write accumulated stats to the DOM (called at ~4 Hz, not 30 Hz). */
function flushStatsDom() {
  if (!State.statsDirty) return;
  State.statsDirty = false;

  DOM.statTotal.textContent = State.totalFaceSum;
  DOM.statMax.textContent   = State.maxSimul;
  DOM.statAvg.textContent   = State.detFrames > 0
    ? (State.totalFaceSum / State.detFrames).toFixed(2)
    : '0.00';
  DOM.statConf.textContent  = State.confSamples > 0
    ? `${Math.round((State.confSum / State.confSamples) * 100)}%`
    : '—';

  const maxBucket = Math.max(...State.freqBuckets, 1);
  for (let i = 0; i <= 3; i++) {
    DOM.barFills[i].style.width  = `${Math.round(State.freqBuckets[i] / maxBucket * 100)}%`;
    DOM.barVals[i].textContent   = State.freqBuckets[i];
  }
}

/** Reset stats DOM to initial values. */
function resetStatsDom() {
  DOM.statTotal.textContent = '0';
  DOM.statMax.textContent   = '0';
  DOM.statAvg.textContent   = '0.00';
  DOM.statConf.textContent  = '—';
  DOM.statTime.textContent  = '0s';
  for (let i = 0; i <= 3; i++) {
    DOM.barFills[i].style.width = '0%';
    DOM.barVals[i].textContent  = '0';
  }
}

function startSessionTimer() {
  State.sessionStart = Date.now();
  State.sessionInterval = setInterval(() => {
    const s = Math.floor((Date.now() - State.sessionStart) / 1000);
    const m = Math.floor(s / 60);
    DOM.statTime.textContent = m > 0
      ? `${m}m ${String(s % 60).padStart(2, '0')}s`
      : `${s}s`;
  }, 1000);
}

function stopSessionTimer() {
  clearInterval(State.sessionInterval);
  State.sessionInterval = null;
}



/** Sync canvas logical size to the current video dimensions (no-op if unchanged). */
function syncOverlaySize() {
  const vw = DOM.video.videoWidth;
  const vh = DOM.video.videoHeight;
  if (vw > 0 && vh > 0 && (vw !== State.overlayW || vh !== State.overlayH)) {
    DOM.overlay.width  = vw;
    DOM.overlay.height = vh;
    State.overlayW = vw;
    State.overlayH = vh;
  }
}

/** Erase the overlay canvas without resizing it. */
function clearOverlay() {
  syncOverlaySize();
  DOM.ctx.clearRect(0, 0, State.overlayW, State.overlayH);
}

/**
 * Draw four corner brackets for a bounding box.
 * Extracted from the per-face loop to avoid allocating a temporary array on every call.
 */
function drawCornerBrackets(ctx, x, y, bw, bh, cs) {
  ctx.beginPath(); ctx.moveTo(x,      y + cs); ctx.lineTo(x,      y);      ctx.lineTo(x + cs, y);      ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x + bw - cs, y); ctx.lineTo(x + bw, y);      ctx.lineTo(x + bw, y + cs); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x,      y + bh - cs); ctx.lineTo(x,      y + bh); ctx.lineTo(x + cs, y + bh); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x + bw - cs, y + bh); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw, y + bh - cs); ctx.stroke();
}

/** Render all detected face bounding boxes, labels, and landmarks. */
function drawDetections(detections) {
  syncOverlaySize();
  const ctx = DOM.ctx;
  const w   = State.overlayW;
  const h   = State.overlayH;
  ctx.clearRect(0, 0, w, h);

  let maxConf = 0;

  for (let i = 0; i < detections.length; i++) {
    const det   = detections[i];
    const bb    = det.boundingBox;
    const x     = bb.xCenter * w - (bb.width  * w) / 2;
    const y     = bb.yCenter * h - (bb.height * h) / 2;
    const bw    = bb.width  * w;
    const bh    = bb.height * h;
    const score = (det.score && det.score[0] != null) ? det.score[0] : 0.5;
    if (score > maxConf) maxConf = score;

    // ── Bounding box ──
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur  = 16;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth   = 2;
    ctx.strokeRect(x, y, bw, bh);

    // ── Corner brackets ──
    ctx.shadowBlur = 22;
    ctx.lineWidth  = 3;
    drawCornerBrackets(ctx, x, y, bw, bh, 14);
    ctx.shadowBlur = 0;

    // ── Label ──
    const label = `FACE_${String(i).padStart(2, '0')}  ${Math.round(score * 100)}%`;
    ctx.font = '11px Share Tech Mono, monospace';
    const tw  = ctx.measureText(label).width;
    const ly  = y - 20 < 0 ? 0 : y - 20;
    ctx.fillStyle = 'rgba(0,0,0,0.76)';
    ctx.fillRect(x, ly, tw + 10, 18);
    ctx.fillStyle   = '#00ff88';
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur  = 6;
    ctx.fillText(label, x + 5, ly + 13);
    ctx.shadowBlur  = 0;

    // ── Landmarks ──
    if (det.landmarks) {
      ctx.fillStyle   = '#00ff88';
      ctx.shadowColor = '#00ff88';
      ctx.shadowBlur  = 8;
      for (const pt of det.landmarks) {
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
  }

  DOM.faceCount.textContent = detections.length;
  DOM.confVal.textContent   = detections.length > 0
    ? `${Math.round(maxConf * 100)}%`
    : '—';
}


/** Capture the current video frame + overlay into an in-memory PNG. */
function takeSnapshot() {
  if (!State.running) return;

  const vw = DOM.video.videoWidth;
  const vh = DOM.video.videoHeight;
  if (!vw || !vh) {
    log('Snapshot skipped — video not ready', 'wn');
    return;
  }

  const sc   = document.createElement('canvas');
  sc.width   = vw;
  sc.height  = vh;
  const sctx = sc.getContext('2d');

  // Draw mirrored video frame
  sctx.save();
  sctx.translate(vw, 0);
  sctx.scale(-1, 1);
  sctx.drawImage(DOM.video, 0, 0, vw, vh);
  sctx.restore();

  // Draw mirrored overlay (bounding boxes) on top
  if (State.detectionOn && State.overlayW > 0) {
    sctx.save();
    sctx.translate(vw, 0);
    sctx.scale(-1, 1);
    sctx.drawImage(DOM.overlay, 0, 0, vw, vh);
    sctx.restore();
  }

  // Timestamp watermark
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
  sctx.font = '12px Share Tech Mono, monospace';
  const tw  = sctx.measureText(ts).width;
  sctx.fillStyle = 'rgba(0,0,0,0.6)';
  sctx.fillRect(8, vh - 26, tw + 12, 19);
  sctx.fillStyle = '#00ff88';
  sctx.fillText(ts, 14, vh - 12);

  let dataUrl;
  try {
    dataUrl = sc.toDataURL('image/png');  // stays in RAM — never transmitted
  } catch (e) {
    // SecurityError if canvas were tainted — shouldn't happen with getUserMedia
    log(`Snapshot failed — canvas tainted: ${e.message}`, 'er');
    return;
  }

  const snap = { id: Date.now(), dataUrl, ts, faceCount: State.lastDetections.length };
  if (State.snapshots.length >= CONFIG.MAX_SNAPS) State.snapshots.shift();
  State.snapshots.push(snap);

  renderGallery();
  DOM.statSnaps.textContent = State.snapshots.length;
  log(`Snapshot #${State.snapshots.length} — ${snap.faceCount} face(s)`, 'hl');

  // White flash using setTimeout to guarantee a full paint cycle between add/remove
  clsAdd(DOM.flash, 'go');
  setTimeout(() => clsRemove(DOM.flash, 'go'), 80);
}

/** Re-render the snapshot gallery from State.snapshots (newest first). */
function renderGallery() {
  DOM.gallery.innerHTML = '';
  DOM.snapCountLbl.textContent = `${State.snapshots.length} / ${CONFIG.MAX_SNAPS}`;

  if (State.snapshots.length === 0) {
    DOM.gallery.appendChild(DOM.galleryEmpty);
    updateGalleryScroll();
    return;
  }

  for (let i = State.snapshots.length - 1; i >= 0; i--) {
    DOM.gallery.appendChild(buildThumb(State.snapshots[i]));
  }
  // Scroll to top so newest snap is always visible after capture
  DOM.gallery.scrollTop = 0;
  updateGalleryScroll();
}

function updateGalleryScroll() {
  requestAnimationFrame(() => {
    const el        = DOM.gallery;
    const wrapper   = DOM.galleryWrapper;
    const scrollTop = el.scrollTop;
    const maxScroll = el.scrollHeight - el.clientHeight;
    const canUp     = scrollTop > 1;
    const canDown   = maxScroll > 1 && scrollTop < maxScroll - 1;

    DOM.galleryScrollUp.disabled   = !canUp;
    DOM.galleryScrollDown.disabled = !canDown;

    wrapper.classList.toggle('can-scroll-up',   canUp);
    wrapper.classList.toggle('can-scroll-down', canDown);
  });
}


function scrollGalleryBy(direction) {
  const firstThumb = DOM.gallery.firstElementChild;
  const rowHeight  = firstThumb
    ? firstThumb.offsetHeight + 5   // thumb height + gap
    : 77;                            // fallback matches grid-auto-rows: 72px + 5px gap
  DOM.gallery.scrollBy({ top: direction * rowHeight, behavior: 'smooth' });
}

/** Build a single snapshot thumbnail element. */
function buildThumb(snap) {
  const div = document.createElement('div');
  div.className = 'snap-thumb';
  div.setAttribute('role', 'listitem');
  div.setAttribute('tabindex', '0');
  div.setAttribute('aria-label', `Snapshot from ${snap.ts}, ${snap.faceCount} faces`);

  const img = document.createElement('img');
  img.src = snap.dataUrl;
  img.alt = `Snapshot at ${snap.ts}`;

  const badge = document.createElement('div');
  badge.className   = 'snap-fc';
  badge.textContent = `${snap.faceCount}F`;
  badge.setAttribute('aria-hidden', 'true');

  const del = document.createElement('button');
  del.className   = 'snap-del';
  del.textContent = '✕';
  del.setAttribute('aria-label', 'Delete this snapshot');
  del.addEventListener('click', e => {
    e.stopPropagation();
    State.snapshots = State.snapshots.filter(s => s.id !== snap.id);
    renderGallery();
    DOM.statSnaps.textContent = State.snapshots.length;
    log('Snapshot deleted');
  });

  div.appendChild(img);
  div.appendChild(badge);
  div.appendChild(del);
  div.addEventListener('click', () => openLightbox(snap));
  div.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openLightbox(snap); }
  });
  return div;
}

/** Open the full-size lightbox for a snapshot. */
function openLightbox(snap) {
  DOM.lbImg.src = snap.dataUrl;
  DOM.lbImg.alt = `Snapshot at ${snap.ts}, ${snap.faceCount} faces detected`;
  DOM.lbMeta.textContent = `${snap.ts}  ·  ${snap.faceCount} face(s) detected`;
  DOM.lbDl.onclick = () => downloadSnap(snap);
  clsAdd(DOM.lightbox, 'open');
}

/** Close the lightbox. */
function closeLightbox() { clsRemove(DOM.lightbox, 'open'); }


function downloadSnap(snap) {
  try {
    const [header, b64] = snap.dataUrl.split(',');
    const mime   = header.match(/:(.*?);/)[1];
    const raw    = atob(b64);
    const bytes  = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const blob   = new Blob([bytes], { type: mime });
    const url    = URL.createObjectURL(blob);

    triggerDownload(url, `face-detect-${snap.id}.png`);
    setTimeout(() => URL.revokeObjectURL(url), CONFIG.SNAP_REVOKE_MS);

  } catch (_) {
    // Fallback: direct dataURL download (may not work in Safari for large files)
    triggerDownload(snap.dataUrl, `face-detect-${snap.id}.png`);
  }
}

/** Programmatically click a temporary <a download> link. */
function triggerDownload(href, filename) {
  const a   = document.createElement('a');
  a.href     = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}


function ensureModelReady() {
  return new Promise((resolve, reject) => {
    // Already warmed up — nothing to do
    if (State.modelReady && State.faceDetector) {
      resolve(); return;
    }

    // Instance exists but init hasn't resolved yet — re-await it
    if (State.faceDetector) {
      State.faceDetector.initialize()
        .then(() => { State.modelReady = true; resolve(); })
        .catch(reject);
      return;
    }

    // First-time creation
    State.faceDetector = new FaceDetection({
      locateFile: f => CONFIG.MEDIAPIPE_CDN + f,
    });
    State.faceDetector.setOptions({
      model: 'short',
      minDetectionConfidence: CONFIG.MIN_DETECTION_CONF,
    });

    State.faceDetector.onResults(onDetectionResults);

    State.faceDetector.initialize()
      .then(() => {
        State.modelReady = true;
        log('Face detection model loaded', 'hl');
        clsAdd(DOM.dotModel, 'active');
        resolve();
      })
      .catch(err => {
        log(`Model load error: ${err && err.message ? err.message : err}`, 'er');
        reject(err);
      });
  });
}

/** Called by MediaPipe whenever a frame has been processed. */
function onDetectionResults(results) {
  State.inFlight       = false;   // release the in-flight guard
  State.lastDetections = results.detections || [];

  if (!State.running) return;

  if (State.detectionOn) {
    clsRemove(DOM.dotDetect, 'paused');
    clsAdd(DOM.dotDetect, 'active');
    accumulateStats(State.lastDetections);
    drawDetections(State.lastDetections);
  } else {
    // Detection is toggled off — still pump model, but hide results
    clearOverlay();
    DOM.faceCount.textContent = '—';
    DOM.confVal.textContent   = '—';
  }
}


function startDetection() {
  if (State.initializing || State.running) return;
  if (!checkSecureContext()) return;

  State.initializing = true;
  DOM.startBtn.disabled     = true;
  DOM.startBtn.textContent  = '[ INITIALIZING... ]';
  hideBanner();
  log('Checking camera permission...');
  setProgress(10);

  checkCameraPermission(permState => {
    if (permState === 'denied') {
      showBanner(
        '⚠ Camera access is blocked. Click the camera/lock icon in your address bar ' +
        'and set Camera to "Allow", then refresh.',
        true
      );
      log('Camera permission denied', 'er');
      resetStartButton();
      setProgress(0);
      return;
    }
    log('Requesting camera stream...');
    setProgress(20);
    requestCamera({ width: { ideal: CONFIG.IDEAL_WIDTH }, height: { ideal: CONFIG.IDEAL_HEIGHT }, facingMode: 'user' });
  });
}


function requestCamera(videoConstraints) {
  navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false })
    .then(onStreamAcquired)
    .catch(err => {
      if ((err.name === 'OverconstrainedError' || err.name === 'ConstraintNotSatisfiedError')
          && videoConstraints !== true) {
        log('Overconstrained — retrying with default constraints', 'wn');
        requestCamera(true);
        return;
      }
      const msg = classifyMediaError(err);
      showBanner(`⚠ ${msg}`, true);
      log(msg, 'er');
      resetStartButton();
      setProgress(0);
    });
}

/** Called once getUserMedia resolves with a live stream. */
function onStreamAcquired(stream) {
  State.activeStream = stream;

  // Detect physical camera disconnect (e.g. USB webcam unplugged)
  const tracks = stream.getVideoTracks();
  if (tracks.length > 0) {
    tracks[0].addEventListener('ended', () => {
      log('Camera track ended unexpectedly', 'er');
      showBanner('⚠ Camera disconnected. Please reconnect and reinitialize.', true);
      if (State.running) stopDetection();
    });
  }

  DOM.video.srcObject = stream;
  clsAdd(DOM.dotCam, 'active');
  log('Camera stream acquired', 'hl');
  setProgress(40);

  waitForVideoDimensions(CONFIG.STREAM_TIMEOUT_MS)
    .then(() => {
      const vw = DOM.video.videoWidth;
      const vh = DOM.video.videoHeight;
      DOM.resLabel.textContent = `${vw}×${vh}`;

      // Prime canvas size immediately
      DOM.overlay.width  = vw;
      DOM.overlay.height = vh;
      State.overlayW = vw;
      State.overlayH = vh;

      DOM.idleMsg.style.display = 'none';
      log(`Video ready at ${vw}×${vh}`);
      setProgress(55);
      log('Loading detection model...');
      return ensureModelReady();
    })
    .then(() => {
      setProgress(85);
      resetSessionState();
      resetStatsDom();

      // Create MediaPipe's camera-pump loop
      State.mpCamera = new Camera(DOM.video, {
        onFrame: onCameraFrame,
        width:   DOM.video.videoWidth,
        height:  DOM.video.videoHeight,
      });
      State.mpCamera.start();

      State.running     = true;
      State.initializing = false;
      startSessionTimer();

      DOM.startBtn.textContent = '[ STOP ]';
      DOM.startBtn.disabled    = false;
      DOM.startBtn.onclick     = stopDetection;
      DOM.toggleBtn.disabled   = false;
      DOM.snapBtn.disabled     = false;
      setProgress(100);
      log('Detection running', 'hl');
    })
    .catch(err => {
      log(`Startup error: ${err && err.message ? err.message : err}`, 'er');
      showBanner(`⚠ Failed to start: ${err && err.message ? err.message : err}`, true);
      resetStartButton();
      setProgress(0);
      releaseStream();
    });
}


function onCameraFrame() {
  if (document.hidden) return Promise.resolve();   // save CPU when tab is backgrounded
  if (State.inFlight)  return Promise.resolve();   // previous frame still processing

  State.inFlight = true;
  const t0 = perf();

  return State.faceDetector.send({ image: DOM.video })
    .then(() => {
      DOM.latencyVal.textContent = `${Math.round(perf() - t0)}ms`;

      State.frameCount++;
      const now     = perf();
      const elapsed = now - State.fpsWindowStart;
      if (elapsed >= 1000) {
        DOM.fpsVal.textContent = Math.round((State.frameCount / elapsed) * 1000);
        State.frameCount     = 0;
        State.fpsWindowStart = now;
      }
      // Throttled total-frame display
      if (State.frameCount % CONFIG.FRAME_COUNTER_EVERY === 0) {
        DOM.framesVal.textContent = State.detFrames;
      }
    })
    .catch(err => {
      State.inFlight = false;   // always release, even on error
      log(`Detection error: ${err.message}`, 'wn');
    });
}

/** Stop everything and return UI to its initial state. */
function stopDetection() {
  State.running      = false;
  State.initializing = false;
  State.inFlight     = false;

  if (State.mpCamera) {
    try { State.mpCamera.stop(); } catch (_) { /* ignore race on stop */ }
    State.mpCamera = null;
  }

  releaseStream();
  clearOverlay();
  State.overlayW = 0;
  State.overlayH = 0;
  stopSessionTimer();

  clsRemove(DOM.dotCam,    'active');
  clsRemove(DOM.dotDetect, 'active');
  clsRemove(DOM.dotDetect, 'paused');

  DOM.idleMsg.style.display = 'flex';
  DOM.idleText.innerHTML    = 'SESSION ENDED<br>PRESS INITIALIZE TO RESTART';
  DOM.faceCount.textContent = '0';
  DOM.fpsVal.textContent    = '—';
  DOM.latencyVal.textContent = '—';
  DOM.confVal.textContent   = '—';

  resetStartButton();
  DOM.toggleBtn.disabled    = true;
  DOM.toggleBtn.textContent = '[ PAUSE DETECT ]';
  clsRemove(DOM.toggleBtn, 'red');
  clsAdd(DOM.toggleBtn, 'amber');
  DOM.snapBtn.disabled      = true;

  // Restore detection-on state for the next session
  setDetectionOn(true, /* silent */ true);

  setProgress(0);
  DOM.resLabel.textContent = '—';
  log('Session ended');
}

/** Stop and discard the active media stream. */
function releaseStream() {
  if (State.activeStream) {
    State.activeStream.getTracks().forEach(t => t.stop());
    State.activeStream = null;
  }
  DOM.video.srcObject = null;
}


function waitForVideoDimensions(timeout) {
  return new Promise((resolve, reject) => {
    // Already ready
    if (DOM.video.readyState >= 1 && DOM.video.videoWidth > 0 && DOM.video.videoHeight > 0) {
      resolve(); return;
    }
    const timer = setTimeout(() => {
      DOM.video.removeEventListener('loadedmetadata', check);
      DOM.video.removeEventListener('canplay', check);
      reject(new Error('Timed out waiting for video dimensions'));
    }, timeout);

    function check() {
      if (DOM.video.videoWidth > 0 && DOM.video.videoHeight > 0) {
        clearTimeout(timer);
        DOM.video.removeEventListener('loadedmetadata', check);
        DOM.video.removeEventListener('canplay', check);
        resolve();
      }
    }
    DOM.video.addEventListener('loadedmetadata', check);
    DOM.video.addEventListener('canplay', check);
  });
}


function setDetectionOn(on, silent = false) {
  State.detectionOn = on;

  if (on) {
    DOM.toggleBtn.textContent = '[ PAUSE DETECT ]';
    clsRemove(DOM.toggleBtn, 'red');
    clsAdd(DOM.toggleBtn, 'amber');
    DOM.toggleBtn.setAttribute('aria-pressed', 'false');
    DOM.detBadge.textContent = 'DETECT: ON';
    clsRemove(DOM.detBadge, 'off');
    clsRemove(DOM.dotDetect, 'paused');
    if (!silent) log('Detection resumed', 'hl');
  } else {
    DOM.toggleBtn.textContent = '[ RESUME DETECT ]';
    clsRemove(DOM.toggleBtn, 'amber');
    clsAdd(DOM.toggleBtn, 'red');
    DOM.toggleBtn.setAttribute('aria-pressed', 'true');
    DOM.detBadge.textContent = 'DETECT: OFF';
    clsAdd(DOM.detBadge, 'off');
    clsRemove(DOM.dotDetect, 'active');
    clsAdd(DOM.dotDetect, 'paused');
    clearOverlay();
    DOM.faceCount.textContent = '—';
    DOM.confVal.textContent   = '—';
    if (!silent) log('Detection paused', 'wn');
  }
}

/** Update the progress bar and its ARIA value. */
function setProgress(pct) {
  DOM.loadFill.style.width = `${pct}%`;
  DOM.loadBar.setAttribute('aria-valuenow', pct);
}

/** Reset the start button to its initial "INITIALIZE" state. */
function resetStartButton() {
  State.initializing        = false;
  DOM.startBtn.disabled     = false;
  DOM.startBtn.textContent  = '[ INITIALIZE ]';
  DOM.startBtn.onclick      = startDetection;
}

/** Live UTC clock in the footer. */
function startClock() {
  const tick = () => {
    DOM.clock.textContent = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  };
  tick();
  setInterval(tick, 1000);
}

/** Keyboard shortcuts: Space = snapshot, D = toggle detection, Esc = close lightbox. */
function bindKeyboard() {
  document.addEventListener('keydown', e => {
    // Don't intercept keys when focus is inside a form element
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === 'BUTTON' || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    if (e.code === 'Space'  && State.running) { e.preventDefault(); takeSnapshot(); }
    if (e.code === 'KeyD'   && State.running) { e.preventDefault(); setDetectionOn(!State.detectionOn); }
    if (e.key  === 'Escape')                  { closeLightbox(); }
  });
}

/** Pause frame processing when the browser tab is hidden (saves CPU/battery). */
function bindVisibilityChange() {
  document.addEventListener('visibilitychange', () => {
    if (!State.running) return;
    if (document.hidden) {
      log('Tab hidden — processing paused', 'wn');
    } else {
      log('Tab visible — processing resumed');
      // Reset FPS window so stale counts don't skew the first displayed value
      State.fpsWindowStart = perf();
      State.frameCount     = 0;
    }
  });
}


function checkSecureContext() {
  const isSecure = typeof isSecureContext !== 'undefined'
    ? isSecureContext
    : (location.protocol === 'https:' ||
       location.hostname  === 'localhost' ||
       location.hostname  === '127.0.0.1');

  if (!isSecure) {
    showBanner(
      '⚠ INSECURE CONTEXT: Camera access requires HTTPS or localhost. ' +
      'Please serve this page over HTTPS.',
      true
    );
    DOM.startBtn.disabled = true;
    log('Insecure context — camera unavailable', 'er');
  }
  return isSecure;
}


function checkCameraPermission(callback) {
  if (!navigator.permissions || !navigator.permissions.query) {
    callback('prompt'); return;
  }
  navigator.permissions.query({ name: 'camera' })
    .then(status => {
      State.permStatus = status;
      status.onchange = () => {
        log(`Camera permission changed: ${status.state}`, 'wn');
        if (status.state === 'denied') {
          showBanner('⚠ Camera permission was revoked. Please refresh and allow access.', true);
          if (State.running) stopDetection();
        } else if (status.state === 'granted') {
          hideBanner();
        }
      };
      callback(status.state);
    })
    .catch(() => callback('prompt'));   // some browsers reject 'camera' as a query name
}

   
function classifyMediaError(err) {
  switch (err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Camera access was denied. Click the camera icon in your address bar and allow access, then try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera found on this device. Please connect a webcam and try again.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'Camera is already in use by another application. Please close it and try again.';
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return 'Requested camera resolution is not supported. Retrying with default settings…';
    case 'SecurityError':
      return 'Camera access blocked by browser security policy. Ensure the page is served over HTTPS.';
    case 'AbortError':
      return 'Camera initialization was aborted.';
    default:
      return `Camera error: ${err.message || err.name || 'Unknown'}`;
  }
}


function init() {
  // Persistent UI
  startClock();
  bindKeyboard();
  bindVisibilityChange();
  renderGallery();

  // Button listeners
  DOM.startBtn.onclick  = startDetection;
  DOM.toggleBtn.addEventListener('click', () => setDetectionOn(!State.detectionOn));
  DOM.snapBtn.addEventListener('click', takeSnapshot);

  // Gallery scroll controls
  DOM.galleryScrollUp.addEventListener('click',   () => scrollGalleryBy(-1));
  DOM.galleryScrollDown.addEventListener('click', () => scrollGalleryBy(1));
  DOM.gallery.addEventListener('scroll', updateGalleryScroll, { passive: true });

  // Lightbox listeners
  DOM.lbClose.addEventListener('click', closeLightbox);
  DOM.lightbox.addEventListener('click', e => { if (e.target === DOM.lightbox) closeLightbox(); });

  // Preflight: secure context, getUserMedia availability, permission state
  if (!checkSecureContext()) return;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showBanner(
      '⚠ Your browser does not support camera access. ' +
      'Please use Chrome, Firefox, Edge, or Safari 14+.',
      true
    );
    DOM.startBtn.disabled = true;
    log('getUserMedia not supported', 'er');
    return;
  }

  // Soft permission check — informational only, no action taken here
  checkCameraPermission(state => {
    if (state === 'denied') {
      showBanner(
        '⚠ Camera permission is currently blocked. Click the camera icon in your address bar to allow access.',
        true
      );
      log('Camera permission pre-check: denied', 'wn');
    } else {
      log(`Camera permission pre-check: ${state}`);
    }
  });

  log('System ready — awaiting initialization.');
  log('Keys: SPACE = snapshot  |  D = toggle detect', 'wn');
}


if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

})(); 
