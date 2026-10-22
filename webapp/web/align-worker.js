/* How well the live camera lines up with the Before photo, for the webcam panel (app.js).

   OpenCV.js (vendor/opencv-4.12.0.js, Apache-2.0) runs here, in a worker, so the camera
   preview stays smooth. Both images arrive small (about 320 px wide) and grey-scaled here:
   ORB features in each, Hamming matches with Lowe's ratio test, then a RANSAC affine fit
   from the Before photo to the live frame. The fit says where the Before frame's corners
   land in the live frame; how far they are from the live frame's own corners, as a share
   of its diagonal, is the misalignment `d` (0 = the same view).

   Messages in:  {type: "ref", width, height, data}  the Before photo (RGBA bytes)
                 {type: "frame", id, width, height, data}  a live frame
   Messages out: {type: "ready"} | {type: "error", message}
                 {type: "ref", ok, features}
                 {type: "frame", id, ok, reason?, matches, inliers, d, quad, dx, dy, rotation, scale, ms}
                   quad: the Before frame's corners in the live frame, x0 y0 x1 y1 ... as fractions
                   dx, dy: where the Before frame's centre is, from the live frame's centre, as fractions
*/
/* global cv */
"use strict";

const FEATURES = 700;          // ORB keypoints per image
const RATIO = 0.75;            // Lowe's ratio test
const MIN_MATCHES = 12;        // fewer good matches: the Before scene is not in view
const RANSAC_PX = 3.0;         // RANSAC reprojection threshold, in pixels of the small image

let ready = false, orb = null, matcher = null, ref = null;

try {
  importScripts("vendor/opencv-4.12.0.js");
} catch (e) {
  postMessage({ type: "error", message: "OpenCV.js did not load: " + e });
}

// The runtime starts on its own; `cv` is a thenable that must not be awaited (it resolves
// to itself), so wait until its functions are there.
(function waitForRuntime(tries) {
  if (self.cv && self.cv.Mat && self.cv.ORB) {
    orb = new cv.ORB(FEATURES);
    matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
    ready = true;
    postMessage({ type: "ready" });
  } else if (tries > 0) {
    setTimeout(() => waitForRuntime(tries - 1), 50);
  } else {
    postMessage({ type: "error", message: "OpenCV.js did not start" });
  }
})(400);

function features(msg) {
  const rgba = cv.matFromImageData({ data: msg.data, width: msg.width, height: msg.height });
  const gray = new cv.Mat();
  cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
  rgba.delete();
  const kp = new cv.KeyPointVector(), des = new cv.Mat(), none = new cv.Mat();
  orb.detectAndCompute(gray, none, kp, des);
  gray.delete(); none.delete();
  return { kp, des, width: msg.width, height: msg.height };
}

function freeFeatures(f) {
  if (f) { f.kp.delete(); f.des.delete(); }
}

function align(msg) {
  const t0 = performance.now();
  const out = { type: "frame", id: msg.id, ok: false, matches: 0, inliers: 0 };
  if (!ref || ref.kp.size() < MIN_MATCHES) { out.reason = "reference"; return out; }
  const cur = features(msg);
  const knn = new cv.DMatchVectorVector();
  const from = [], to = [];
  try {
    if (cur.kp.size() < MIN_MATCHES || cur.des.rows === 0) { out.reason = "texture"; return out; }
    matcher.knnMatch(ref.des, cur.des, knn, 2);
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() < 2) continue;
      const a = pair.get(0), b = pair.get(1);
      if (a.distance < RATIO * b.distance) {
        const p = ref.kp.get(a.queryIdx).pt, q = cur.kp.get(a.trainIdx).pt;
        from.push(p.x, p.y); to.push(q.x, q.y);
      }
    }
    out.matches = from.length / 2;
    if (out.matches < MIN_MATCHES) { out.reason = "matches"; return out; }

    const src = cv.matFromArray(out.matches, 1, cv.CV_32FC2, from);
    const dst = cv.matFromArray(out.matches, 1, cv.CV_32FC2, to);
    const inl = new cv.Mat();
    const M = cv.estimateAffine2D(src, dst, inl, cv.RANSAC, RANSAC_PX, 2000, 0.99, 10);
    src.delete(); dst.delete();
    if (M.empty()) { inl.delete(); M.delete(); out.reason = "fit"; return out; }
    for (let i = 0; i < inl.rows; i++) out.inliers += inl.data[i] ? 1 : 0;
    inl.delete();
    // read the 2x3 fit element by element: its depth (64F or 32F) and layout are not fixed across builds
    const at = (r, k) => (M.type() === cv.CV_64F ? M.doubleAt(r, k) : M.floatAt(r, k));
    const a = at(0, 0), b = at(0, 1), tx = at(0, 2), c = at(1, 0), d = at(1, 1), ty = at(1, 2);
    out.fit = [M.rows, M.cols, M.type()];
    M.delete();
    if (out.inliers < MIN_MATCHES * 0.75) { out.reason = "matches"; return out; }

    // the Before frame's corners and centre, carried into the live frame
    const W = ref.width, H = ref.height, w = cur.width, h = cur.height;
    const map = (x, y) => [(a * x + b * y + tx) / w, (c * x + d * y + ty) / h];
    const corners = [[0, 0], [W, 0], [W, H], [0, H]];
    const quad = [];
    let dist = 0;
    corners.forEach(([x, y]) => {
      const [u, v] = map(x, y);
      quad.push(u, v);
      dist += Math.hypot((u - x / W) * w, (v - y / H) * h);
    });
    const [cx, cy] = map(W / 2, H / 2);
    out.ok = true;
    out.quad = quad;
    out.d = dist / 4 / Math.hypot(w, h);
    out.dx = cx - 0.5;
    out.dy = cy - 0.5;
    out.scale = Math.sqrt(Math.abs(a * d - b * c));
    out.rotation = Math.atan2(c - b, a + d) * 180 / Math.PI;
    return out;
  } finally {
    knn.delete();
    freeFeatures(cur);
    out.ms = Math.round(performance.now() - t0);
  }
}

onmessage = (e) => {
  const msg = e.data;
  if (!ready) { postMessage({ type: msg.type, id: msg.id, ok: false, reason: "starting" }); return; }
  try {
    if (msg.type === "ref") {
      freeFeatures(ref);
      ref = features(msg);
      postMessage({ type: "ref", ok: ref.kp.size() >= MIN_MATCHES, features: ref.kp.size() });
    } else if (msg.type === "frame") {
      postMessage(align(msg));
    } else if (msg.type === "clear") {
      freeFeatures(ref); ref = null;
    }
  } catch (err) {
    postMessage({ type: msg.type, id: msg.id, ok: false, reason: "error", message: String(err && err.message || err) });
  }
};
