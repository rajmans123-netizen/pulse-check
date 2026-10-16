// Pulse Check engine: fingertip photoplethysmography (PPG) from a phone camera.
// Each heartbeat pushes blood into the fingertip, so the lit finger gets a tiny bit
// darker once per beat. We track the average colour, clean it up and find the rhythm.
(function (root) {
  const FS = 30;                         // signal is resampled to 30 samples a second
  const BPM_MIN = 40, BPM_MAX = 200;

  // timestamps arrive unevenly (dropped frames), so put samples on a fixed 30 Hz grid
  function resample(ts, vs, tEnd, dur) {
    const n = Math.floor(dur * FS), out = new Float64Array(n), t0 = tEnd - dur;
    let j = 0;
    for (let i = 0; i < n; i++) {
      const t = t0 + i / FS;
      while (j < ts.length - 2 && ts[j + 1] < t) j++;
      const a = ts[j], b = ts[j + 1];
      const f = b > a ? Math.min(1, Math.max(0, (t - a) / (b - a))) : 0;
      out[i] = vs[j] + (vs[j + 1] - vs[j]) * f;
    }
    return out;
  }

  function movAvg(x, w) {               // centred moving average, shrinks at the edges
    const n = x.length, p = new Float64Array(n + 1), out = new Float64Array(n), h = w >> 1;
    for (let i = 0; i < n; i++) p[i + 1] = p[i] + x[i];
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - h), b = Math.min(n, i + h + 1);
      out[i] = (p[b] - p[a]) / (b - a);
    }
    return out;
  }

  // remove slow drift (pressure, auto-exposure), smooth sensor noise, scale to unit variance
  function clean(x) {
    const base = movAvg(x, Math.round(1.5 * FS));
    const d = x.map((v, i) => v - base[i]);
    const s = movAvg(d, 5);
    let m = 0; for (const v of s) m += v; m /= s.length;
    let sd = 0; for (const v of s) sd += (v - m) * (v - m);
    sd = Math.sqrt(sd / s.length) || 1;
    return s.map(v => (v - m) / sd);
  }

  // rhythm = the delay at which the wave best lines up with itself (autocorrelation)
  function rhythm(x) {
    const n = x.length;
    const lo = Math.floor(FS * 60 / BPM_MAX), hi = Math.ceil(FS * 60 / BPM_MIN);
    if (n < hi * 2) return null;
    const ac = new Float64Array(hi + 2);
    for (let L = lo - 1; L <= hi + 1; L++) {
      let s = 0;
      for (let i = 0; i + L < n; i++) s += x[i] * x[i + L];
      ac[L] = s / (n - L);
    }
    let top = -1;
    for (let L = lo; L <= hi; L++) if (ac[L] > top) top = ac[L];
    if (top <= 0) return null;
    // first strong peak, so two beats (2x the delay) never win over one
    for (let L = lo; L <= hi; L++) {
      if (ac[L] >= ac[L - 1] && ac[L] >= ac[L + 1] && ac[L] >= 0.85 * top) {
        const y0 = ac[L - 1], y1 = ac[L], y2 = ac[L + 1], den = y0 - 2 * y1 + y2;
        const lag = (L + (den ? 0.5 * (y0 - y2) / den : 0)) / FS;
        return { bpm: 60 / lag, quality: y1 };
      }
    }
    return null;
  }

  // beats = the dips (more blood = darker), found on the flipped wave
  function beats(wave, bpm) {
    const gap = Math.max(5, Math.floor(0.6 * 60 / bpm * FS)), out = [];
    for (let i = 2; i < wave.length - 2; i++) {
      const v = wave[i];
      if (v > 0.3 && v >= wave[i - 1] && v >= wave[i + 1] && v >= wave[i - 2] && v >= wave[i + 2]) {
        const last = out[out.length - 1];
        if (out.length && i - last < gap) { if (v > wave[last]) out[out.length - 1] = i; }
        else out.push(i);
      }
    }
    return out;
  }

  function Analyzer() { this.reset(); }
  Analyzer.prototype.reset = function () { this.t = []; this.r = []; this.g = []; };
  Analyzer.prototype.push = function (t, r, g) {
    this.t.push(t); this.r.push(r); this.g.push(g);
    while (this.t.length && t - this.t[0] > 12) { this.t.shift(); this.r.shift(); this.g.shift(); }
  };
  Analyzer.prototype.span = function () { const T = this.t; return T.length ? T[T.length - 1] - T[0] : 0; };
  Analyzer.prototype.analyze = function (win) {
    const T = this.t, span = this.span();
    if (T.length < 30 || span < 4) return null;
    const dur = Math.min(win || 8, span), tEnd = T[T.length - 1];
    let best = null;
    for (const [name, v] of [['red', this.r], ['green', this.g]]) {
      let mean = 0; for (const x of v) mean += x; mean /= v.length;
      if (mean > 252 || mean < 3) continue;          // clipped or black: no signal there
      const x = clean(resample(T, v, tEnd, dur)), e = rhythm(x);
      if (e && (!best || e.quality > best.quality)) best = { channel: name, x, ...e };
    }
    if (!best) return null;
    const wave = best.x.map(v => -v);
    return { bpm: best.bpm, quality: best.quality, channel: best.channel, wave,
             beats: beats(wave, best.bpm), tEnd, fs: FS };
  };

  // synthetic fingertip signal for demos and tests: dips per beat, breathing drift, noise
  function synth(bpmAt, opts) {
    const o = Object.assign({ noise: 0.6, drift: 6, amp: 3, level: 190, seed: 1 }, opts || {});
    let s = o.seed;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    let phase = 0, lastT = 0;
    return function (t) {
      const dt = t - lastT; lastT = t;
      phase += dt * bpmAt(t) / 60 * (1 + (rnd() - 0.5) * 0.04);
      const p = phase % 1;
      const pulse = Math.exp(-((p - 0.18) ** 2) / 0.006) + 0.35 * Math.exp(-((p - 0.5) ** 2) / 0.01);
      const drift = o.drift * Math.sin(2 * Math.PI * 0.22 * t) + o.drift * 0.5 * Math.sin(2 * Math.PI * 0.05 * t);
      const base = o.level + drift + (rnd() - 0.5) * 2 * o.noise;
      return { r: Math.min(255, base - o.amp * pulse), g: Math.min(255, base * 0.35 - o.amp * 1.4 * pulse) };
    };
  }

  const API = { FS, Analyzer, resample, clean, rhythm, beats, synth };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.Pulse = API;
})(typeof self !== 'undefined' ? self : this);
