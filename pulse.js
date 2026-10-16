// Pulse Check engine: fingertip photoplethysmography (PPG) from a phone camera.
// Each heartbeat pushes blood into the fingertip, so the lit finger gets a tiny bit
// darker once per beat. We track the average colour, clean it up and find the rhythm.
//
// v2 (tuned on a real Pixel 10 recording): the raw signal is dominated by step jumps
// (camera re-exposing, finger pressure) and a fast sensor flicker, with the pulse a small
// wave underneath. So: remove steps, band-pass to heart-rate frequencies (zero-phase),
// then read the rhythm from the spectrum, checked against autocorrelation.
(function (root) {
  const FS = 30;                         // signal is resampled to 30 samples a second
  const BPM_MIN = 40, BPM_MAX = 200;
  const F_LO = 0.55, F_HI = 3.5;         // band-pass, Hz (33 to 210 BPM)

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

  function median(a) { const s = Array.from(a).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; }

  // steps and spikes: any sample-to-sample jump far bigger than normal is zeroed out
  function removeSteps(x) {
    const n = x.length, d = new Float64Array(n);
    for (let i = 1; i < n; i++) d[i] = x[i] - x[i - 1];
    const md = median(d.subarray(1)), mad = median(Array.from(d.subarray(1), v => Math.abs(v - md))) * 1.4826 || 1e-6;
    const out = new Float64Array(n);
    for (let i = 1; i < n; i++) out[i] = out[i - 1] + (Math.abs(d[i] - md) > 4 * mad ? 0 : d[i]);
    return out;
  }

  // 2nd-order Butterworth sections (RBJ biquads), run forward then backward (zero phase)
  function biquad(type, f0) {
    const w = 2 * Math.PI * f0 / FS, c = Math.cos(w), al = Math.sin(w) / (2 * Math.SQRT1_2);
    const b = type === 'lp' ? [(1 - c) / 2, 1 - c, (1 - c) / 2] : [(1 + c) / 2, -(1 + c), (1 + c) / 2];
    const a0 = 1 + al;
    return { b: b.map(v => v / a0), a: [-2 * c / a0, (1 - al) / a0] };
  }
  function runBiquad(q, x) {
    const y = new Float64Array(x.length);
    let x1 = x[0], x2 = x[0], y1 = 0, y2 = 0;
    if (q.b[0] + q.b[1] + q.b[2] > 0.5) { y1 = y2 = x[0]; }      // low-pass starts settled at the first value
    for (let i = 0; i < x.length; i++) {
      const v = q.b[0] * x[i] + q.b[1] * x1 + q.b[2] * x2 - q.a[0] * y1 - q.a[1] * y2;
      x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
    }
    return y;
  }
  const HP = biquad('hp', F_LO), LP = biquad('lp', F_HI);
  function bandpass(x) {
    let y = x;
    for (const q of [HP, HP, LP, LP]) y = runBiquad(q, y);           // forward
    y = y.reverse();
    for (const q of [HP, HP, LP, LP]) y = runBiquad(q, y);           // backward
    return y.reverse();
  }

  function clean(x) {
    let m = 0; for (const v of x) m += v; m /= x.length;
    const y = bandpass(removeSteps(x.map(v => v - m)));
    const edge = Math.round(0.6 * FS);                                // filter warm-up at both ends
    const core = y.subarray(edge, y.length - edge);
    let s = 0; for (const v of core) s += v * v;
    const sd = Math.sqrt(s / core.length) || 1;
    const out = core.map(v => v / sd);
    out.rms = sd;                                                     // pulse-band strength, in brightness units
    return out;
  }

  // spectrum peak in the heart-rate band, refined by interpolation; quality = peak vs typical level
  function spectrumPeak(x) {
    const n = x.length, N = 2048, re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < n; i++) re[i] = x[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
    fft(re, im);
    const kLo = Math.ceil(BPM_MIN / 60 * N / FS), kHi = Math.floor(BPM_MAX / 60 * N / FS), mag = [];
    for (let k = kLo; k <= kHi; k++) mag.push(Math.hypot(re[k], im[k]));
    let bi = 0; for (let i = 1; i < mag.length; i++) if (mag[i] > mag[bi]) bi = i;
    const y0 = mag[bi - 1] ?? mag[bi], y1 = mag[bi], y2 = mag[bi + 1] ?? mag[bi], den = y0 - 2 * y1 + y2;
    const k = kLo + bi + (den ? 0.5 * (y0 - y2) / den : 0);
    return { bpm: k * FS / N * 60, snr: y1 / (median(mag) || 1e-9), atEdge: bi === 0 || bi === mag.length - 1 };
  }
  function fft(re, im) {                                             // in-place radix-2
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = i + j, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr; } } }
  }

  // autocorrelation at a given period: how well the wave lines up with itself one beat later
  function acAt(x, bpm) {
    const L = Math.round(FS * 60 / bpm); let s = 0, s0 = 0;
    for (let i = 0; i + L < x.length; i++) s += x[i] * x[i + L];
    for (let i = 0; i < x.length; i++) s0 += x[i] * x[i];
    return s / (x.length - L) / (s0 / x.length);
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
    while (this.t.length && t - this.t[0] > 14) { this.t.shift(); this.r.shift(); this.g.shift(); }
  };
  Analyzer.prototype.span = function () { const T = this.t; return T.length ? T[T.length - 1] - T[0] : 0; };
  Analyzer.prototype.analyze = function (win) {
    const T = this.t, span = this.span();
    if (T.length < 60 || span < 5) return null;
    const dur = Math.min(win || 10, span), tEnd = T[T.length - 1];
    let best = null;
    for (const [name, v] of [['red', this.r], ['green', this.g]]) {
      let mean = 0; for (const x of v) mean += x; mean /= v.length;
      if (mean > 252 || mean < 3) continue;          // clipped or black: no signal there
      const x = clean(resample(T, v, tEnd, dur)), p = spectrumPeak(x), pi = 100 * x.rms / mean;   // % of brightness
      if (p.atEdge) continue;
      const ac = acAt(x, p.bpm);
      const quality = Math.max(0, Math.min(1, (p.snr - 2) / 6)) * Math.max(0, ac);   // both must agree
      if (!best || quality > best.quality) best = { channel: name, x, bpm: p.bpm, snr: p.snr, ac, quality, pi };
    }
    if (!best) return null;
    const wave = best.x.map(v => -v);
    return { bpm: best.bpm, quality: best.quality, snr: best.snr, ac: best.ac, pi: best.pi, channel: best.channel, wave,
             beats: beats(wave, best.bpm), tEnd: tEnd - 0.6, fs: FS };
  };

  // synthetic fingertip signal for demos and tests: dips per beat, drift, steps, flicker, noise
  function synth(bpmAt, opts) {
    const o = Object.assign({ noise: 0.6, drift: 6, amp: 3, level: 190, seed: 1, steps: 0, flicker: 0 }, opts || {});
    let s = o.seed;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    let phase = 0, lastT = 0, offset = 0, nextStep = 2 + rnd() * 4, frame = 0;
    return function (t) {
      const dt = t - lastT; lastT = t; frame++;
      phase += dt * bpmAt(t) / 60 * (1 + (rnd() - 0.5) * 0.04);
      if (o.steps && t > nextStep) { offset += (rnd() - 0.5) * 2 * o.steps; nextStep = t + 1.5 + rnd() * 4; }
      const p = phase % 1;
      const pulse = Math.exp(-((p - 0.18) ** 2) / 0.006) + 0.35 * Math.exp(-((p - 0.5) ** 2) / 0.01);
      const drift = o.drift * Math.sin(2 * Math.PI * 0.22 * t) + o.drift * 0.5 * Math.sin(2 * Math.PI * 0.05 * t);
      const flick = o.flicker * (frame % 2 ? 1 : -1);
      const base = o.level + drift + offset + flick + (rnd() - 0.5) * 2 * o.noise;
      return { r: Math.min(255, base - o.amp * pulse), g: Math.min(255, base * 0.68 - o.amp * 1.4 * pulse) };
    };
  }

  const API = { FS, Analyzer, resample, clean, spectrumPeak, beats, synth };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.Pulse = API;
})(typeof self !== 'undefined' ? self : this);
