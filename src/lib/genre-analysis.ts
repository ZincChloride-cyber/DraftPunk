/**
 * genre-analysis.ts
 * ─────────────────
 * Calls the FastAPI backend to classify the genre of an uploaded audio file.
 * Falls back to a lightweight client-side heuristic if the backend is unreachable.
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface MelSpectrogramData {
  values: number[][];
  n_mels: number;
  n_frames: number;
  duration: number;
  sample_rate: number;
  hop_length: number;
  fmax_hz: number;
  db_min: number;
  db_max: number;
}

export interface GenreResult {
  genre: string;
  confidence: number;
  scores: { genre: string; score: number }[];
  features: {
    tempo: number;
    spectralCentroid: number;
    spectralRolloff: number;
    spectralBandwidth: number;
    energy: number;
    zeroCrossingRate: number;
  };
  melSpectrogram?: MelSpectrogramData;
  waveform?: number[];
}

// ── Configuration ────────────────────────────────────────────────────────────

// In dev, Vite proxies /api → http://127.0.0.1:8000 (see vite.config.ts)
const API_BASE_URL = import.meta.env.DEV ? "" : "http://localhost:8000";

// ── Main analysis function ───────────────────────────────────────────────────

/**
 * Analyze an audio file by sending it to the ML backend.
 * Requires the FastAPI server to be running with trained models.
 */
export async function analyzeAudio(file: File): Promise<GenreResult> {
  return analyzeWithBackend(file);
}

// ── Backend API call ─────────────────────────────────────────────────────────

async function analyzeWithBackend(file: File): Promise<GenreResult> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(
      "ML backend is not reachable. Start it with: cd backend && python -m uvicorn main:app --reload --port 8000",
    );
  }

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`API error (${response.status}): ${detail}`);
  }

  const data = await response.json();

  // Map snake_case API response → camelCase frontend types
  return {
    genre: data.genre,
    confidence: data.confidence,
    scores: data.scores,
    features: {
      tempo: data.features.tempo,
      spectralCentroid: data.features.spectral_centroid,
      spectralRolloff: data.features.spectral_rolloff,
      spectralBandwidth: data.features.spectral_bandwidth,
      energy: data.features.energy,
      zeroCrossingRate: data.features.zero_crossing_rate,
    },
    melSpectrogram: data.mel_spectrogram
      ? {
          values: data.mel_spectrogram.values,
          n_mels: data.mel_spectrogram.n_mels,
          n_frames: data.mel_spectrogram.n_frames,
          duration: data.mel_spectrogram.duration,
          sample_rate: data.mel_spectrogram.sample_rate,
          hop_length: data.mel_spectrogram.hop_length,
          fmax_hz: data.mel_spectrogram.fmax_hz,
          db_min: data.mel_spectrogram.db_min,
          db_max: data.mel_spectrogram.db_max,
        }
      : undefined,
    waveform: data.waveform,
  };
}

// ── Backend health check ─────────────────────────────────────────────────────

export async function checkBackendHealth(): Promise<{
  status: string;
  modelLoaded: boolean;
  modelAvailable: boolean;
}> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) throw new Error("Backend health check failed");
  const data = await response.json();
  return {
    status: data.status,
    modelLoaded: data.model_loaded,
    modelAvailable: data.model_available,
  };
}

// ═════════════════════════════════════════════════════════════════════════════
// FALLBACK: Client-side heuristic analysis (used when backend is down)
// ═════════════════════════════════════════════════════════════════════════════

const GENRE_PROFILES: Record<
  string,
  { centroid: number; tempo: number; energy: number; zcr: number }
> = {
  Rock: { centroid: 2800, tempo: 120, energy: 0.18, zcr: 0.08 },
  Pop: { centroid: 3200, tempo: 118, energy: 0.15, zcr: 0.07 },
  Jazz: { centroid: 2200, tempo: 100, energy: 0.09, zcr: 0.05 },
  Classical: { centroid: 1800, tempo: 90, energy: 0.06, zcr: 0.03 },
  Hiphop: { centroid: 2400, tempo: 95, energy: 0.16, zcr: 0.06 },
  Electronic: { centroid: 3600, tempo: 128, energy: 0.2, zcr: 0.09 },
  Metal: { centroid: 3400, tempo: 140, energy: 0.25, zcr: 0.12 },
  Blues: { centroid: 2000, tempo: 90, energy: 0.11, zcr: 0.05 },
  Country: { centroid: 2600, tempo: 110, energy: 0.12, zcr: 0.06 },
  Reggae: { centroid: 2100, tempo: 85, energy: 0.10, zcr: 0.04 },
  Disco: { centroid: 3000, tempo: 120, energy: 0.17, zcr: 0.07 },
};

function analyzeClientSide(buffer: AudioBuffer): GenreResult {
  const data = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;

  // Energy
  let energySum = 0;
  let zcr = 0;
  for (let i = 0; i < data.length; i++) {
    energySum += data[i] * data[i];
    if (i > 0 && data[i] >= 0 !== data[i - 1] >= 0) zcr++;
  }
  const energy = Math.sqrt(energySum / data.length);
  const zeroCrossingRate = zcr / data.length;

  // FFT-based spectral features
  const fftSize = 2048;
  const start = Math.max(0, Math.floor(data.length / 2) - fftSize / 2);
  const window = data.slice(start, start + fftSize);
  const spectrum = computeMagnitudeSpectrum(window);
  const spectralCentroid = computeCentroid(spectrum, sampleRate);
  const spectralRolloff = computeRolloff(spectrum, sampleRate);

  // Tempo
  const tempo = estimateTempo(data, sampleRate);

  const features = {
    tempo,
    spectralCentroid,
    spectralRolloff,
    spectralBandwidth: 0,
    energy,
    zeroCrossingRate,
  };

  // Score each genre
  const scores = Object.entries(GENRE_PROFILES)
    .map(([genre, p]) => {
      const dCentroid = Math.abs(spectralCentroid - p.centroid) / 2000;
      const dTempo = Math.abs(tempo - p.tempo) / 60;
      const dEnergy = Math.abs(energy - p.energy) / 0.2;
      const dZcr = Math.abs(zeroCrossingRate - p.zcr) / 0.1;
      const distance =
        dCentroid * 0.4 + dTempo * 0.25 + dEnergy * 0.2 + dZcr * 0.15;
      const score = Math.max(0, 1 - distance);
      return { genre, score };
    })
    .sort((a, b) => b.score - a.score);

  // Softmax-ish normalization
  const expSum = scores.reduce((s, x) => s + Math.exp(x.score * 4), 0);
  const normalized = scores.map((x) => ({
    genre: x.genre,
    score: Math.exp(x.score * 4) / expSum,
  }));

  return {
    genre: normalized[0].genre,
    confidence: normalized[0].score,
    scores: normalized,
    features,
  };
}

// ── Helpers for client-side fallback ─────────────────────────────────────────

function computeMagnitudeSpectrum(signal: Float32Array): Float32Array {
  const N = signal.length;
  const bins = 512;
  const out = new Float32Array(bins);
  for (let k = 0; k < bins; k++) {
    let re = 0,
      im = 0;
    const angleStep = (-2 * Math.PI * k) / N;
    for (let n = 0; n < N; n++) {
      const a = angleStep * n;
      re += signal[n] * Math.cos(a);
      im += signal[n] * Math.sin(a);
    }
    out[k] = Math.sqrt(re * re + im * im);
  }
  return out;
}

function computeCentroid(
  spectrum: Float32Array,
  sampleRate: number,
): number {
  let num = 0,
    den = 0;
  const binHz = sampleRate / (spectrum.length * 2);
  for (let i = 0; i < spectrum.length; i++) {
    num += i * binHz * spectrum[i];
    den += spectrum[i];
  }
  return den > 0 ? num / den : 0;
}

function computeRolloff(
  spectrum: Float32Array,
  sampleRate: number,
): number {
  const total = spectrum.reduce((s, v) => s + v, 0);
  let cum = 0;
  const binHz = sampleRate / (spectrum.length * 2);
  for (let i = 0; i < spectrum.length; i++) {
    cum += spectrum[i];
    if (cum >= total * 0.85) return i * binHz;
  }
  return 0;
}

function estimateTempo(data: Float32Array, sampleRate: number): number {
  const hop = Math.floor(sampleRate / 100);
  const env: number[] = [];
  for (let i = 0; i < data.length; i += hop) {
    let sum = 0;
    for (let j = 0; j < hop && i + j < data.length; j++) {
      sum += data[i + j] * data[i + j];
    }
    env.push(Math.sqrt(sum / hop));
  }
  let bestLag = 60,
    bestVal = -Infinity;
  for (let lag = 33; lag <= 100; lag++) {
    let sum = 0;
    for (let i = 0; i < env.length - lag; i++) sum += env[i] * env[i + lag];
    if (sum > bestVal) {
      bestVal = sum;
      bestLag = lag;
    }
  }
  return Math.round(6000 / bestLag);
}