import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
// @ts-ignore
import CLOUDS from "vanta/dist/vanta.clouds.min";
import { Upload, Music, Play, Pause, RotateCcw, AudioWaveform } from "lucide-react";
import { Waveform } from "@/components/Waveform";
import { AudioVisualization } from "@/components/AudioVisualization";
import { analyzeAudio, checkBackendHealth, type GenreResult } from "@/lib/genre-analysis";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  const [file, setFile] = useState<File | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<GenreResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const vantaRef = useRef<HTMLDivElement>(null);
  const [vantaEffect, setVantaEffect] = useState<any>(null);

  // Check if the ML backend is online (retry every 5s until connected)
  useEffect(() => {
    let cancelled = false;

    const check = () => {
      checkBackendHealth()
        .then(() => {
          if (!cancelled) setBackendOnline(true);
        })
        .catch(() => {
          if (!cancelled) setBackendOnline(false);
        });
    };

    check();
    const interval = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!vantaEffect && vantaRef.current) {
      setVantaEffect(
        CLOUDS({
          el: vantaRef.current,
          THREE,
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          speed: 1.60,
          backgroundColor: 0xffffff,
          skyColor: 0x68b8d7,
          cloudColor: 0xadc1de,
          cloudShadowColor: 0x183550,
          sunColor: 0xff9919,
          sunGlareColor: 0xff6633,
          sunlightColor: 0xff9933,
        })
      );
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy();
    };
  }, [vantaEffect]);

  const handleFile = useCallback(async (f: File) => {
    if (!/audio\/(mpeg|mp3|wav|wave|x-wav)/.test(f.type) && !/\.(mp3|wav)$/i.test(f.name)) {
      setError("Please upload an MP3 or WAV file.");
      return;
    }
    setError(null);
    setFile(f);
    setResult(null);
    setProgress(0);
    const url = URL.createObjectURL(f);
    setAudioUrl(url);
    setAnalyzing(true);
    try {
      const arrayBuffer = await f.arrayBuffer();
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const buffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
      setAudioBuffer(buffer);
      // Send file to ML backend (falls back to client-side heuristic)
      setTimeout(async () => {
        try {
          const r = await analyzeAudio(f);
          setResult(r);
        } catch (e) {
          const message =
            e instanceof Error
              ? e.message
              : "Analysis failed. Start the backend: cd backend && python -m uvicorn main:app --reload --port 8000";
          setError(message);
          setResult(null);
        } finally {
          setAnalyzing(false);
        }
      }, 50);
    } catch (e) {
      console.error(e);
      setError("Could not decode audio file.");
      setAnalyzing(false);
    }
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTime = () => setProgress(audio.currentTime / (audio.duration || 1));
    const onEnd = () => setIsPlaying(false);
    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("ended", onEnd);
    return () => {
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("ended", onEnd);
    };
  }, [audioUrl]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) audio.pause();
    else audio.play();
    setIsPlaying(!isPlaying);
  };

  const seek = (ratio: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = ratio * (audio.duration || 0);
    setProgress(ratio);
  };

  const reset = () => {
    setFile(null);
    setAudioBuffer(null);
    setResult(null);
    setIsPlaying(false);
    setProgress(0);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
  };

  return (
    <div ref={vantaRef} className="min-h-screen relative overflow-hidden">
      <div className="relative z-10">
        <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-foreground">
            <AudioWaveform className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight">Draft Punk</h1>
            <div className="flex items-center gap-2">
              <p className="text-xs text-muted-foreground">AI Music Genre Analyzer</p>
              {backendOnline !== null && (
                <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                  backendOnline
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-amber-100 text-amber-700"
                }`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${backendOnline ? "bg-emerald-500" : "bg-amber-500"}`} />
                  {backendOnline ? "ML Model" : "Offline"}
                </span>
              )}
            </div>
          </div>
        </div>
        {file && (
          <button
            onClick={reset}
            className="inline-flex items-center gap-2 rounded-full border border-white/40 bg-white/30 backdrop-blur-md px-4 py-2 text-sm font-medium text-foreground transition hover:bg-white/50 shadow-sm"
          >
            <RotateCcw className="h-4 w-4" /> New file
          </button>
        )}
      </header>

      <main className="mx-auto max-w-6xl px-6 pb-20">
        {!file ? (
          <section className="flex flex-col items-center pt-16 text-center animate-fade-in">
            <h2 className="max-w-2xl text-5xl font-semibold tracking-tight text-foreground sm:text-6xl animate-fade-in">
              Discover the genre of any track
            </h2>
            <p className="mt-4 max-w-xl text-base text-muted-foreground animate-fade-in">
              Upload an MP3 or WAV file and instantly get a predicted genre, confidence score, and waveform visualization.
            </p>

            <label
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              className={`group mt-12 flex w-full max-w-2xl cursor-pointer flex-col items-center justify-center rounded-3xl bg-white/30 backdrop-blur-md border border-white/40 px-8 py-14 transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl animate-scale-in ${
                dragOver ? "ring-2 ring-primary bg-white/50 scale-[1.02]" : "shadow-xl"
              }`}
            >
              <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-white/50 border border-white/60 shadow-sm transition-transform duration-300 group-hover:scale-110 group-hover:-translate-y-0.5">
                <Upload className="h-7 w-7 text-foreground transition-transform duration-300 group-hover:-translate-y-0.5" />
              </div>
              <p className="text-lg font-medium text-foreground">Upload your music file</p>
              <p className="mt-1 text-sm text-muted-foreground">Drag & drop, or click to browse</p>
              <p className="mt-4 text-xs text-muted-foreground">Supported formats: MP3 · WAV</p>
              <input
                type="file"
                accept="audio/mpeg,audio/wav,audio/mp3,.mp3,.wav"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
            </label>
            {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
          </section>
        ) : (
          <section className="pt-4">
            {audioUrl && <audio ref={audioRef} src={audioUrl} className="hidden" />}

            <div className="rounded-3xl border border-white/40 bg-white/30 backdrop-blur-md p-6 sm:p-8 shadow-xl">
              <div className="mb-5 flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/50 shadow-sm">
                  <Music className="h-5 w-5 text-foreground" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-foreground">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                    {audioBuffer && ` · ${audioBuffer.duration.toFixed(1)}s`}
                  </p>
                </div>
                <button
                  onClick={togglePlay}
                  disabled={!audioBuffer}
                  className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-primary text-primary-foreground transition hover:opacity-90 disabled:opacity-40"
                >
                  {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="ml-0.5 h-5 w-5" />}
                </button>
              </div>

              <Waveform audioBuffer={audioBuffer} progress={progress} onSeek={seek} />

              <label htmlFor="playback-seek" className="sr-only">
                Playback position
              </label>
              <input
                id="playback-seek"
                type="range"
                min={0}
                max={1000}
                value={progress * 1000}
                onChange={(e) => seek(Number(e.target.value) / 1000)}
                className="mt-4 w-full accent-primary"
              />
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-3">
              <div className="rounded-3xl border border-white/40 bg-white/30 backdrop-blur-md p-6 lg:col-span-2 shadow-xl">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Predicted Genre</p>
                {analyzing ? (
                  <div className="mt-4">
                    <div className="h-12 w-48 animate-pulse rounded-lg bg-secondary" />
                    <div className="mt-3 h-4 w-32 animate-pulse rounded bg-secondary" />
                  </div>
                ) : error ? (
                  <p className="mt-4 text-sm text-destructive">{error}</p>
                ) : result ? (
                  <>
                    <h3 className="mt-2 text-6xl font-bold tracking-tight text-foreground">{result.genre}</h3>
                    <div className="mt-6">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-foreground">Confidence</span>
                        <span className="font-mono text-foreground">{Math.round(result.confidence * 100)}%</span>
                      </div>
                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-secondary">
                        <div
                          className="h-full rounded-full bg-primary transition-all duration-700"
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                    <div className="mt-8 grid grid-cols-2 gap-4 sm:grid-cols-3">
                      <Stat label="Tempo" value={`${result.features.tempo} BPM`} />
                      <Stat label="Energy" value={result.features.energy.toFixed(3)} />
                      <Stat label="Centroid" value={`${Math.round(result.features.spectralCentroid)} Hz`} />
                      <Stat label="ZCR" value={result.features.zeroCrossingRate.toFixed(5)} />
                      <Stat label="Roll-off" value={`${Math.round(result.features.spectralRolloff)} Hz`} />
                      <Stat label="Bandwidth" value={`${Math.round(result.features.spectralBandwidth)} Hz`} />
                    </div>
                  </>
                ) : null}
              </div>

              <div className="rounded-3xl border border-white/40 bg-white/30 backdrop-blur-md p-6 shadow-xl">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">All scores</p>
                <ul className="mt-4 space-y-3">
                  {(result?.scores ?? []).map((s) => (
                    <li key={s.genre}>
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-foreground">{s.genre}</span>
                        <span className="font-mono text-muted-foreground">{Math.round(s.score * 100)}%</span>
                      </div>
                      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-secondary">
                        <div className="h-full rounded-full bg-foreground/70" style={{ width: `${s.score * 100}%` }} />
                      </div>
                    </li>
                  ))}
                  {analyzing && [1, 2, 3, 4].map((i) => (
                    <li key={i} className="h-5 animate-pulse rounded bg-secondary" />
                  ))}
                </ul>
              </div>
            </div>

            {result && !analyzing && (
              <AudioVisualization
                audioBuffer={audioBuffer}
                result={result}
                progress={progress}
                onSeek={seek}
              />
            )}
          </section>
        )}
      </main>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/30 bg-white/40 shadow-sm px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 font-mono text-sm font-medium text-foreground">{value}</p>
    </div>
  );
}
