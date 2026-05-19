import { useEffect, useRef } from "react";

interface WaveformLineProps {
  audioBuffer?: AudioBuffer | null;
  samples?: number[];
  progress: number;
  duration?: number;
  onSeek?: (ratio: number) => void;
}

export function WaveformLine({
  audioBuffer,
  samples: samplesProp,
  progress,
  duration: durationOverride,
  onSeek,
}: WaveformLineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const samplesRef = useRef<number[]>([]);
  const progressRef = useRef(progress);
  const durationRef = useRef<number | undefined>(durationOverride);

  // Keep refs in sync so the resize handler / draw() always see latest values
  progressRef.current = progress;
  durationRef.current = durationOverride;

  useEffect(() => {
    if (samplesProp?.length) {
      samplesRef.current = samplesProp;
      draw();
      return;
    }
    if (!audioBuffer) {
      samplesRef.current = [];
      draw();
      return;
    }
    const data = audioBuffer.getChannelData(0);
    const targetSamples = 600;
    const blockSize = Math.max(1, Math.floor(data.length / targetSamples));
    const samples: number[] = [];
    for (let i = 0; i < targetSamples; i++) {
      const start = i * blockSize;
      // Signed peak per block — preserves dynamics. Averaging min+max would
      // collapse to ~0 for typical (symmetric) music and flatten the line.
      let peakAbs = 0;
      let peakSigned = 0;
      for (let j = 0; j < blockSize && start + j < data.length; j++) {
        const v = data[start + j];
        const a = v < 0 ? -v : v;
        if (a > peakAbs) {
          peakAbs = a;
          peakSigned = v;
        }
      }
      samples.push(peakSigned);
    }
    const peak = Math.max(...samples.map((s) => Math.abs(s)), 0.01);
    samplesRef.current = samples.map((s) => s / peak);
    draw();
  }, [audioBuffer, samplesProp]);

  useEffect(() => {
    draw();
  }, [progress]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const samples = samplesRef.current;
    if (!samples.length) return;

    const styles = getComputedStyle(document.documentElement);
    const active = styles.getPropertyValue("--waveform").trim() || "oklch(0.22 0.02 250)";
    const muted = styles.getPropertyValue("--waveform-muted").trim() || "oklch(0.85 0.01 250)";
    const grid = styles.getPropertyValue("--border").trim() || "oklch(0.92 0.008 250)";

    const pad = { left: 44, right: 12, top: 16, bottom: 28 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const midY = pad.top + plotH / 2;
    const currentProgress = progressRef.current;
    const playedX = pad.left + currentProgress * plotW;

    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, midY);
    ctx.lineTo(pad.left + plotW, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    const drawPath = (color: string, endRatio: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const endIdx = Math.floor(endRatio * (samples.length - 1));
      for (let i = 0; i <= endIdx; i++) {
        const x = pad.left + (i / (samples.length - 1)) * plotW;
        const y = midY - samples[i] * (plotH / 2) * 0.92;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    drawPath(muted, 1);
    if (currentProgress > 0) drawPath(active, currentProgress);

    if (currentProgress > 0 && currentProgress < 1) {
      ctx.strokeStyle = active;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(playedX, pad.top);
      ctx.lineTo(playedX, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = styles.getPropertyValue("--muted-foreground").trim() || "oklch(0.5 0.015 250)";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("1.0", pad.left - 6, pad.top + 4);
    ctx.fillText("0", pad.left - 6, midY + 4);
    ctx.fillText("-1.0", pad.left - 6, pad.top + plotH);

    const duration =
      durationRef.current ?? (audioBuffer ? audioBuffer.duration : 0);
    ctx.textAlign = "center";
    ctx.fillText("0", pad.left, h - 8);
    ctx.fillText(
      duration >= 10 ? `${Math.round(duration)}` : duration.toFixed(1),
      pad.left + plotW,
      h - 8,
    );
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const padLeft = 44;
    const plotW = rect.width - padLeft - 12;
    const x = e.clientX - rect.left - padLeft;
    onSeek(Math.max(0, Math.min(1, x / plotW)));
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="h-40 w-full cursor-pointer rounded-xl bg-secondary/30"
    />
  );
}
