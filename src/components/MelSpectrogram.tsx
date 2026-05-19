import { useEffect, useRef } from "react";
import type { MelSpectrogramData } from "@/lib/genre-analysis";

interface MelSpectrogramProps {
  data: MelSpectrogramData;
  progress: number;
  onSeek?: (ratio: number) => void;
}

function parseOklch(color: string): [number, number, number] | null {
  const match = color.match(/oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)/);
  if (!match) return null;
  return [parseFloat(match[1]), parseFloat(match[2]), parseFloat(match[3])];
}

function lerpOklch(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): string {
  const l = a[0] + (b[0] - a[0]) * t;
  const c = a[1] + (b[1] - a[1]) * t;
  const h = a[2] + (b[2] - a[2]) * t;
  return `oklch(${l} ${c} ${h})`;
}

function buildColorStops(): (t: number) => string {
  const styles = getComputedStyle(document.documentElement);
  const low =
    parseOklch(styles.getPropertyValue("--secondary").trim()) ??
    [0.96, 0.005, 250];
  const mid =
    parseOklch(styles.getPropertyValue("--chart-3").trim()) ??
    [0.5, 0.08, 230];
  const high =
    parseOklch(styles.getPropertyValue("--chart-2").trim()) ??
    [0.35, 0.12, 250];
  const peak =
    parseOklch(styles.getPropertyValue("--primary").trim()) ??
    [0.22, 0.02, 250];

  return (t: number) => {
    if (t < 0.33) return lerpOklch(low, mid, t / 0.33);
    if (t < 0.66) return lerpOklch(mid, high, (t - 0.33) / 0.33);
    return lerpOklch(high, peak, (t - 0.66) / 0.34);
  };
}

export function MelSpectrogram({ data, progress, onSeek }: MelSpectrogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const progressRef = useRef(progress);

  // Keep the latest progress accessible to the resize handler without having
  // to re-bind the listener on every animation frame.
  progressRef.current = progress;

  useEffect(() => {
    draw();
  }, [data, progress]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [data]);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas || !data.values.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const pad = { left: 44, right: 52, top: 16, bottom: 28 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const { values, db_min, db_max, n_mels, n_frames, duration, fmax_hz } = data;
    const dbRange = db_max - db_min || 1;
    const colorAt = buildColorStops();

    const cellW = plotW / n_frames;
    const cellH = plotH / n_mels;

    for (let row = 0; row < n_mels; row++) {
      for (let col = 0; col < n_frames; col++) {
        const db = values[row][col];
        const t = (db - db_min) / dbRange;
        ctx.fillStyle = colorAt(Math.max(0, Math.min(1, t)));
        const y = pad.top + (n_mels - 1 - row) * cellH;
        ctx.fillRect(pad.left + col * cellW, y, cellW + 0.5, cellH + 0.5);
      }
    }

    const currentProgress = progressRef.current;
    const playedX = pad.left + currentProgress * plotW;
    if (currentProgress > 0 && currentProgress < 1) {
      ctx.strokeStyle = "oklch(0.99 0.002 250 / 0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(playedX, pad.top);
      ctx.lineTo(playedX, pad.top + plotH);
      ctx.stroke();
    }

    const styles = getComputedStyle(document.documentElement);
    const labelColor =
      styles.getPropertyValue("--muted-foreground").trim() || "oklch(0.5 0.015 250)";
    ctx.fillStyle = labelColor;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(fmax_hz / 1000)}k`, pad.left - 6, pad.top + 10);
    ctx.fillText("0", pad.left - 6, pad.top + plotH);

    ctx.textAlign = "center";
    ctx.fillText("0", pad.left, h - 8);
    const endLabel =
      duration >= 10 ? `${Math.round(duration)}` : duration.toFixed(1);
    ctx.fillText(endLabel, pad.left + plotW, h - 8);

    const legendX = w - pad.right + 10;
    const legendH = plotH;
    const steps = 48;
    for (let i = 0; i < steps; i++) {
      const t = i / (steps - 1);
      ctx.fillStyle = colorAt(t);
      ctx.fillRect(legendX, pad.top + legendH - (i + 1) * (legendH / steps), 10, legendH / steps + 0.5);
    }
    ctx.fillStyle = labelColor;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${Math.round(db_max)} dB`, legendX + 14, pad.top + 10);
    ctx.fillText(`${Math.round(db_min)} dB`, legendX + 14, pad.top + plotH);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const padLeft = 44;
    const plotW = rect.width - padLeft - 52;
    const x = e.clientX - rect.left - padLeft;
    onSeek(Math.max(0, Math.min(1, x / plotW)));
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="h-52 w-full cursor-pointer rounded-xl bg-secondary/30"
    />
  );
}
