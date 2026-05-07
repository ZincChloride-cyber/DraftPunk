import { useEffect, useRef } from "react";

interface WaveformProps {
  audioBuffer: AudioBuffer | null;
  progress: number; // 0..1
  onSeek?: (ratio: number) => void;
}

export function Waveform({ audioBuffer, progress, onSeek }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peaksRef = useRef<number[]>([]);

  useEffect(() => {
    if (!audioBuffer) return;
    const data = audioBuffer.getChannelData(0);
    const samples = 200;
    const blockSize = Math.floor(data.length / samples);
    const peaks: number[] = [];
    for (let i = 0; i < samples; i++) {
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(data[i * blockSize + j] || 0);
      }
      peaks.push(sum / blockSize);
    }
    const max = Math.max(...peaks, 0.01);
    peaksRef.current = peaks.map((p) => p / max);
    draw();
  }, [audioBuffer]);

  useEffect(() => {
    draw();
  }, [progress]);

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
    const peaks = peaksRef.current;
    if (!peaks.length) return;
    const barWidth = w / peaks.length;
    const playedIdx = Math.floor(progress * peaks.length);

    const styles = getComputedStyle(document.documentElement);
    const active = `oklch(${styles.getPropertyValue("--waveform").trim().replace("oklch(", "").replace(")", "")})`;
    const muted = `oklch(${styles.getPropertyValue("--waveform-muted").trim().replace("oklch(", "").replace(")", "")})`;

    peaks.forEach((peak, i) => {
      const barHeight = Math.max(2, peak * h * 0.85);
      const x = i * barWidth;
      const y = (h - barHeight) / 2;
      ctx.fillStyle = i <= playedIdx ? active : muted;
      ctx.fillRect(x + 1, y, Math.max(1, barWidth - 2), barHeight);
    });
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek((e.clientX - rect.left) / rect.width);
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="h-32 w-full cursor-pointer rounded-xl bg-secondary/40"
    />
  );
}