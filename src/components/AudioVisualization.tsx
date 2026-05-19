import type { ReactNode } from "react";
import { WaveformLine } from "@/components/WaveformLine";
import { MelSpectrogram } from "@/components/MelSpectrogram";
import type { GenreResult } from "@/lib/genre-analysis";

interface AudioVisualizationProps {
  audioBuffer: AudioBuffer | null;
  result: GenreResult;
  progress: number;
  onSeek?: (ratio: number) => void;
}

export function AudioVisualization({
  audioBuffer,
  result,
  progress,
  onSeek,
}: AudioVisualizationProps) {
  const mel = result.melSpectrogram;
  if (!mel) return null;

  const duration = mel.duration;

  return (
    <div className="mt-6 rounded-3xl border border-white/40 bg-white/30 backdrop-blur-md p-6 sm:p-8 shadow-xl animate-fade-in">
      <header className="mb-8">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Audio visualisation
        </p>
        <h3 className="mt-1 text-2xl font-semibold tracking-tight text-foreground">
          Spectrogram analysis
        </h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Predicted{" "}
          <span className="font-medium text-foreground">{result.genre}</span>
          {" · "}
          <span className="font-mono">
            {(result.confidence * 100).toFixed(1)}% confidence
          </span>
        </p>
      </header>

      <VizPlot
        title="Waveform"
        subtitle="Amplitude over time"
        axisY="Amplitude"
        axisX="Time (seconds)"
      >
        <WaveformLine
          samples={result.waveform}
          audioBuffer={result.waveform ? null : audioBuffer}
          progress={progress}
          duration={duration}
          onSeek={onSeek}
        />
      </VizPlot>

      <VizPlot
        className="mt-8"
        title="Mel-spectrogram"
        subtitle="Frequency energy over time"
        axisY="Frequency (Hz, mel)"
        axisX="Time (seconds)"
      >
        <MelSpectrogram data={mel} progress={progress} onSeek={onSeek} />
      </VizPlot>
    </div>
  );
}

function VizPlot({
  title,
  subtitle,
  axisY,
  axisX,
  children,
  className = "",
}: {
  title: string;
  subtitle: string;
  axisY: string;
  axisX: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={className}>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-foreground">{title}</p>
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        </div>
        <div className="flex gap-4 text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>{axisY}</span>
          <span>{axisX}</span>
        </div>
      </div>
      {children}
    </section>
  );
}
