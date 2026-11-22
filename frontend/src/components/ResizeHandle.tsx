import { useRef, useState } from "react";

interface ResizeHandleProps {
  side: "left" | "right";
  width: number;
  minWidth: number;
  maxWidth: number;
  defaultWidth: number;
  label: string;
  onResize: (nextWidth: number) => void;
}

export function ResizeHandle({
  side,
  width,
  minWidth,
  maxWidth,
  defaultWidth,
  label,
  onResize,
}: ResizeHandleProps) {
  const [dragging, setDragging] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(width);

  const clamp = (v: number) => Math.max(minWidth, Math.min(maxWidth, Math.round(v)));

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    startXRef.current = e.clientX;
    startWidthRef.current = width;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    const delta = e.clientX - startXRef.current;
    const next = side === "left" ? startWidthRef.current + delta : startWidthRef.current - delta;
    onResize(clamp(next));
  };

  const handleLostCapture = () => {
    setDragging(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 40 : 20;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      onResize(clamp(side === "left" ? width - step : width + step));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      onResize(clamp(side === "left" ? width + step : width - step));
    } else if (e.key === "Home") {
      e.preventDefault();
      onResize(minWidth);
    } else if (e.key === "End") {
      e.preventDefault();
      onResize(maxWidth);
    }
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={`${label}. Drag or use left/right arrow keys to resize. Double-click to reset.`}
      aria-valuenow={width}
      aria-valuemin={minWidth}
      aria-valuemax={maxWidth}
      tabIndex={0}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onLostPointerCapture={handleLostCapture}
      onDoubleClick={() => onResize(defaultWidth)}
      onKeyDown={handleKeyDown}
      title="Drag to resize panel (double-click to reset)"
      className={`panel-resize-handle hidden lg:flex w-1.5 shrink-0 items-center justify-center bg-slate-900/90 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
        dragging ? "bg-cyan-500/40" : "hover:bg-cyan-500/25"
      }`}
    >
      <span
        aria-hidden="true"
        className={`h-10 w-0.5 rounded-full transition-colors ${
          dragging ? "bg-cyan-300" : "bg-slate-700"
        }`}
      />
    </div>
  );
}
