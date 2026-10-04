/** 米字格 in the 1024 canvas frame: border, dashed centre cross and diagonals. */
export function RiceGrid() {
  const m = 1024;
  const h = m / 2;
  return (
    <g className="rice-grid">
      <rect x={4} y={4} width={m - 8} height={m - 8} className="rice-border" />
      <line x1={0} y1={h} x2={m} y2={h} />
      <line x1={h} y1={0} x2={h} y2={m} />
      <line x1={0} y1={0} x2={m} y2={m} />
      <line x1={m} y1={0} x2={0} y2={m} />
    </g>
  );
}

/** A square SVG overlay in canvas coordinates. */
export function CanvasSvg({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <svg viewBox="0 0 1024 1024" className={className ?? "overlay"} preserveAspectRatio="xMidYMid meet">
      {children}
    </svg>
  );
}
