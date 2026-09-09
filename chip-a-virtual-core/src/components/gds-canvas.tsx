import { useCallback, useEffect, useRef } from "react";
import {
  BLOCK_TINT,
  KIND_FILL,
  LAYER_STYLE,
  type CellInst,
  type KindInfo,
  type LayoutData,
} from "@/lib/layout-types";

type Cam = { panX: number; panY: number; scale: number };

type Props = {
  layout: LayoutData;
  layers: Record<string, boolean>;
  highlight: string | null;
  selected: CellInst | null;
  onSelect: (cell: CellInst | null) => void;
  onHover: (cell: CellInst | null, um: { x: number; y: number } | null) => void;
  onCam: (info: { scale: number; x: number; y: number }) => void;
  fitToken: number;
  zoomNudge: number;
};

function kindOf(name: string, kinds: KindInfo[]) {
  return kinds.find((k) => k.name === name)?.kind ?? "fill";
}

export function GdsCanvas({
  layout,
  layers,
  highlight,
  selected,
  onSelect,
  onHover,
  onCam,
  fitToken,
  zoomNudge,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cam = useRef<Cam>({ panX: 0, panY: 0, scale: 4 });
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const hoverRef = useRef<CellInst | null>(null);
  const layoutRef = useRef(layout);
  layoutRef.current = layout;
  const layersRef = useRef(layers);
  layersRef.current = layers;
  const highlightRef = useRef(highlight);
  highlightRef.current = highlight;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const onCamRef = useRef(onCam);
  onCamRef.current = onCam;
  const onHoverRef = useRef(onHover);
  onHoverRef.current = onHover;

  const dieW = layout.die[2] - layout.die[0];
  const dieH = layout.die[3] - layout.die[1];

  const worldFromEvent = (e: { clientX: number; clientY: number }) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = cam.current;
    return {
      x: (x - c.panX) / c.scale,
      y: dieH - (y - c.panY) / c.scale,
    };
  };

  const hit = (wx: number, wy: number): CellInst | null => {
    const L = layoutRef.current;
    for (let i = L.cells.length - 1; i >= 0; i--) {
      const cell = L.cells[i];
      if (wx >= cell.x && wx <= cell.x + cell.w && wy >= cell.y && wy <= cell.y + L.rowHeight) {
        return cell;
      }
    }
    return null;
  };

  const fit = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    const pad = 28;
    const scale = Math.min((w - pad * 2) / dieW, (h - pad * 2) / dieH);
    cam.current = {
      scale,
      panX: (w - dieW * scale) / 2,
      panY: (h - dieH * scale) / 2,
    };
  }, [dieW, dieH]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const L = layoutRef.current;
    const vis = layersRef.current;
    const c = cam.current;
    const s = c.scale;
    const toX = (x: number) => c.panX + x * s;
    const toY = (y: number) => c.panY + (dieH - y) * s;

    // substrate
    ctx.fillStyle = "#0c0e12";
    ctx.fillRect(0, 0, w, h);

    // die
    ctx.fillStyle = "#12151b";
    ctx.fillRect(toX(0), toY(dieH), dieW * s, dieH * s);

    // nwell
    if (vis.nwell) {
      const [x0, y0, x1, y1] = L.core;
      ctx.fillStyle = LAYER_STYLE.nwell.color;
      ctx.globalAlpha = LAYER_STYLE.nwell.alpha;
      ctx.fillRect(toX(x0), toY(y1), (x1 - x0) * s, (y1 - y0) * s);
      ctx.globalAlpha = 1;
    }

    // block tints
    for (const b of L.blocks) {
      const [x0, y0, x1, y1] = b.bbox;
      const active = highlightRef.current === b.id;
      ctx.fillStyle = active ? "rgba(126, 200, 184, 0.16)" : BLOCK_TINT[b.id] ?? "transparent";
      ctx.fillRect(toX(x0), toY(y1), (x1 - x0) * s, (y1 - y0) * s);
      if (active) {
        ctx.strokeStyle = "#7ec8b8";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(toX(x0), toY(y1), (x1 - x0) * s, (y1 - y0) * s);
      }
    }

    // std cells
    if (vis.cell) {
      const detail = s > 14;
      const rh = L.rowHeight;
      for (const cell of L.cells) {
        if (highlightRef.current && cell.b !== highlightRef.current && cell.b !== "tap") continue;
        const k = kindOf(cell.k, L.kinds);
        const x = toX(cell.x);
        const y = toY(cell.y + rh);
        const cw = cell.w * s;
        const ch = rh * s;
        ctx.fillStyle = KIND_FILL[k] ?? "#1a1d24";
        ctx.fillRect(x, y, cw, ch);
        if (s > 4) {
          ctx.strokeStyle = "rgba(255,255,255,0.06)";
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y, cw, ch);
        }
        if (detail) {
          // rails
          if (vis.met1) {
            ctx.fillStyle = LAYER_STYLE.met1.color;
            ctx.globalAlpha = 0.55;
            const rail = 0.48 * s;
            ctx.fillRect(x, y, cw, rail);
            ctx.fillRect(x, y + ch - rail, cw, rail);
            ctx.globalAlpha = 1;
          }
          if (vis.poly && k !== "fill" && k !== "tap") {
            ctx.fillStyle = LAYER_STYLE.poly.color;
            ctx.globalAlpha = 0.75;
            const n = Math.max(1, Math.round(cell.w / 0.46) - 1);
            for (let i = 0; i < n; i++) {
              const gx = x + ((i + 1) / (n + 1)) * cw - 0.07 * s;
              ctx.fillRect(gx, y + 0.5 * s, 0.15 * s, ch - s);
            }
            ctx.globalAlpha = 1;
          }
          if (vis.diff && k !== "fill") {
            ctx.fillStyle = LAYER_STYLE.diff.color;
            ctx.globalAlpha = 0.4;
            ctx.fillRect(x + 0.08 * s, y + ch * 0.55, cw - 0.16 * s, ch * 0.22);
            ctx.fillRect(x + 0.08 * s, y + ch * 0.22, cw - 0.16 * s, ch * 0.18);
            ctx.globalAlpha = 1;
          }
        }
      }
    }

    // power stripes
    for (const st of L.stripes) {
      if (!vis[st.layer]) continue;
      const style = LAYER_STYLE[st.layer];
      if (!style) continue;
      ctx.fillStyle = style.color;
      ctx.globalAlpha = style.alpha;
      ctx.fillRect(toX(st.x0), toY(st.y1), (st.x1 - st.x0) * s, (st.y1 - st.y0) * s);
      ctx.globalAlpha = 1;
    }

    const drawRoutes = (routes: typeof L.routes, key: "clock" | "signal") => {
      for (const r of routes) {
        if (!vis[r.layer]) continue;
        const style = LAYER_STYLE[r.layer];
        if (!style) continue;
        ctx.strokeStyle = style.color;
        ctx.globalAlpha = key === "clock" ? 0.45 : style.alpha;
        ctx.lineWidth = Math.max(0.6, r.w * s * (key === "clock" ? 0.7 : 1));
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        r.pts.forEach((p, i) => {
          const x = toX(p[0]);
          const y = toY(p[1]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    };
    drawRoutes(L.clock, "clock");
    drawRoutes(L.routes, "signal");

    // pins
    if (vis.pin) {
      ctx.font = `${Math.max(8, Math.min(11, 0.55 * s))}px "IBM Plex Mono", monospace`;
      ctx.textBaseline = "middle";
      for (const p of L.pins) {
        ctx.fillStyle = "#e7e5e4";
        ctx.globalAlpha = 0.85;
        ctx.fillRect(toX(p.x), toY(p.y + p.h), p.w * s, p.h * s);
        ctx.globalAlpha = 1;
        if (s > 3.2) {
          ctx.fillStyle = "#0b0c0e";
          ctx.textAlign = p.side === "right" ? "right" : p.side === "left" ? "left" : "center";
          const tx =
            p.side === "right" ? toX(p.x + p.w) - 3 : p.side === "left" ? toX(p.x) + 3 : toX(p.x + p.w / 2);
          const ty = toY(p.y + p.h / 2);
          ctx.fillText(p.name, tx, ty);
        }
      }
    }

    // die outline
    ctx.strokeStyle = "rgba(232,234,237,0.35)";
    ctx.lineWidth = 1;
    ctx.strokeRect(toX(0) + 0.5, toY(dieH) + 0.5, dieW * s - 1, dieH * s - 1);
    const [cx0, cy0, cx1, cy1] = L.core;
    const coreIsDie = cx0 === 0 && cy0 === 0 && cx1 === dieW && cy1 === dieH;
    if (!coreIsDie) {
      ctx.strokeStyle = "rgba(126,200,184,0.28)";
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(toX(cx0), toY(cy1), (cx1 - cx0) * s, (cy1 - cy0) * s);
      ctx.setLineDash([]);
    }

    // selected cell
    const sel = selectedRef.current;
    if (sel) {
      ctx.strokeStyle = "#7ec8b8";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(toX(sel.x), toY(sel.y + L.rowHeight), sel.w * s, L.rowHeight * s);
    }
    const hv = hoverRef.current;
    if (hv && hv !== sel) {
      ctx.strokeStyle = "rgba(232,234,237,0.55)";
      ctx.lineWidth = 1;
      ctx.strokeRect(toX(hv.x), toY(hv.y + L.rowHeight), hv.w * s, L.rowHeight * s);
    }

    // scale bar
    const targets = [1, 2, 5, 10, 20, 50];
    const targetPx = 80;
    let barUm = targets[0];
    for (const t of targets) {
      if (Math.abs(t * s - targetPx) < Math.abs(barUm * s - targetPx)) barUm = t;
    }
    const barX = 16;
    const barY = h - 22;
    ctx.fillStyle = "rgba(8,9,11,0.72)";
    ctx.fillRect(barX - 8, barY - 16, barUm * s + 54, 28);
    ctx.fillStyle = "#e8eaed";
    ctx.fillRect(barX, barY, barUm * s, 2);
    ctx.fillRect(barX, barY - 4, 1.5, 10);
    ctx.fillRect(barX + barUm * s, barY - 4, 1.5, 10);
    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`${barUm} µm`, barX + barUm * s + 8, barY + 1);

    onCamRef.current({
      scale: c.scale,
      x: (w / 2 - c.panX) / c.scale,
      y: dieH - (h / 2 - c.panY) / c.scale,
    });
  }, [dieH, dieW]);

  useEffect(() => {
    fit();
    draw();
  }, [fit, draw, fitToken]);

  useEffect(() => {
    if (!zoomNudge) return;
    const wrap = wrapRef.current;
    if (!wrap) return;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    const c = cam.current;
    const mx = w / 2;
    const my = h / 2;
    const wx = (mx - c.panX) / c.scale;
    const wy = (my - c.panY) / c.scale;
    const factor = zoomNudge > 0 ? 1.18 : 1 / 1.18;
    const next = Math.min(80, Math.max(0.6, c.scale * factor));
    cam.current = { scale: next, panX: mx - wx * next, panY: my - wy * next };
    draw();
  }, [zoomNudge, draw]);

  useEffect(() => {
    draw();
  }, [draw, layers, highlight, selected]);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => {
      fit();
      draw();
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [fit, draw]);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = wrap.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const c = cam.current;
      const wx = (mx - c.panX) / c.scale;
      const wy = (my - c.panY) / c.scale;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const next = Math.min(80, Math.max(0.6, c.scale * factor));
      cam.current = {
        scale: next,
        panX: mx - wx * next,
        panY: my - wy * next,
      };
      draw();
    };
    wrap.addEventListener("wheel", onWheel, { passive: false });
    return () => wrap.removeEventListener("wheel", onWheel);
  }, [draw]);

  return (
    <div
      ref={wrapRef}
      className="relative h-full min-h-0 w-full overflow-hidden bg-bg touch-none"
      onPointerDown={(e) => {
        (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
        drag.current = {
          x: e.clientX,
          y: e.clientY,
          panX: cam.current.panX,
          panY: cam.current.panY,
        };
      }}
      onPointerMove={(e) => {
        const um = worldFromEvent(e);
        if (drag.current) {
          const moved =
            Math.abs(e.clientX - drag.current.x) + Math.abs(e.clientY - drag.current.y);
          if (moved > 2) {
            cam.current.panX = drag.current.panX + (e.clientX - drag.current.x);
            cam.current.panY = drag.current.panY + (e.clientY - drag.current.y);
            draw();
          }
        } else {
          const cell = hit(um.x, um.y);
          if (cell !== hoverRef.current) {
            hoverRef.current = cell;
            onHoverRef.current(cell, um);
            draw();
          } else {
            onHoverRef.current(cell, um);
          }
        }
      }}
      onPointerUp={(e) => {
        const start = drag.current;
        drag.current = null;
        if (!start) return;
        const moved = Math.abs(e.clientX - start.x) + Math.abs(e.clientY - start.y);
        if (moved < 4) {
          const um = worldFromEvent(e);
          onSelect(hit(um.x, um.y));
        }
      }}
      onPointerLeave={() => {
        drag.current = null;
        hoverRef.current = null;
        onHoverRef.current(null, null);
        draw();
      }}
      onDoubleClick={() => {
        fit();
        draw();
      }}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  );
}
