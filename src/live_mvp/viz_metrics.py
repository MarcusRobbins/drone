from __future__ import annotations
import time
from collections import deque
from queue import Empty
from typing import Dict, List

import numpy as np
import pyvista as pv

DEFAULT_SERIES = ["loss"]


class MetricsViewer:
    def __init__(self):
        self.series: List[str] = DEFAULT_SERIES[:]
        self.window: int = 5000
        self.ema_alpha: float = 0.0
        self.paused: bool = False

        # Data buffers
        self.x = deque(maxlen=self.window)  # iterations
        self.y: Dict[str, deque] = {}
        self.ema_y: Dict[str, deque] = {}

        # PyVista setup (same window type as map view)
        pv.set_plot_theme("document")
        try:
            self.pl = pv.Plotter(window_size=(900, 520), enable_keybindings=False)
        except TypeError:
            self.pl = pv.Plotter(window_size=(900, 520))
            for attr in ("enable_keybindings", "enable_key_bindings"):
                if hasattr(self.pl, attr):
                    try:
                        setattr(self.pl, attr, False)
                    except Exception:
                        pass
        self.pl.add_text("live_mvp :: metrics (loss & scalars)", font_size=10)
        try:
            self.chart = pv.Chart2D()
        except Exception as e:
            raise RuntimeError(f"PyVista Chart2D not available: {e!r}")
        self.lines: Dict[str, any] = {}
        self.ema_lines: Dict[str, any] = {}
        self._init_chart()

        # Key bindings
        self._bind_keys()

    def _bind_keys(self):
        for k in ("p", "s", "escape", "Esc"):
            try:
                self.pl.add_key_event(k, lambda k=k: self._on_key(k))
            except Exception:
                pass

    def _on_key(self, sym: str):
        k = (sym or "").lower()
        if k in ("escape", "esc"):
            try:
                self.pl.close()
            except Exception:
                pass
        elif k == "p":
            self.paused = not self.paused
        elif k == "s":
            try:
                ts = int(time.time())
                self.pl.screenshot(f"metrics_{ts}.png")
            except Exception:
                pass

    def _init_chart(self):
        # Clear and (re)build chart
        try:
            self.pl.remove_chart(self.chart)
        except Exception:
            pass
        self.chart = pv.Chart2D()
        self.lines.clear(); self.ema_lines.clear()
        # Initialize data structures
        self.y = {}
        self.ema_y = {}
        for s in self.series:
            self.y[s] = deque(maxlen=self.window)
            line = self.chart.line([], [], label=s)
            self.lines[s] = line
            if self.ema_alpha > 0.0:
                self.ema_y[s] = deque(maxlen=self.window)
                eline = self.chart.line([], [], label=f"{s}-ema", style="--", color="gray")
                self.ema_lines[s] = eline
        self.chart.x_label = "iteration"
        self.chart.y_label = "value"
        try:
            self.chart.legend.visible = True
        except Exception:
            pass
        self.pl.add_chart(self.chart)

    def _apply_ema(self, s: str, val: float) -> float:
        if self.ema_alpha <= 0.0:
            return val
        if len(self.ema_y.get(s, [])) == 0:
            return val
        return self.ema_alpha * val + (1.0 - self.ema_alpha) * self.ema_y[s][-1]

    def on_init(self, msg: dict):
        self.series = list(msg.get("series", self.series))
        self.window = int(msg.get("window", self.window))
        self.ema_alpha = float(msg.get("ema", self.ema_alpha))
        self.x = deque(maxlen=self.window)
        self._init_chart()

    def on_metrics(self, it: int, scalars: Dict[str, float]):
        if self.paused:
            return
        self.x.append(int(it))
        for s in self.series:
            v = float(scalars.get(s, np.nan))
            if not np.isfinite(v):
                continue
            if s not in self.y:
                self.y[s] = deque(maxlen=self.window)
            self.y[s].append(v)
            if self.ema_alpha > 0.0:
                ev = self._apply_ema(s, v)
                if s not in self.ema_y:
                    self.ema_y[s] = deque(maxlen=self.window)
                self.ema_y[s].append(float(ev))

        self._redraw()

    def _redraw(self):
        if len(self.x) == 0:
            return
        xs = np.fromiter(self.x, dtype=np.int64)
        updated_any = False
        for s in self.series:
            ys = np.fromiter(self.y.get(s, []), dtype=np.float32)
            if ys.size and s in self.lines:
                try:
                    # Prefer explicit update method if available
                    upd = getattr(self.lines[s], 'update', None)
                    if callable(upd):
                        upd(xs[-ys.size:], ys)
                    else:
                        try:
                            self.lines[s].x = xs[-ys.size:]
                            self.lines[s].y = ys
                        except Exception:
                            try:
                                pts = np.column_stack([xs[-ys.size:], ys])
                                setattr(self.lines[s], 'points', pts)
                            except Exception:
                                raise
                except Exception:
                    # Fallback: rebuild lines if properties not supported
                    self._init_chart()
                    break
                updated_any = True
            if self.ema_alpha > 0.0 and s in self.ema_y and s in self.ema_lines:
                eys = np.fromiter(self.ema_y.get(s, []), dtype=np.float32)
                if eys.size:
                    try:
                        upd = getattr(self.ema_lines[s], 'update', None)
                        if callable(upd):
                            upd(xs[-eys.size:], eys)
                        else:
                            try:
                                self.ema_lines[s].x = xs[-eys.size:]
                                self.ema_lines[s].y = eys
                            except Exception:
                                try:
                                    pts = np.column_stack([xs[-eys.size:], eys])
                                    setattr(self.ema_lines[s], 'points', pts)
                                except Exception:
                                    raise
                    except Exception:
                        self._init_chart()
                        break
                    updated_any = True
        if updated_any:
            try:
                self.pl.render()
            except Exception:
                pass

    def run(self, msg_q):
        try:
            self.pl.show(auto_close=False, interactive_update=True)
        except TypeError:
            try:
                self.pl.show(auto_close=False)
            except TypeError:
                self.pl.show()
        alive = True
        while alive:
            drained = False
            for _ in range(16):
                try:
                    m = msg_q.get_nowait()
                except Empty:
                    break
                drained = True
                tp = m.get("type", "")
                if tp == "init":
                    self.on_init(m)
                elif tp == "metrics":
                    self.on_metrics(int(m.get("iter", 0)), dict(m.get("scalars", {})))
                elif tp == "bye":
                    alive = False
            if not drained:
                time.sleep(0.01)
            try:
                self.pl.update()
            except Exception:
                alive = False
        try:
            self.pl.close()
        except Exception:
            pass


def run_metrics(msg_q):
    MetricsViewer().run(msg_q)
