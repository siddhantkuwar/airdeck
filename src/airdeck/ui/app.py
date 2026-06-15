from __future__ import annotations

import logging
import time
import tkinter as tk
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from airdeck.budget.counters import BudgetCounters
from airdeck.budget.limiter import BudgetLimiter, BudgetLimits
from airdeck.camera.frame_queue import LatestFrameQueue
from airdeck.camera.gesture_feedback import (
    GESTURE_LABELS,
    GestureFeedback,
    LocalGestureFeedbackAnalyzer,
)
from airdeck.camera.preview import (
    draw_feedback_overlay,
    frame_to_ppm,
    mirror_frame,
    resize_frame_to_fit,
)
from airdeck.camera.producer import CaptureFactory, CaptureProducer
from airdeck.config import Settings
from airdeck.gestures.state_machine import GestureStateMachine
from airdeck.lifecycle.shutdown import ShutdownManager
from airdeck.overshoot.schemas import GestureInference


BG = "#F7F6F2"
SURFACE = "#FDFCFB"
WHITE = SURFACE
ELEVATED = "#F9F8F5"
INK = "#171717"
MUTED = "#6F6D68"
SUBTLE = "#A29F98"
LINE = "#E4E0D8"
SOFT = "#EFEDE8"
ACTIVE_BG = "#ECEAE4"
GREEN = "#14A66A"
BLUE = "#2F6FED"
ORANGE = "#C96B25"
RED = "#D6404F"
RED_SOFT = "#FFF1F2"
FONT_UI = "Avenir Next"


@dataclass(frozen=True)
class StatsPoint:
    fps: float
    drops_per_second: float
    confidence: float
    hand_area: float
    hand_visible: bool


@dataclass(frozen=True)
class ButtonPalette:
    bg: str
    fg: str
    border: str
    hover_bg: str
    pressed_bg: str
    disabled_bg: str = "#ECEAE4"
    disabled_fg: str = "#A5A29B"
    disabled_border: str = "#E1DDD5"


BUTTON_PALETTES = {
    "primary": ButtonPalette(
        bg=INK,
        fg=SURFACE,
        border=INK,
        hover_bg="#2B2B2B",
        pressed_bg="#050505",
    ),
    "secondary": ButtonPalette(
        bg=ELEVATED,
        fg=INK,
        border=LINE,
        hover_bg=SOFT,
        pressed_bg="#E8E5DD",
    ),
    "danger": ButtonPalette(
        bg=RED_SOFT,
        fg="#9F1239",
        border="#FFD5DC",
        hover_bg="#FFE4E8",
        pressed_bg="#FFD6DE",
        disabled_bg="#F1EFEB",
        disabled_fg="#B6B1A8",
    ),
}


class ActionButton(tk.Canvas):
    def __init__(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], None],
        *,
        variant: str,
        height: int = 54,
    ) -> None:
        super().__init__(
            parent,
            bg=BG,
            bd=0,
            highlightthickness=0,
            height=height,
            cursor="hand2",
            takefocus=1,
        )
        self._text = text
        self._command = command
        self._variant = variant
        self._height = height
        self._enabled = True
        self._hovered = False
        self._pressed = False
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Key-Return>", lambda _event: self.invoke())
        self.bind("<Key-space>", lambda _event: self.invoke())
        self.bind("<FocusIn>", lambda _event: self._draw())
        self.bind("<FocusOut>", lambda _event: self._draw())

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._hovered = False if not enabled else self._hovered
        self._pressed = False if not enabled else self._pressed
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def invoke(self) -> None:
        if self._enabled:
            self._command()

    def _on_enter(self, _event: tk.Event[tk.Misc]) -> None:
        if self._enabled:
            self._hovered = True
            self._draw()

    def _on_leave(self, _event: tk.Event[tk.Misc]) -> None:
        self._hovered = False
        self._pressed = False
        self._draw()

    def _on_press(self, _event: tk.Event[tk.Misc]) -> None:
        if self._enabled:
            self.focus_set()
            self._pressed = True
            self._draw()

    def _on_release(self, event: tk.Event[tk.Misc]) -> None:
        was_pressed = self._pressed
        self._pressed = False
        self._draw()
        if was_pressed and self._enabled and 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height():
            self.invoke()

    def _draw(self) -> None:
        self.delete("all")
        palette = BUTTON_PALETTES[self._variant]
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), self._height)
        if not self._enabled:
            fill = palette.disabled_bg
            outline = palette.disabled_border
            text_fill = palette.disabled_fg
        elif self._pressed:
            fill = palette.pressed_bg
            outline = palette.border
            text_fill = palette.fg
        elif self._hovered:
            fill = palette.hover_bg
            outline = palette.border
            text_fill = palette.fg
        else:
            fill = palette.bg
            outline = palette.border
            text_fill = palette.fg

        radius = min(14, max(8, height // 2 - 4))
        _rounded_rect(self, 1, 1, width - 1, height - 1, radius, fill=fill, outline=outline, width=1)
        if self == self.focus_get():
            _rounded_rect(self, 4, 4, width - 4, height - 4, radius - 3, outline=BLUE, width=2)
        self.create_text(
            width / 2,
            height / 2,
            text=self._text,
            fill=text_fill,
            font=(FONT_UI, 16, "bold"),
        )


def _rounded_rect(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    **kwargs: object,
) -> int:
    points = (
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    )
    return int(canvas.create_polygon(points, smooth=True, splinesteps=16, **kwargs))


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def fps_ratio(fps: float, target_fps: float) -> float:
    return clamp01(fps / max(target_fps, 1.0))


def drop_pressure(drops_per_second: float, target_fps: float) -> float:
    # The latest-frame queue intentionally drops stale frames; scale this as pressure, not a full chart axis.
    return clamp01(drops_per_second / max(target_fps * 4.0, 1.0))


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _feedback_to_inference(feedback: GestureFeedback) -> GestureInference:
    gesture = feedback.gesture if feedback.gesture != "HAND_TRACKED" else "UNCERTAIN"
    if gesture == "NO_GESTURE":
        hand_visible = False
    else:
        hand_visible = feedback.hand_visible
    return GestureInference(
        gesture=gesture,
        confidence=feedback.confidence,
        hand_visible=hand_visible,
        description=feedback.description,
        epoch=0,
        latency_ms=0.0,
    )


class AirDeckApp:
    def __init__(
        self,
        root: tk.Tk,
        settings: Settings,
        *,
        capture_factory: CaptureFactory | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.root = root
        self.settings = settings
        self._capture_factory = capture_factory
        self._logger = logger or logging.getLogger("airdeck.ui")
        self._frame_queue = LatestFrameQueue()
        self._producer: CaptureProducer | None = None
        self._gesture_analyzer = LocalGestureFeedbackAnalyzer()
        self._gesture_state = GestureStateMachine()
        self._budget_counters = BudgetCounters(started_at_monotonic=time.monotonic())
        self._budget_limiter = BudgetLimiter(
            BudgetLimits(
                max_completion_requests=settings.max_completion_requests,
                max_session_minutes=settings.max_session_minutes,
                max_requests_per_minute=settings.max_requests_per_minute,
                max_inference_hz=settings.max_inference_hz,
            )
        )
        self._last_preview_frame_index: int | None = None
        self._preview_photo: tk.PhotoImage | None = None
        self._closed = False
        self._tick_after_id: str | None = None
        self._last_metric_time = time.monotonic()
        self._last_frame_count = 0
        self._last_drop_count = 0
        self._stats_history: deque[StatsPoint] = deque(maxlen=160)
        self._gesture_rows: dict[str, tk.Label] = {}
        self._last_logged_gesture = "NO_GESTURE"
        self._last_feedback = GestureFeedback(
            gesture="NO_GESTURE",
            confidence=0.0,
            hand_visible=False,
            motion_score=0.0,
            description="Ready",
            timestamp_monotonic=time.monotonic(),
        )

        self.shutdown_manager = ShutdownManager(logger=self._logger)
        self.shutdown_manager.register("capture_producer", self._stop_capture, priority=10)

        self.status_var = tk.StringVar(value="camera off")
        self.listening_var = tk.StringVar(value="disabled")
        self.preview_var = tk.StringVar(value="Start camera")
        self.frames_var = tk.StringVar(value="0")
        self.drops_var = tk.StringVar(value="0")
        self.fps_var = tk.StringVar(value="0.0")
        self.queue_var = tk.StringVar(value="0 / 1")
        self.gesture_var = tk.StringVar(value="NO GESTURE")
        self.confidence_var = tk.StringVar(value="0%")
        self.area_var = tk.StringVar(value="0%")
        self.box_var = tk.StringVar(value="no hand box")
        self.event_var = tk.StringVar(value="Ready")
        self.telemetry_summary_var = tk.StringVar(value="idle · 0.0 fps · 0% confidence")
        self.overshoot_var = tk.StringVar(value="overshoot offline")
        self.latency_var = tk.StringVar(value="0 ms")
        self.requests_var = tk.StringVar(value="0")
        self.budget_var = tk.StringVar(value="OK")
        self.session_timer_var = tk.StringVar(value="00:00")
        self.candidate_var = tk.StringVar(value="0 / 2")
        self.command_var = tk.StringVar(value="none")
        self.emergency_disabled = False
        self.start_button: ActionButton | None = None
        self.stop_button: ActionButton | None = None
        self.emergency_button: ActionButton | None = None

        self._build()
        self._maximize()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._schedule_tick()

    def start_session(self) -> None:
        if self.emergency_disabled:
            self._set_status("disabled")
            self._log_event("Emergency disable is active")
            self._update_controls()
            return
        if self._producer and self._producer.is_running:
            return

        self._frame_queue = LatestFrameQueue()
        self._gesture_analyzer = LocalGestureFeedbackAnalyzer()
        self._gesture_state = GestureStateMachine()
        self._budget_counters.start_stream(time.monotonic())
        self._last_preview_frame_index = None
        self._last_metric_time = time.monotonic()
        self._last_frame_count = 0
        self._last_drop_count = 0
        self._stats_history.clear()
        self._set_status("connecting")
        self._log_event("Opening camera")
        self._producer = CaptureProducer(
            self._frame_queue,
            camera_index=self.settings.camera_index,
            target_fps=self.settings.target_camera_fps,
            capture_factory=self._capture_factory,
            logger=self._logger,
            on_error=self._handle_camera_error,
        )
        try:
            self._producer.start()
        except BaseException as exc:  # noqa: BLE001 - surface camera permission/open failures to UI.
            self._producer = None
            self._handle_camera_error(exc)
            self._update_controls()
            return
        self.listening_var.set("enabled")
        self._set_status("streaming")
        self.overshoot_var.set("demo-local")
        self._log_event("Camera streaming")
        self._update_controls()
        self._logger.info("session_started camera_index=%s", self.settings.camera_index)

    def stop_session(self) -> None:
        self._stop_capture()
        self._budget_counters.stop_stream(time.monotonic())
        self.listening_var.set("disabled")
        self._set_status("camera off")
        self.overshoot_var.set("offline")
        self._log_event("Camera stopped")
        self._update_controls()
        self._logger.info("session_stopped")

    def emergency_disable(self) -> None:
        self.emergency_disabled = True
        self.listening_var.set("disabled")
        self._stop_capture()
        self._budget_counters.stop_stream(time.monotonic())
        self._set_status("disabled")
        self.overshoot_var.set("disabled")
        self._apply_gesture_feedback(
            GestureFeedback(
                gesture="NO_GESTURE",
                confidence=0.0,
                hand_visible=False,
                motion_score=0.0,
                description="Emergency disable active",
                timestamp_monotonic=time.monotonic(),
            )
        )
        self._log_event("Emergency disable activated")
        self._update_controls()
        self._logger.warning("emergency_disable_activated")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._tick_after_id is not None:
            try:
                self.root.after_cancel(self._tick_after_id)
            except tk.TclError:
                pass
            self._tick_after_id = None
        self.shutdown_manager.shutdown("window_closed")
        self.root.destroy()

    def _build(self) -> None:
        self.root.title("AirDeck")
        self.root.configure(bg=BG)
        self.root.minsize(1200, 760)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        shell = tk.Frame(self.root, bg=BG, padx=24, pady=22)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, minsize=230)
        shell.columnconfigure(1, weight=1)
        shell.columnconfigure(2, minsize=360)
        shell.rowconfigure(1, weight=1)
        shell.rowconfigure(2, minsize=260)

        header = tk.Frame(shell, bg=BG)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 22))
        header.columnconfigure(1, weight=1)
        tk.Label(
            header,
            text="AirDeck",
            bg=BG,
            fg=INK,
            font=(FONT_UI, 34, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Realtime gesture stream",
            bg=BG,
            fg=MUTED,
            font=(FONT_UI, 16),
        ).grid(row=0, column=1, sticky="w", padx=(18, 0))
        tk.Label(
            header,
            textvariable=self.status_var,
            bg=SOFT,
            fg=INK,
            padx=16,
            pady=8,
            font=(FONT_UI, 14, "bold"),
        ).grid(row=0, column=2, sticky="e")

        nav = tk.Frame(shell, bg=BG)
        nav.grid(row=1, column=0, sticky="nsew", padx=(0, 24))
        nav.columnconfigure(0, weight=1)
        self._section_label(nav, "Gestures").grid(row=0, column=0, sticky="w", pady=(0, 12))
        for index, gesture in enumerate(GESTURE_LABELS):
            row = tk.Label(
                nav,
                text=gesture.replace("_", " "),
                bg=BG,
                fg=MUTED,
                anchor="w",
                padx=14,
                pady=9,
                font=(FONT_UI, 15, "bold"),
            )
            row.grid(row=index + 1, column=0, sticky="ew", pady=2)
            self._gesture_rows[gesture] = row

        camera = tk.Frame(shell, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        camera.grid(row=1, column=1, sticky="nsew", padx=(0, 24))
        camera.columnconfigure(0, weight=1)
        camera.rowconfigure(1, weight=1)
        tk.Label(
            camera,
            text="latest frame",
            bg=SURFACE,
            fg=MUTED,
            font=(FONT_UI, 14, "bold"),
            anchor="w",
            padx=18,
            pady=12,
        ).grid(row=0, column=0, sticky="ew")
        self.preview_label = tk.Label(
            camera,
            textvariable=self.preview_var,
            anchor="center",
            compound="center",
            bg="#0E0E10",
            fg="#FAFAFA",
            font=(FONT_UI, 28, "bold"),
            bd=0,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        inspector = tk.Frame(shell, bg=BG)
        inspector.grid(row=1, column=2, sticky="nsew")
        inspector.columnconfigure(0, weight=1)
        inspector.rowconfigure(2, weight=1)

        signal = tk.Frame(inspector, bg=BG)
        signal.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        signal.columnconfigure(0, weight=1)
        self._section_label(signal, "Signal").grid(row=0, column=0, sticky="w")
        tk.Label(
            signal,
            textvariable=self.gesture_var,
            bg=BG,
            fg=INK,
            anchor="w",
            justify="left",
            wraplength=360,
            font=(FONT_UI, 34, "bold"),
        ).grid(row=1, column=0, sticky="ew", pady=(7, 2))
        tk.Label(
            signal,
            textvariable=self.box_var,
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=(FONT_UI, 15),
        ).grid(row=2, column=0, sticky="ew")
        tk.Label(
            signal,
            textvariable=self.event_var,
            bg=BG,
            fg=MUTED,
            anchor="w",
            wraplength=360,
            font=(FONT_UI, 15),
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        controls = tk.Frame(inspector, bg=BG)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 24))
        controls.columnconfigure((0, 1), weight=1)
        self.start_button = self._button(controls, "Start", self.start_session, variant="primary")
        self.start_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        self.stop_button = self._button(controls, "Stop", self.stop_session, variant="secondary")
        self.stop_button.grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        self.emergency_button = self._button(
            controls,
            "Emergency disable",
            self.emergency_disable,
            variant="danger",
        )
        self.emergency_button.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0)
        )

        metrics = tk.Frame(inspector, bg=BG)
        metrics.grid(row=2, column=0, sticky="nsew")
        metrics.columnconfigure((0, 1), weight=1)
        self._metric(metrics, "fps", self.fps_var, 0, 0)
        self._metric(metrics, "confidence", self.confidence_var, 0, 1)
        self._metric(metrics, "latency", self.latency_var, 1, 0)
        self._metric(metrics, "requests", self.requests_var, 1, 1)
        self._metric(metrics, "candidate", self.candidate_var, 2, 0)
        self._metric(metrics, "budget", self.budget_var, 2, 1)
        self._metric(metrics, "session", self.session_timer_var, 3, 0)
        self._metric(metrics, "queue", self.queue_var, 3, 1)
        self._metric(metrics, "overshoot", self.overshoot_var, 4, 0)
        self._metric(metrics, "command", self.command_var, 4, 1)

        graph = tk.Frame(shell, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        graph.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(24, 0))
        graph.columnconfigure(0, weight=1)
        graph.rowconfigure(2, weight=1)
        graph_header = tk.Frame(graph, bg=SURFACE)
        graph_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 4))
        graph_header.columnconfigure(1, weight=1)
        tk.Label(
            graph_header,
            text="Stream telemetry",
            bg=SURFACE,
            fg=INK,
            font=(FONT_UI, 18, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            graph_header,
            textvariable=self.telemetry_summary_var,
            bg=SURFACE,
            fg=MUTED,
            font=(FONT_UI, 13),
            anchor="e",
        ).grid(row=0, column=1, sticky="e")
        legend = tk.Frame(graph, bg=SURFACE)
        legend.grid(row=1, column=0, sticky="ew", padx=18, pady=(2, 8))
        self._legend_item(legend, "fps", INK).pack(side="left", padx=(0, 18))
        self._legend_item(legend, "confidence", GREEN).pack(side="left", padx=(0, 18))
        self._legend_item(legend, "hand area", BLUE).pack(side="left", padx=(0, 18))
        self._legend_item(legend, "stale drops", ORANGE).pack(side="left")
        self.stats_canvas = tk.Canvas(graph, bg=SURFACE, bd=0, highlightthickness=0, height=190)
        self.stats_canvas.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self._update_controls()

    def _maximize(self) -> None:
        self.root.update_idletasks()
        try:
            self.root.state("zoomed")
            return
        except tk.TclError:
            pass
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.root.geometry(f"{width}x{height}+0+0")

    def _schedule_tick(self) -> None:
        if self._closed:
            return
        self._update_metrics()
        self._update_preview()
        self._draw_stats()
        if not self._closed:
            self._tick_after_id = self.root.after(100, self._schedule_tick)

    def _update_metrics(self) -> None:
        metrics = self._frame_queue.metrics()
        now = time.monotonic()
        elapsed = max(now - self._last_metric_time, 0.001)
        frame_delta = max(metrics.frames_received - self._last_frame_count, 0)
        drop_delta = max(metrics.frames_dropped - self._last_drop_count, 0)
        fps = frame_delta / elapsed
        drops_per_second = drop_delta / elapsed

        self._last_metric_time = now
        self._last_frame_count = metrics.frames_received
        self._last_drop_count = metrics.frames_dropped

        feedback = self._last_feedback
        if self._producer and self._producer.is_running:
            self._stats_history.append(
                StatsPoint(
                    fps=fps,
                    drops_per_second=drops_per_second,
                    confidence=feedback.confidence,
                    hand_area=feedback.motion_score,
                    hand_visible=feedback.hand_visible,
                )
            )

        self.fps_var.set(f"{fps:.1f}")
        self.frames_var.set(str(metrics.frames_received))
        self.drops_var.set(f"{drops_per_second:.1f}")
        self.queue_var.set(f"{metrics.qsize} / {self._frame_queue.maxsize}")
        snapshot = self._budget_counters.snapshot(
            now=now,
            max_completion_requests=self.settings.max_completion_requests,
        )
        budget_decision = self._budget_limiter.can_request(self._budget_counters, now=now)
        self.requests_var.set(str(snapshot.total_completion_requests))
        self.budget_var.set(budget_decision.status)
        self.session_timer_var.set(_format_duration(snapshot.session_seconds))
        self.telemetry_summary_var.set(
            f"{fps:.1f} fps · {round(feedback.confidence * 100)}% confidence · "
            f"{drops_per_second:.1f} stale drops/sec · {snapshot.total_completion_requests} requests"
        )

    def _update_preview(self) -> None:
        envelope = self._frame_queue.peek_latest()
        if envelope is None or envelope.frame_index == self._last_preview_frame_index:
            return
        try:
            mirrored_frame = mirror_frame(envelope.frame)
            feedback = self._gesture_analyzer.analyze(mirrored_frame)
            overlay_frame = draw_feedback_overlay(mirrored_frame, feedback)
            overlay_frame = resize_frame_to_fit(
                overlay_frame,
                max_width=max(self.preview_label.winfo_width() - 2, 640),
                max_height=max(self.preview_label.winfo_height() - 2, 480),
            )
            image_data = frame_to_ppm(overlay_frame, mirror=False)
        except ValueError as exc:
            self.preview_var.set(str(exc))
            return

        self._preview_photo = tk.PhotoImage(data=image_data, format="PPM")
        self.preview_label.configure(image=self._preview_photo, text="")
        self._last_preview_frame_index = envelope.frame_index
        self._apply_gesture_feedback(feedback)

    def _apply_gesture_feedback(self, feedback: GestureFeedback) -> None:
        self._last_feedback = feedback
        display_name = feedback.gesture.replace("_", " ")
        self.gesture_var.set(display_name)
        self.confidence_var.set(f"{round(feedback.confidence * 100)}%")
        self.area_var.set(f"{round(feedback.motion_score * 100)}%")
        if feedback.bounding_box:
            box = feedback.bounding_box
            self.box_var.set(f"{box.width} x {box.height} at {box.x}, {box.y}")
        else:
            self.box_var.set("no hand box")

        decision = self._gesture_state.observe(_feedback_to_inference(feedback), now=time.monotonic())
        self.candidate_var.set(f"{decision.confirmation_count} / 2")
        self.listening_var.set("enabled" if decision.listening_enabled else "disabled")
        if decision.accepted and decision.command is not None:
            self.command_var.set(decision.command.label)
            self._log_event(f"{decision.command.label} accepted")
        elif decision.reason not in {"neutral gesture", "awaiting confirmation"}:
            self.command_var.set(decision.reason)

        active = feedback.gesture
        for gesture, row in self._gesture_rows.items():
            is_active = gesture == active
            row.configure(
                bg=ACTIVE_BG if is_active else BG,
                fg=INK if is_active else MUTED,
            )

        if feedback.hand_visible:
            self._set_status("tracking")
            self._log_event(f"{display_name.lower()} · {round(feedback.confidence * 100)}%")
        elif self._producer and self._producer.is_running:
            self._set_status("streaming")
            self._log_event(feedback.description)
        if feedback.gesture != self._last_logged_gesture:
            self._logger.info(
                "gesture_feedback_changed gesture=%s confidence=%.2f hand_visible=%s",
                feedback.gesture,
                feedback.confidence,
                feedback.hand_visible,
            )
            self._last_logged_gesture = feedback.gesture

    def _draw_stats(self) -> None:
        canvas = self.stats_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 720)
        height = max(canvas.winfo_height(), 190)
        canvas.create_rectangle(0, 0, width, height, fill=SURFACE, outline="")
        history = list(self._stats_history)
        left, right, top, bottom = 58, width - 24, 18, height - 36
        _rounded_rect(canvas, left, top, right, bottom, 10, fill=ELEVATED, outline=LINE, width=1)

        if len(history) < 2:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Start camera to stream telemetry",
                fill=MUTED,
                font=(FONT_UI, 15, "bold"),
            )
            return

        for label, ratio in (("100", 0.0), ("75", 0.25), ("50", 0.5), ("25", 0.75), ("0", 1.0)):
            y = top + (bottom - top) * ratio
            canvas.create_line(left, y, right, y, fill="#EBE8E1")
            canvas.create_text(
                left - 14,
                y,
                text=label,
                fill=SUBTLE,
                anchor="e",
                font=(FONT_UI, 10, "bold"),
            )

        def points_for(values: list[float]) -> list[float]:
            points: list[float] = []
            span = max(len(values) - 1, 1)
            for index, value in enumerate(values):
                x = left + (right - left) * (index / span)
                y = bottom - (bottom - top) * clamp01(value)
                points.extend((x, y))
            return points

        fps_values = [fps_ratio(point.fps, self.settings.target_camera_fps) for point in history]
        confidence_values = [point.confidence for point in history]
        area_values = [point.hand_area for point in history]

        drop_band_height = (bottom - top) * 0.26
        for index, point in enumerate(history):
            value = drop_pressure(point.drops_per_second, self.settings.target_camera_fps)
            x = left + (right - left) * (index / max(len(history) - 1, 1))
            bar_height = drop_band_height * value
            canvas.create_rectangle(
                x - 1,
                bottom - bar_height,
                x + 1,
                bottom,
                fill="#F2B36D",
                outline="",
            )

        canvas.create_line(*points_for(fps_values), fill=INK, width=2)
        canvas.create_line(*points_for(confidence_values), fill=GREEN, width=2)
        canvas.create_line(*points_for(area_values), fill=BLUE, width=2)
        canvas.create_line(left, bottom, right, bottom, fill="#D7D2C8")
        canvas.create_text(
            left,
            height - 14,
            text="last 160 samples",
            fill=SUBTLE,
            anchor="w",
            font=(FONT_UI, 10),
        )
        canvas.create_text(
            right,
            height - 14,
            text=f"target {self.settings.target_camera_fps:.0f} fps",
            fill=SUBTLE,
            anchor="e",
            font=(FONT_UI, 10),
        )

    def _stop_capture(self) -> None:
        if not self._producer:
            return
        self._producer.stop()
        self._producer.join(timeout=2.0)
        self._producer = None

    def _handle_camera_error(self, exc: BaseException) -> None:
        def update() -> None:
            if self._closed:
                return
            self.listening_var.set("disabled")
            self._set_status("error")
            self._log_event(str(exc))
            self._update_controls()

        self.root.after(0, update)

    def _update_controls(self) -> None:
        running = bool(self._producer and self._producer.is_running)
        if self.start_button is not None:
            self.start_button.set_enabled(not running and not self.emergency_disabled)
        if self.stop_button is not None:
            self.stop_button.set_enabled(running)
        if self.emergency_button is not None:
            self.emergency_button.set_enabled(not self.emergency_disabled)

    def _set_status(self, status: str) -> None:
        self.status_var.set(status)

    def _log_event(self, message: str) -> None:
        self.event_var.set(message)

    def _section_label(self, parent: tk.Misc, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=BG, fg=INK, font=(FONT_UI, 16, "bold"))

    def _button(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], None],
        *,
        variant: str,
    ) -> ActionButton:
        return ActionButton(parent, text, command, variant=variant)

    def _metric(
        self,
        parent: tk.Misc,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
    ) -> None:
        tile = tk.Frame(
            parent,
            bg=ELEVATED,
            highlightthickness=1,
            highlightbackground=LINE,
            padx=12,
            pady=11,
        )
        tile.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
        tk.Label(tile, text=label, bg=ELEVATED, fg=MUTED, font=(FONT_UI, 12, "bold")).pack(anchor="w")
        tk.Label(tile, textvariable=variable, bg=ELEVATED, fg=INK, font=(FONT_UI, 24, "bold")).pack(anchor="w")

    def _legend_item(self, parent: tk.Misc, label: str, color: str) -> tk.Frame:
        item = tk.Frame(parent, bg=SURFACE)
        swatch = tk.Canvas(item, width=18, height=10, bg=SURFACE, bd=0, highlightthickness=0)
        swatch.pack(side="left", padx=(0, 6))
        swatch.create_line(1, 5, 17, 5, fill=color, width=3)
        tk.Label(item, text=label, bg=SURFACE, fg=MUTED, font=(FONT_UI, 12, "bold")).pack(side="left")
        return item


def run_app(
    settings: Settings,
    *,
    capture_factory: CaptureFactory | None = None,
    logger: logging.Logger | None = None,
    root_factory: Callable[[], tk.Tk] = tk.Tk,
) -> None:
    root = root_factory()
    AirDeckApp(root, settings, capture_factory=capture_factory, logger=logger)
    root.mainloop()
