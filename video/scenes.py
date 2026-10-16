"""Manim scenes. Full-frame: Planning, LandingCard, EndCard. Transparent overlays: HookTag, JumpOverlay,
HoldOverlay, LandingOverlay. Each reads its section length from audio/sections.json.

Render: manim render -r 1920,1080 --fps 30 [-t] video/scenes.py <Scene>
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from manim import (BLUE, DOWN, GREY_B, GREY_D, LEFT, ORANGE, RED_B, RIGHT, UP, WHITE, YELLOW, Arrow, Axes,
                   Circle, Create, DashedLine, DoubleArrow, FadeIn, FadeOut, Line, Polygon, Rectangle, Scene, Tex,
                   ValueTracker, VGroup, VMobject, Write, always_redraw, config, linear, rate_functions)

from roadster import params as P
from video import facts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AUDIO = ROOT / "audio"
FRAME_W, FRAME_H = 14.222, 8.0
PX_W, PX_H = 1920, 1080


def section(name):
    for s in json.loads((AUDIO / "sections.json").read_text()):
        if s["name"] == name:
            return s
    raise KeyError(name)


def clip_timing(name):
    """Sim-time window and slow-motion factor each sim clip uses (shared with assemble.py)."""
    return json.loads((DATA / "clip_timing.json").read_text())[name]


def label(text, size=36, color=WHITE):
    return Tex(text, font_size=size, color=color)


class TimedScene(Scene):
    """Helper: wait until an absolute time in the scene."""

    def until(self, t):
        if t > self.renderer.time + 1e-3:
            self.wait(t - self.renderer.time)


# ---------------------------------------------------------------- full-frame scenes

class Planning(TimedScene):
    def construct(self):
        from roadster.trajopt import Plan
        sec = section("planning")
        D, st = sec["duration"], sec["sentences"]
        plan = Plan.load(DATA / "kickflip_plan.npz")
        T = plan.T
        omega = plan.X[:, 5]
        spin = np.abs(omega) > 0.5 * np.abs(omega).max()
        t_start, t_stop = plan.t[np.argmax(spin)], plan.t[len(spin) - 1 - np.argmax(spin[::-1])]

        def glyph_at(t, center, scale):
            """The car seen from behind, rolled to the plan's angle, with its lift (blue) and roof (orange) jets."""
            x, u = plan.at(t)
            body = Rectangle(width=1.9, height=0.42, stroke_color=WHITE, stroke_width=2, fill_color=GREY_D, fill_opacity=1)
            cabin = Polygon([-0.55, 0.21, 0], [0.55, 0.21, 0], [0.38, 0.5, 0], [-0.38, 0.5, 0],
                            stroke_color=WHITE, stroke_width=2, fill_color="#1b1d22", fill_opacity=1)
            # seen from behind, a tire shows its tread: a dark rectangle (tread width x tire diameter),
            # drawn behind the body so the top of each tire is hidden
            wl = Rectangle(width=0.3, height=0.62, stroke_color=GREY_B, stroke_width=2, fill_color="#111214",
                           fill_opacity=1).move_to([-0.76, -0.26, 0])
            wr = wl.copy().move_to([0.76, -0.26, 0])
            jets = VGroup()  # viewed from behind: the car's left (+y) is screen left
            for side, xs in ((0, -0.75), (1, 0.75)):
                lift = u[side] / (2 * P.F_LIFT_MAX)
                roof = u[2 + side] / (2 * P.F_ROOF_MAX)
                if lift > 0.05:
                    jets.add(Line([xs, -0.25, 0], [xs, -0.25 - 0.9 * lift, 0], color=BLUE, stroke_width=10 * lift + 2))
                if roof > 0.05:
                    jets.add(Line([xs, 0.25, 0], [xs, 0.25 + 0.7 * roof, 0], color=ORANGE, stroke_width=8 * roof + 2))
            g = VGroup(jets, wl, wr, body, cabin)
            g.rotate(-x[4], about_point=np.zeros(3))  # positive roll lifts the left side
            return g.scale(scale, about_point=np.zeros(3)).shift(center)

        # intro: the spin itself, big, while the narration says it must start and stop on time
        t_intro = max(2.5, st[2][1] - 1.9)
        ti = ValueTracker(0.0)
        big = always_redraw(lambda: glyph_at(ti.get_value(), np.array([0, -0.2, 0]), 2.0))
        big_title = label("seen from behind", 34, GREY_B).move_to(UP * 2.6)
        self.add(big)
        self.play(FadeIn(big_title), run_time=0.3)
        rate_i = T / (t_intro - 0.3)
        for t_mark, text in ((t_start, "start the spin"), (t_stop, "stop the spin")):
            self.play(ti.animate.set_value(t_mark), run_time=(t_mark - ti.get_value()) / rate_i, rate_func=linear)
            note = label(text, 40, YELLOW).move_to(DOWN * 2.6)
            self.add(note)
            self.wait(0.8)
            self.remove(note)
        self.play(ti.animate.set_value(T), run_time=max(0.1, t_intro - self.renderer.time), rate_func=linear)

        ax = Axes(x_range=[-2, 56, 10], y_range=[0, 9, 3], x_length=12.4, y_length=2.5,
                  axis_config={"color": GREY_B, "include_tip": False, "stroke_width": 2},
                  x_axis_config={"numbers_to_include": [0, 10, 20, 30, 40, 50], "font_size": 30},
                  y_axis_config={"numbers_to_include": [3, 6, 9], "font_size": 30}).to_edge(UP, buff=0.4)
        xl = label("distance (m)", 30, GREY_B).next_to(ax.x_axis, DOWN, buff=0.45).align_to(ax.x_axis, RIGHT)
        yl = label("height (m)", 30, GREY_B).next_to(ax.y_axis, UP, buff=0.1)
        slot = P.ROW_LEN / 5
        heights = [1.5, 1.9, P.ROW_HEIGHT, 2.0, 1.6]
        halfx = [1.0, 1.0, 1.3, 1.0, 1.0]
        obst = VGroup(*[Polygon(ax.c2p(P.ROW_START + (i + .5) * slot - hx, 0), ax.c2p(P.ROW_START + (i + .5) * slot + hx, 0),
                                ax.c2p(P.ROW_START + (i + .5) * slot + hx, h), ax.c2p(P.ROW_START + (i + .5) * slot - hx, h),
                                stroke_width=0, fill_color=GREY_D, fill_opacity=1)
                        for i, (h, hx) in enumerate(zip(heights, halfx))])
        self.play(FadeOut(big), FadeOut(big_title), Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(obst), run_time=0.8)

        def path(X, T_i, color, width=3):
            x = P.VX0 * np.linspace(0, T_i, len(X))
            v = VMobject(stroke_color=color, stroke_width=width)
            v.set_points_smoothly([ax.c2p(xi, zi) for xi, zi in zip(x, X[:, 1])])
            return v

        # optimizer iterates converge
        self.until(st[2][1] - 0.3)
        its = list(zip(plan.iterates, plan.iterates_T))
        cur = path(*its[0], GREY_B)
        it_label = label("guess 1", 34, GREY_B).next_to(ax.x_axis, DOWN, buff=0.45).to_edge(LEFT, buff=0.8)
        self.play(Create(cur), FadeIn(it_label), run_time=0.8)
        t_conv = max(4.0, 0.5 * (st[3][2] - self.renderer.time))
        per = t_conv / (len(its) - 1)
        for k, (X, T_i) in enumerate(its[1:], start=2):
            nxt = path(X, T_i, GREY_B)
            nxt.set_color(interpolate_hex(GREY_B, BLUE, rate_functions.smooth(k / len(its))))
            new_label = label(f"guess {k}", 34, GREY_B).move_to(it_label, aligned_edge=LEFT)
            self.play(cur.animate.become(nxt), run_time=per, rate_func=linear)
            self.remove(it_label)
            it_label = new_label
            self.add(it_label)
        final = path(plan.X, plan.T, BLUE, 5)
        done = label(f"final plan: lands upright, uses {round(100 * facts.air_fraction(plan))}\\% of the air", 34,
                     BLUE).move_to(it_label, aligned_edge=LEFT)
        self.play(cur.animate.become(final), FadeOut(it_label), FadeIn(done), run_time=0.6)

        # car moving along the plan, roll inset and left/right jet timeline
        tt = ValueTracker(0.0)
        dot = always_redraw(lambda: Circle(radius=0.1, color=YELLOW, fill_opacity=1).move_to(
            ax.c2p(P.VX0 * tt.get_value(), plan.at(tt.get_value())[0][1])))
        inset_c = np.array([5.2, -1.75, 0])
        glyph = always_redraw(lambda: glyph_at(tt.get_value(), inset_c, 1.0))

        tl = Axes(x_range=[0, T, 0.5], y_range=[0, 32, 10], x_length=6.0, y_length=1.2,
                  axis_config={"color": GREY_B, "include_tip": False, "stroke_width": 2}).move_to([-3.0, -1.7, 0])
        tl_lab = label("jet force", 28, GREY_B).rotate(np.pi / 2).next_to(tl, LEFT, buff=0.2)
        leg_l = label("left jets", 28, BLUE).next_to(tl, RIGHT, buff=0.25).shift(UP * 0.35)
        leg_r = label("right jets", 28, YELLOW).next_to(leg_l, DOWN, buff=0.15).align_to(leg_l, LEFT)
        left_kN = (plan.U[:, 0] + plan.U[:, 2]) / 1e3
        right_kN = (plan.U[:, 1] + plan.U[:, 3]) / 1e3

        def series(vals, color):
            def f():
                n = max(2, int(np.searchsorted(plan.t[:-1], tt.get_value())) + 1)
                pts = []
                for k in range(min(n, len(vals))):
                    pts += [tl.c2p(plan.t[k], vals[k]), tl.c2p(plan.t[k + 1], vals[k])]
                return VMobject(stroke_color=color, stroke_width=3).set_points_as_corners(pts)
            return always_redraw(f)

        self.play(FadeIn(dot), FadeIn(glyph), Create(tl), FadeIn(tl_lab), FadeIn(leg_l),
                  FadeIn(leg_r), run_time=0.8)
        self.add(series(left_kN, BLUE), series(right_kN, YELLOW))
        rate = T / (D - self.renderer.time - 0.6)
        for t_mark, text in ((t_start, "spin starts"), (t_stop, "spin stops")):
            self.play(tt.animate.set_value(t_mark), run_time=(t_mark - tt.get_value()) / rate, rate_func=linear)
            self.add(VGroup(DashedLine(tl.c2p(t_mark, 0), tl.c2p(t_mark, 32), color=WHITE, stroke_width=2),
                            label(text, 28).next_to(tl.c2p(t_mark, 32), UP, buff=0.05)))
        self.play(tt.animate.set_value(T), run_time=max(0.1, (T - tt.get_value()) / rate), rate_func=linear)
        self.until(D)


def interpolate_hex(c1, c2, a):
    from manim import interpolate_color
    return interpolate_color(c1, c2, a)


class LandingCard(TimedScene):
    def construct(self):
        sec = section("landing")
        st = sec["sentences"]
        D = sec["duration"] - st[2][1] + 0.0
        title = Tex("TinyMPC", font_size=96, color=YELLOW)
        from roadster.flight import Log
        med, worst = facts.solve_ms([Log.load(DATA / f"log_{s}_cl.npz") for s in ("nominal", "gust", "fast_launch")])
        l1 = label(f"one landing solve in this video: {med:.2f} ms (laptop)", 34)
        l2 = label("open source, built for small computers like microcontrollers", 34, GREY_B)
        l2b = label("related: Conic-TinyMPC adds cone constraints, as in rocket landing", 34, GREY_B)
        l3 = label("tinympc.org", 30, BLUE)
        g = VGroup(title, l1, l2, l2b, l3).arrange(DOWN, buff=0.4)
        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(l1, shift=UP * 0.2), run_time=0.5)
        self.play(FadeIn(l2, shift=UP * 0.2), run_time=0.5)
        self.play(FadeIn(l2b, shift=UP * 0.2), run_time=0.5)
        self.play(FadeIn(l3), run_time=0.4)
        self.until(D)


class EndCard(TimedScene):
    def construct(self):
        from manim import RESAMPLING_ALGORITHMS, ImageMobject

        from video.qr import REPO_LABEL, make_qr
        sec = section("end")
        D = sec["duration"]
        rows = VGroup(*[VGroup(label(a, 56, c), label(b, 32, GREY_B)).arrange(RIGHT, buff=0.4, aligned_edge=DOWN)
                        for a, b, c in (("plan", "trajectory optimization", BLUE),
                                        ("correct", "LQR feedback", WHITE),
                                        ("land", "model predictive control (TinyMPC)", YELLOW))])
        rows.arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([-2.6, 1.0, 0])
        qr = ImageMobject(make_qr(str(ROOT / "out" / "repo_qr.png")))
        qr.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        qr.height = 3.2
        qr.move_to([3.6, 1.1, 0])
        link = label(REPO_LABEL, 28, BLUE).next_to(qr, DOWN, buff=0.2)
        code = label("code", 30, GREY_B).next_to(qr, UP, buff=0.15)
        credit = VGroup(label("Thanks to Tesla and SpaceX for an amazing demo (Oct 15, 2026).", 26, GREY_B),
                        label("My own implementation, not Tesla's code. Numbers are estimates.", 26, GREY_B),
                        label("Khai Nguyen", 30, WHITE)).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=1.25)
        self.play(FadeIn(qr), FadeIn(code), FadeIn(link), run_time=0.4)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.3), run_time=0.4)
        self.play(FadeIn(credit), run_time=0.5)
        self.until(D)


# ---------------------------------------------------------------- transparent overlays

def side_projector():
    """World point -> manim frame coordinates for the model's fixed 'side' camera."""
    import mujoco
    from roadster.model import load
    m, d = load()
    cid = m.camera("side").id
    pos, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
    f = (PX_H / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)

    def proj(p):
        v = np.asarray(p, float) - pos
        xc, yc, zc = v @ R[:, 0], v @ R[:, 1], -(v @ R[:, 2])
        u, w = PX_W / 2 + f * xc / zc, PX_H / 2 - f * yc / zc
        return np.array([(u / PX_W - 0.5) * FRAME_W, (0.5 - w / PX_H) * FRAME_H, 0])
    return proj


class HookTag(TimedScene):
    def construct(self):
        D = section("hook")["duration"]
        divider = Line([0, FRAME_H / 2, 0], [0, -FRAME_H / 2, 0], color=GREY_D, stroke_width=4)
        lt = label("kickflip", 44, WHITE).move_to([-FRAME_W / 4, 3.4, 0])
        rt = label("ramp", 44, WHITE).move_to([FRAME_W / 4, 3.4, 0])
        tag = label("my simulation in MuJoCo", 30, GREY_B).move_to([0, 2.75, 0])
        tag.add_background_rectangle(color="#000000", opacity=0.85, buff=0.12)
        self.add(divider, lt, rt, tag)
        self.until(D)


class RampTag(TimedScene):
    def construct(self):
        D = section("ramp")["duration"]
        ct = clip_timing("ramp")
        t_lip = (0.0 - ct["t0"]) / ct["slowmo"]  # the car leaves the lip at launch time
        tag = label("the ramp stunt: same plan, feedback and landing", 34, GREY_B).to_corner(UP + LEFT, buff=0.45)
        suction = label("fan suction holds it on (Tesla's claim)", 36, YELLOW).move_to([-2.4, 1.6, 0])
        suction.add_background_rectangle(color="#000000", opacity=0.75, buff=0.12)
        self.play(FadeIn(tag), run_time=0.5)
        self.play(FadeIn(suction), run_time=0.3)
        self.until(max(self.renderer.time, t_lip - 0.2))
        self.play(FadeOut(suction), run_time=0.3)
        self.until(D)


class SemiTag(TimedScene):
    def construct(self):
        from roadster.flight import Log, touchdown_metrics
        D = section("semi")["duration"]
        ct = clip_timing("semi")
        log = Log.load(DATA / "log_semi_cl.npz")
        vz, y_cm = facts.semi_touchdown(log)
        t_td = (touchdown_metrics(log)["touchdown_t"] - ct["t0"]) / ct["slowmo"]
        tag = label("my own stunt: onto a moving semi", 34, GREY_B).to_corner(UP + LEFT, buff=0.45)
        width = label(f"trailer {P.DECK_W:.1f} m wide, car {2 * 0.98:.2f} m", 32, WHITE).to_corner(UP + RIGHT, buff=0.45)
        width.add_background_rectangle(color="#000000", opacity=0.75, buff=0.1)
        self.play(FadeIn(tag), FadeIn(width), run_time=0.5)
        self.until(max(self.renderer.time, t_td + 0.3))
        result = label(f"lands at {vz:.1f} m/s, {y_cm:.0f} cm off center", 40, BLUE).move_to(UP * 2.2)
        result.add_background_rectangle(color="#000000", opacity=0.75, buff=0.12)
        self.play(FadeIn(result), run_time=0.4)
        self.until(D)


class JumpOverlay(TimedScene):
    def construct(self):
        from roadster.flight import Log
        D = section("jump")["duration"]
        ct = clip_timing("jump")
        log = Log.load(DATA / "log_nominal_cl.npz")
        tp = log.t - log.launch_t
        proj = side_projector()

        def tau(t_sim):
            return (t_sim - ct["t0"]) / ct["slowmo"]

        def pos_at(t_sim):
            i = int(np.clip(np.searchsorted(tp, t_sim), 0, len(tp) - 1))
            return log.qpos[i, :3]

        from roadster.trajopt import Plan
        flight = np.isin(log.phase, ["flight", "landing"])
        burn_end = facts.burn_end(Plan.load(DATA / "kickflip_plan.npz"))
        apex_i = int(np.argmax(np.where(flight, log.qpos[:, 2], -1)))
        land_i = int(np.flatnonzero(log.phase == "rollout")[0])
        t_apex, t_land = tp[apex_i], facts.apex_and_airtime(log)[1]
        apex = log.qpos[apex_i, :3]

        note = label("numbers are my estimates", 32, GREY_B).to_corner(UP + RIGHT, buff=0.45)
        self.add(note)

        self.until(tau(burn_end))
        p = proj(pos_at(burn_end))
        off = label("jets off", 40, BLUE).move_to(p + np.array([-0.2, 1.0, 0]))
        self.play(FadeIn(off, shift=DOWN * 0.2), run_time=0.4)

        self.until(tau(t_apex))
        top, base = proj(apex), proj([apex[0], 0, 0])
        h = DoubleArrow(base, top, buff=0, color=YELLOW, stroke_width=3, tip_length=0.18)
        hl = label(f"{apex[2]:.1f} m", 40, YELLOW).next_to(h, RIGHT, buff=0.15)
        self.play(FadeOut(off), Create(h), FadeIn(hl), run_time=0.6)

        self.until(tau(t_land) + 0.3)
        a, b = proj([0, 0, 0]), proj(log.qpos[land_i, :3] * np.array([1, 1, 0]))
        span = DoubleArrow(a + DOWN * 0.35, b + DOWN * 0.35, buff=0, color=WHITE, stroke_width=3, tip_length=0.18)
        sl = label(f"{t_land:.1f} s in the air", 40).next_to(span, DOWN, buff=0.15)
        self.play(Create(span), FadeIn(sl), run_time=0.6)
        self.until(D)


class HoldOverlay(TimedScene):
    def construct(self):
        from roadster.flight import Log
        D = section("hold")["duration"]
        ct = clip_timing("hold")
        from roadster.flight import MODEL_ERROR, touchdown_metrics
        log = Log.load(DATA / "log_gust_model_error_cl.npz")
        log_ol = Log.load(DATA / "log_gust_model_error_ol.npz")
        tp = log.t - log.launch_t
        g = np.flatnonzero(log.gust_on)
        g0, g1 = tp[g[0]], tp[g[-1]]
        tau = lambda t: (t - ct["t0"]) / ct["slowmo"]

        left_c, right_c = np.array([-FRAME_W / 4, 0, 0]), np.array([FRAME_W / 4, 0, 0])
        divider = Line([0, FRAME_H / 2, 0], [0, -FRAME_H / 2, 0], color=GREY_D, stroke_width=4)
        lt = label("same jet timing, no feedback", 44, RED_B).move_to(left_c + UP * 3.3)
        rt = label("plan + LQR feedback", 44, BLUE).move_to(right_c + UP * 3.3)
        err = MODEL_ERROR
        both = label(f"both cars: {round(100 * (err['mass'] - 1))}\\% heavier, jets {round(100 * (1 - err['thrust']))}\\% weaker, "
                     f"weight {round(100 * err['com_y'])} cm off-center", 36, WHITE).move_to(UP * 2.5)
        both.add_background_rectangle(color="#000000", opacity=0.9, buff=0.15)
        self.add(divider)
        self.play(FadeIn(lt), FadeIn(rt), FadeIn(both), run_time=0.5)

        self.until(tau(g0))
        arrows = VGroup()
        for c in (left_c, right_c):
            arr = Arrow(c + np.array([-3.0, 0.9, 0]), c + np.array([-0.9, 0.2, 0]), color=WHITE, stroke_width=6,
                        buff=0)
            arrows.add(arr, label("gust", 40).next_to(arr.get_start(), UP, buff=0.1))
        self.play(FadeIn(arrows), run_time=0.25)
        self.until(tau(g1) + 0.8)
        self.play(FadeOut(arrows), run_time=0.4)

        land = np.flatnonzero(log.phase == "rollout")[0]
        self.until(tau(tp[land]) + 0.6)
        lo = label(f"hits the ground at {touchdown_metrics(log_ol)['touchdown_vz']:.0f} m/s", 44, RED_B).move_to(left_c + DOWN * 1.9)
        ro = label(f"lands at {touchdown_metrics(log)['touchdown_vz']:.1f} m/s", 44, BLUE).move_to(right_c + DOWN * 1.9)
        self.play(FadeIn(lo), FadeIn(ro), run_time=0.5)
        self.until(D)


class LandingOverlay(TimedScene):
    def construct(self):
        sec = section("landing")
        D = sec["sentences"][2][1]  # the TinyMPC card takes over at the third sentence
        leg_dot = Circle(radius=0.09, color=YELLOW, fill_opacity=1)
        leg = VGroup(leg_dot, label("predicted fall, updated 50 times a second", 38, YELLOW)).arrange(RIGHT, buff=0.2)
        leg.to_corner(UP + LEFT, buff=0.5)
        self.play(FadeIn(leg), run_time=0.5)
        self.until(D)
