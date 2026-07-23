"""
Physics Concept Lab
An interactive tool for building intuition in classical mechanics and
continuum systems, paired with an AI tutor that explains the "why" behind
the motion.

Author: Tabassum Tariq

Everything (physics simulations, AI tutor, and UI) lives in this single
file for simplicity -- no other .py files are needed alongside it.
"""

import os
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# =============================================================================
# PHYSICS ENGINE
# Core simulation functions. Each "time series" function returns
# t, y, info. Field (PDE) functions return x, t, U, info.
# =============================================================================


def simple_harmonic_oscillator(mass=1.0, k=10.0, x0=1.0, v0=0.0, t_max=10.0, n_points=500):
    """Undamped mass-spring system: m*x'' + k*x = 0"""
    omega = np.sqrt(k / mass)
    t = np.linspace(0, t_max, n_points)
    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)

    info = {
        "system": "Simple Harmonic Oscillator",
        "angular_frequency": omega,
        "period": 2 * np.pi / omega,
        "amplitude": np.sqrt(x0**2 + (v0 / omega) ** 2),
    }
    return t, x, info


def simple_pendulum(length=1.0, theta0_deg=15.0, omega0=0.0, t_max=10.0, n_points=500, g=9.81, small_angle=True):
    """
    Simple pendulum. If small_angle=True, uses the linear approximation.
    Otherwise solves the full nonlinear ODE: theta'' + (g/L) sin(theta) = 0
    """
    theta0 = np.radians(theta0_deg)
    t = np.linspace(0, t_max, n_points)

    if small_angle:
        omega_n = np.sqrt(g / length)
        theta = theta0 * np.cos(omega_n * t) + (omega0 / omega_n) * np.sin(omega_n * t)
    else:
        def deriv(state, t):
            th, om = state
            return [om, -(g / length) * np.sin(th)]

        sol = odeint(deriv, [theta0, omega0], t)
        theta = sol[:, 0]
        omega_n = np.sqrt(g / length)  # small-angle approx frequency, for reference

    info = {
        "system": "Simple Pendulum",
        "small_angle_approximation": small_angle,
        "angular_frequency_small_angle": omega_n,
        "period_small_angle": 2 * np.pi / omega_n,
        "initial_angle_deg": theta0_deg,
    }
    return t, np.degrees(theta), info


def damped_harmonic_oscillator(mass=1.0, k=10.0, c=1.0, x0=1.0, v0=0.0, t_max=10.0, n_points=500):
    """
    Damped mass-spring system: m*x'' + c*x' + k*x = 0
    Classifies the system as underdamped, critically damped, or overdamped.
    """
    omega0 = np.sqrt(k / mass)
    zeta = c / (2 * np.sqrt(mass * k))  # damping ratio
    t = np.linspace(0, t_max, n_points)

    if zeta < 1:
        regime = "underdamped"
        omega_d = omega0 * np.sqrt(1 - zeta**2)
        A = x0
        B = (v0 + zeta * omega0 * x0) / omega_d
        x = np.exp(-zeta * omega0 * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    elif np.isclose(zeta, 1):
        regime = "critically damped"
        A = x0
        B = v0 + omega0 * x0
        x = (A + B * t) * np.exp(-omega0 * t)
    else:
        regime = "overdamped"
        r1 = -omega0 * (zeta - np.sqrt(zeta**2 - 1))
        r2 = -omega0 * (zeta + np.sqrt(zeta**2 - 1))
        A = (v0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)

    info = {
        "system": "Damped Harmonic Oscillator",
        "natural_frequency": omega0,
        "damping_ratio": zeta,
        "regime": regime,
    }
    return t, x, info


def projectile_motion(v0=20.0, angle_deg=45.0, g=9.81, mass=1.0,
                       drag_coeff=0.0, n_points=400):
    """
    Projectile motion, optionally with quadratic air drag.
    Returns x(t), y(t) trajectory arrays, plus info. Stops automatically
    when the projectile returns to y = 0 (ground impact).
    """
    theta = np.radians(angle_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    def deriv(t, state):
        x, y, vx, vy = state
        speed = np.hypot(vx, vy)
        drag_ax = -(drag_coeff / mass) * speed * vx
        drag_ay = -(drag_coeff / mass) * speed * vy
        return [vx, vy, drag_ax, drag_ay - g]

    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    t_upper_bound = 3 * (2 * vy0 / g) if vy0 > 0 else 10.0
    t_eval = np.linspace(0, t_upper_bound, n_points)

    sol = solve_ivp(
        deriv, [0, t_upper_bound], [0, 0, vx0, vy0],
        t_eval=t_eval, events=hit_ground, max_step=t_upper_bound / n_points,
    )

    t = sol.t
    x, y = sol.y[0], sol.y[1]

    range_no_drag = (v0**2) * np.sin(2 * theta) / g
    t_flight_no_drag = 2 * vy0 / g
    max_height_no_drag = (vy0**2) / (2 * g)

    info = {
        "system": "Projectile Motion",
        "has_air_resistance": drag_coeff > 0,
        "actual_range": x[-1],
        "actual_time_of_flight": t[-1],
        "max_height": y.max(),
        "no_drag_range_reference": range_no_drag,
        "no_drag_time_of_flight_reference": t_flight_no_drag,
        "no_drag_max_height_reference": max_height_no_drag,
    }
    return t, x, y, info


def double_pendulum(L1=1.0, L2=1.0, m1=1.0, m2=1.0,
                     theta1_0_deg=120.0, theta2_0_deg=-10.0,
                     g=9.81, t_max=15.0, n_points=1500):
    """
    Full nonlinear double pendulum. Chaotic for large initial angles --
    two runs with almost-identical starting angles diverge completely.
    """
    theta1_0 = np.radians(theta1_0_deg)
    theta2_0 = np.radians(theta2_0_deg)

    def deriv(t, state):
        th1, om1, th2, om2 = state
        delta = th1 - th2

        denom1 = L1 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
        domega1 = (
            -g * (2 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2 * th2)
            - 2 * np.sin(delta) * m2 * (om2**2 * L2 + om1**2 * L1 * np.cos(delta))
        ) / denom1

        denom2 = L2 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
        domega2 = (
            2 * np.sin(delta) * (
                om1**2 * L1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(th1)
                + om2**2 * L2 * m2 * np.cos(delta)
            )
        ) / denom2

        return [om1, domega1, om2, domega2]

    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(
        deriv, [0, t_max], [theta1_0, 0.0, theta2_0, 0.0],
        t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-8,
    )

    theta1, theta2 = sol.y[0], sol.y[2]

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    info = {
        "system": "Double Pendulum",
        "chaotic": True,
        "initial_angle_1_deg": theta1_0_deg,
        "initial_angle_2_deg": theta2_0_deg,
        "note": "Small changes in initial angles lead to completely different trajectories after a few seconds.",
    }
    return sol.t, np.degrees(theta1), np.degrees(theta2), x2, y2, info


def wave_equation_1d(length=1.0, wave_speed=1.0, n_x=200, t_max=2.0, n_t=200,
                      pulse_center=0.5, pulse_width=0.05, method="finite_difference"):
    """
    Solves u_tt = c^2 * u_xx for a plucked string / pulse, both ends fixed
    (finite_difference) or periodic (spectral).
    Returns x, t, U (shape n_t x n_x), info
    """
    x = np.linspace(0, length, n_x)
    dx = x[1] - x[0]
    dt = t_max / (n_t - 1)

    if method == "spectral":
        # Spectral second derivative resolves wavenumbers up to the Nyquist
        # limit, a tighter stability constraint than the standard FD CFL.
        k_max = np.pi / dx
        dt_stable = 1.6 / (wave_speed * k_max)
        if dt > dt_stable:
            n_t = int(np.ceil(t_max / dt_stable)) + 1
            dt = t_max / (n_t - 1)
        cfl = wave_speed * dt / dx
    else:
        cfl = wave_speed * dt / dx
        if cfl > 0.98:
            n_t = int(np.ceil(t_max * wave_speed / (0.9 * dx))) + 1
            dt = t_max / (n_t - 1)
            cfl = wave_speed * dt / dx

    t = np.linspace(0, t_max, n_t)
    u0 = np.exp(-((x - pulse_center * length) ** 2) / (2 * pulse_width**2))

    U = np.zeros((n_t, n_x))
    U[0] = u0

    if method == "spectral":
        k = 2 * np.pi * np.fft.fftfreq(n_x, d=dx)

        def d2dx2(u):
            return np.real(np.fft.ifft(-(k**2) * np.fft.fft(u)))

        U[1] = u0 + 0.5 * (wave_speed * dt) ** 2 * d2dx2(u0)
        for n in range(1, n_t - 1):
            U[n + 1] = 2 * U[n] - U[n - 1] + (wave_speed * dt) ** 2 * d2dx2(U[n])
    else:
        r2 = (wave_speed * dt / dx) ** 2
        U[1, 1:-1] = u0[1:-1] + 0.5 * r2 * (u0[2:] - 2 * u0[1:-1] + u0[:-2])
        U[1, 0] = U[1, -1] = 0.0
        for n in range(1, n_t - 1):
            U[n + 1, 1:-1] = (
                2 * U[n, 1:-1] - U[n - 1, 1:-1]
                + r2 * (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2])
            )
            U[n + 1, 0] = U[n + 1, -1] = 0.0  # fixed ends

    info = {
        "system": "1D Wave Equation",
        "method": method,
        "wave_speed": wave_speed,
        "cfl_number": cfl,
        "boundary_condition": "periodic" if method == "spectral" else "fixed ends (Dirichlet)",
    }
    return x, t, U, info


def heat_diffusion_1d(length=1.0, alpha=0.01, n_x=100, t_max=5.0, n_t=200,
                       hotspot_center=0.5, hotspot_width=0.05):
    """
    Solves u_t = alpha * u_xx using the explicit FTCS scheme.
    Fixed zero-temperature boundaries. Initial condition: a Gaussian hot spot.
    Returns x, t, U (shape n_t x n_x), info
    """
    x = np.linspace(0, length, n_x)
    dx = x[1] - x[0]
    dt = t_max / (n_t - 1)

    r = alpha * dt / dx**2
    if r > 0.5:
        n_t = int(np.ceil(t_max * alpha / (0.4 * dx**2))) + 1
        dt = t_max / (n_t - 1)
        r = alpha * dt / dx**2

    t = np.linspace(0, t_max, n_t)
    u0 = np.exp(-((x - hotspot_center * length) ** 2) / (2 * hotspot_width**2))

    U = np.zeros((n_t, n_x))
    U[0] = u0
    for n in range(n_t - 1):
        U[n + 1, 1:-1] = U[n, 1:-1] + r * (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2])
        U[n + 1, 0] = U[n + 1, -1] = 0.0

    info = {
        "system": "1D Heat Diffusion",
        "diffusivity_alpha": alpha,
        "stability_parameter_r": r,
        "stable": r <= 0.5,
    }
    return x, t, U, info


def rlc_circuit(L=1.0, R=1.0, C=0.1, q0=1.0, i0=0.0, t_max=10.0, n_points=500):
    """
    Series RLC circuit (source-free): L*q'' + R*q' + q/C = 0
    Directly analogous to the damped mass-spring system:
    L <-> mass, R <-> damping, 1/C <-> spring constant.
    """
    omega0 = 1.0 / np.sqrt(L * C)
    zeta = (R / 2.0) * np.sqrt(C / L)
    t = np.linspace(0, t_max, n_points)

    if zeta < 1:
        regime = "underdamped (oscillatory)"
        omega_d = omega0 * np.sqrt(1 - zeta**2)
        A = q0
        B = (i0 + zeta * omega0 * q0) / omega_d
        q = np.exp(-zeta * omega0 * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    elif np.isclose(zeta, 1):
        regime = "critically damped"
        A = q0
        B = i0 + omega0 * q0
        q = (A + B * t) * np.exp(-omega0 * t)
    else:
        regime = "overdamped"
        r1 = -omega0 * (zeta - np.sqrt(zeta**2 - 1))
        r2 = -omega0 * (zeta + np.sqrt(zeta**2 - 1))
        A = (i0 - r2 * q0) / (r1 - r2)
        B = q0 - A
        q = A * np.exp(r1 * t) + B * np.exp(r2 * t)

    info = {
        "system": "RLC Circuit",
        "resonant_frequency": omega0,
        "damping_ratio": zeta,
        "regime": regime,
    }
    return t, q, info


def orbital_motion(GM=1.0, r0=1.0, speed_factor=1.0, n_points=1000):
    """
    Simplified 2-body problem: central mass fixed at origin, orbiting body
    starts at (r0, 0) moving tangentially with speed = speed_factor * circular_velocity.
    speed_factor=1 -> circular. <1 or between 1 and sqrt(2) -> ellipse.
    >= sqrt(2) -> unbound (parabolic/hyperbolic).
    """
    v_circ = np.sqrt(GM / r0)
    v0 = speed_factor * v_circ
    state0 = [r0, 0.0, 0.0, v0]

    energy = 0.5 * v0**2 - GM / r0
    a = None
    period = None
    if energy < -1e-9:
        orbit_type = "elliptical (bound)"
        a = -GM / (2 * energy)
        period = 2 * np.pi * np.sqrt(a**3 / GM)
        t_max = 1.15 * period
    elif abs(energy) <= 1e-9:
        orbit_type = "parabolic (marginal escape)"
        t_max = 20.0 / v_circ
    else:
        orbit_type = "hyperbolic (unbound)"
        t_max = 12.0 / v_circ

    def deriv(t, state):
        x, y, vx, vy = state
        r = np.hypot(x, y)
        return [vx, vy, -GM * x / r**3, -GM * y / r**3]

    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(deriv, [0, t_max], state0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    x, y = sol.y[0], sol.y[1]

    e = None
    if a is not None:
        L_spec = r0 * v0
        e = np.sqrt(max(0.0, 1 - (L_spec**2) / (GM * a)))

    info = {
        "system": "Orbital Motion",
        "orbit_type": orbit_type,
        "circular_velocity_at_r0": v_circ,
        "initial_speed": v0,
        "specific_orbital_energy": energy,
    }
    if a is not None:
        info["semi_major_axis"] = a
        info["eccentricity"] = e
        info["orbital_period"] = period

    return sol.t, x, y, info


def coupled_oscillators(m1=1.0, m2=1.0, k=10.0, k12=5.0, x1_0=1.0, x2_0=0.0, t_max=10.0, n_points=500):
    """
    Two masses, each attached to a wall by a spring (k), coupled to each
    other by a middle spring (k12).
    m1 x1'' = -k x1 - k12 (x1 - x2)
    m2 x2'' = -k x2 - k12 (x2 - x1)
    """
    def deriv(t, state):
        x1, v1, x2, v2 = state
        a1 = (-k * x1 - k12 * (x1 - x2)) / m1
        a2 = (-k * x2 - k12 * (x2 - x1)) / m2
        return [v1, a1, v2, a2]

    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(deriv, [0, t_max], [x1_0, 0.0, x2_0, 0.0],
                     t_eval=t_eval, rtol=1e-9, atol=1e-9)
    x1, x2 = sol.y[0], sol.y[2]

    m_avg = (m1 + m2) / 2
    omega_sym = np.sqrt(k / m_avg)
    omega_antisym = np.sqrt((k + 2 * k12) / m_avg)

    info = {
        "system": "Coupled Oscillators",
        "symmetric_mode_frequency": omega_sym,
        "antisymmetric_mode_frequency": omega_antisym,
        "note": "Motion beats between the two normal-mode frequencies unless the initial displacement matches a pure mode.",
    }
    return sol.t, x1, x2, info


# =============================================================================
# AI TUTOR
# Calls Groq's free API (OpenAI-compatible format). Get a free key at
# https://console.groq.com/keys -- sign in, no card required.
# The key is read from Streamlit secrets (when deployed) or an environment
# variable (when running locally) -- never hardcoded, never committed.
# =============================================================================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Free, fast Groq model. If this model is ever retired, swap the string for
# another from https://console.groq.com/docs/models
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a patient, encouraging physics tutor helping undergraduate
computational physics students in Pakistan build intuition for classical mechanics
and computational methods.

You will be given:
- The physical system (e.g. Simple Harmonic Oscillator, Pendulum, Damped Oscillator)
- Its current parameters (mass, spring constant, damping, initial conditions, etc.)
- Derived quantities (frequency, period, damping regime, etc.)
- A target explanation level: "Beginner" or "Intermediate"
- Optionally, a specific follow-up question from the student

Your job: explain WHY the system behaves the way it does with these specific
parameters — not just describe the motion, but connect the parameters to the
physical cause of that behavior. If a follow-up question is given, answer that
question directly, still grounded in the actual current parameters.

Rules:
- Beginner level: no differential equations, no jargon without defining it,
  use everyday analogies (e.g. a swing, a car's shock absorber). Assume the
  student is new to computational physics.
- Intermediate level: you may reference the governing equation and concepts
  like angular frequency, damping ratio, or restoring force, but still explain
  the physical intuition, not just the math.
- Always relate the explanation to the ACTUAL numbers given, not generic theory.
- Keep responses under 180 words.
- Never invent parameter values that were not given to you.
"""


def build_user_prompt(system_name, params, info, level, question=None):
    """Formats the current simulation state (and optional follow-up question)
    into a prompt for the model."""
    param_lines = "\n".join(f"- {k}: {v}" for k, v in params.items())
    info_lines = "\n".join(f"- {k}: {v}" for k, v in info.items())

    base = f"""System: {system_name}
Explanation level requested: {level}

Current parameters:
{param_lines}

Derived quantities:
{info_lines}
"""
    if question:
        return base + f"\nStudent's follow-up question: {question}\n\nAnswer this question directly, at the {level} level, following your instructions."
    return base + f"\nExplain why this system behaves the way it does with these specific parameters, at the {level} level, following your instructions."


def get_explanation(system_name, params, info, level="Beginner", question=None):
    """
    Calls the Groq API and returns the tutor's explanation as a string.
    Returns a clear warning message (never crashes) if the API key is
    missing or the call fails. Pass `question` for a follow-up question
    instead of the default "explain this behavior" prompt.
    """
    # Check Streamlit Cloud secrets first, then fall back to a local env variable.
    # st.secrets raises an error (rather than being empty) if no secrets.toml
    # file exists at all -- normal for local dev using an env variable instead.
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ No API key found. Set the GROQ_API_KEY environment variable "
            "(see README for setup instructions) to enable the AI Tutor."
        )

    user_prompt = build_user_prompt(system_name, params, info, level, question=question)

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 350,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            return (
                "⚠️ Rate limit reached (too many requests in a short time). "
                "Groq's free tier allows a generous but limited number of "
                "requests per minute. Wait a bit and try again."
            )
        if response.status_code == 401:
            return "⚠️ Invalid API key. Double-check your GROQ_API_KEY value."
        return f"⚠️ Could not reach the AI tutor right now. ({e})"
    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not reach the AI tutor right now. ({e})"
    except (KeyError, IndexError):
        return "⚠️ The AI tutor returned an unexpected response. Please try again."


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Physics Concept Lab", page_icon="🔬", layout="wide")

# -----------------------------------------------------------------------------
# Custom styling -- a dark, "scientific dashboard" look. Streamlit's dark theme
# already gives a dark background; this layers in card-style containers,
# tighter spacing, and a consistent accent color for headers/metrics.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px; }
    h1 { font-weight: 800; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 700; }
    div[data-testid="stMetric"] {
        background-color: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Physics Concept Lab")
st.caption(
    "A computational physics dashboard — adjust parameters, watch the motion, "
    "and ask the AI tutor to explain *why* it behaves that way."
)

SYSTEMS = [
    "Simple Harmonic Oscillator",
    "Simple Pendulum",
    "Damped Harmonic Oscillator",
    "Projectile Motion",
    "Double Pendulum",
    "1D Wave Equation",
    "1D Heat Diffusion",
    "RLC Circuit",
    "Orbital Motion",
    "Coupled Oscillators",
]

# -----------------------------------------------------------------------------
# Static reference info per system: governing equations (LaTeX), the
# numerical/analytical method used, real-world applications, difficulty,
# and prerequisites. Purely descriptive -- doesn't affect the simulation.
# -----------------------------------------------------------------------------
SYSTEM_INFO = {
    "Simple Harmonic Oscillator": {
        "equations": [r"m\ddot{x} + kx = 0", r"\omega = \sqrt{k/m}", r"T = 2\pi\sqrt{m/k}"],
        "method": "Solved analytically (closed-form cosine/sine solution) — no numerical integration needed.",
        "applications": ["Vibrating guitar strings", "Quartz crystal oscillators", "Small-amplitude building sway"],
        "difficulty": "Beginner",
        "prerequisites": ["Newton's second law", "Basic trigonometry", "Introductory ODEs"],
    },
    "Simple Pendulum": {
        "equations": [r"\ddot{\theta} + \frac{g}{L}\sin\theta = 0",
                      r"\text{Small angle: } \ddot{\theta} + \frac{g}{L}\theta = 0",
                      r"T \approx 2\pi\sqrt{L/g}"],
        "method": "Small-angle case: analytical. Full nonlinear case: solved numerically with SciPy's odeint (LSODA).",
        "applications": ["Grandfather clocks", "Seismometers", "Playground swings"],
        "difficulty": "Beginner–Intermediate",
        "prerequisites": ["Rotational Newton's second law", "Small-angle approximation", "Basic ODEs"],
    },
    "Damped Harmonic Oscillator": {
        "equations": [r"m\ddot{x} + c\dot{x} + kx = 0", r"\zeta = \frac{c}{2\sqrt{mk}}",
                      r"\omega_d = \omega_0\sqrt{1-\zeta^2}\ \ (\zeta<1)"],
        "method": "Solved analytically; the damping ratio ζ determines whether the regime is underdamped, critical, or overdamped.",
        "applications": ["Car suspension systems", "Door closers", "Shock absorbers", "Analog measurement gauges"],
        "difficulty": "Intermediate",
        "prerequisites": ["Simple Harmonic Oscillator", "Characteristic equation / complex roots"],
    },
    "Projectile Motion": {
        "equations": [r"\ddot{x} = -\frac{k}{m}v\,v_x,\quad \ddot{y} = -g-\frac{k}{m}v\,v_y",
                      r"\text{No drag: } y = x\tan\theta - \frac{gx^2}{2v_0^2\cos^2\theta}"],
        "method": "No-drag case has a closed form. With air resistance, solved numerically with SciPy's solve_ivp (RK45) plus a ground-impact stopping event.",
        "applications": ["Ballistics", "Sports science (throwing, kicking)", "Early-phase rocketry trajectories"],
        "difficulty": "Beginner",
        "prerequisites": ["Kinematics", "Vector decomposition"],
    },
    "Double Pendulum": {
        "equations": [r"\text{Coupled nonlinear ODEs in } \theta_1,\theta_2 \text{ (see code for full form)}",
                      r"\text{No general closed-form solution — chaotic for large amplitudes}"],
        "method": "Solved numerically with SciPy's solve_ivp (RK45, tight tolerance rtol=atol=1e-8) since the system has no closed-form solution.",
        "applications": ["Chaos theory demonstrations", "Robotics (double-link arms)", "Teaching sensitivity to initial conditions"],
        "difficulty": "Advanced",
        "prerequisites": ["Lagrangian mechanics (for derivation)", "Nonlinear ODEs", "Chaos / sensitivity to initial conditions"],
    },
    "1D Wave Equation": {
        "equations": [r"\frac{\partial^2 u}{\partial t^2} = c^2\frac{\partial^2 u}{\partial x^2}"],
        "method": "Solved with either an explicit finite-difference (leapfrog) scheme or a Fourier pseudospectral method — selectable in the sidebar.",
        "applications": ["Vibrating strings", "Seismic wave propagation", "Acoustic modeling"],
        "difficulty": "Intermediate–Advanced",
        "prerequisites": ["PDEs", "CFL stability condition", "Fourier transforms (for the spectral method)"],
    },
    "1D Heat Diffusion": {
        "equations": [r"\frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2}"],
        "method": "Solved with the explicit FTCS (Forward-Time Central-Space) finite-difference scheme.",
        "applications": ["Heat conduction in materials", "Pollutant diffusion", "Option pricing (Black–Scholes analogy)"],
        "difficulty": "Intermediate",
        "prerequisites": ["PDEs", "Numerical stability (FTCS requires r = α·dt/dx² ≤ 0.5)"],
    },
    "RLC Circuit": {
        "equations": [r"L\ddot{q} + R\dot{q} + \frac{q}{C} = 0", r"\zeta = \frac{R}{2}\sqrt{C/L}"],
        "method": "Solved analytically — mathematically identical in structure to the damped harmonic oscillator.",
        "applications": ["Radio tuning circuits", "Analog filters", "Power supply transient response"],
        "difficulty": "Intermediate",
        "prerequisites": ["Damped Harmonic Oscillator", "Kirchhoff's voltage law"],
    },
    "Orbital Motion": {
        "equations": [r"\ddot{\vec r} = -\frac{GM}{r^3}\vec r",
                      r"a = -\frac{GM}{2\varepsilon},\quad \varepsilon = \frac{v^2}{2}-\frac{GM}{r}"],
        "method": "Solved numerically with SciPy's solve_ivp (RK45, tight tolerance rtol=atol=1e-9).",
        "applications": ["Satellite trajectory design", "Planetary motion", "Space mission planning"],
        "difficulty": "Intermediate–Advanced",
        "prerequisites": ["Newton's law of gravitation", "Conic sections", "Conservation of energy & angular momentum"],
    },
    "Coupled Oscillators": {
        "equations": [r"m_1\ddot{x}_1 = -kx_1 - k_{12}(x_1-x_2)",
                      r"m_2\ddot{x}_2 = -kx_2 - k_{12}(x_2-x_1)",
                      r"\omega_{sym}=\sqrt{k/m},\quad \omega_{antisym}=\sqrt{(k+2k_{12})/m}"],
        "method": "Solved numerically with SciPy's solve_ivp (RK45, tight tolerance rtol=atol=1e-9).",
        "applications": ["Molecular vibration (normal modes)", "Coupled pendulum clocks", "Mechanical filters"],
        "difficulty": "Intermediate–Advanced",
        "prerequisites": ["Normal modes", "Eigenvalue problems", "Simple Harmonic Oscillator"],
    },
}

# Systems for which we compute and display energy + phase-space plots.
# Chosen because their energy decomposition is clean and pedagogically clear;
# see the README for why the others (chaotic / PDE / trajectory systems) are excluded.
ENERGY_SYSTEMS = {"Simple Harmonic Oscillator", "Damped Harmonic Oscillator", "Simple Pendulum", "Coupled Oscillators"}

st.sidebar.header("Choose a system")
system = st.sidebar.selectbox("System", SYSTEMS)

st.sidebar.header("Parameters")

plot_kind = "time_series"  # default; overridden for field / trajectory systems

if system == "Simple Harmonic Oscillator":
    mass = st.sidebar.slider("Mass (kg)", 0.1, 5.0, 1.0, 0.1)
    k = st.sidebar.slider("Spring constant k (N/m)", 1.0, 50.0, 10.0, 1.0)
    x0 = st.sidebar.slider("Initial displacement x0 (m)", -2.0, 2.0, 1.0, 0.1)
    v0 = st.sidebar.slider("Initial velocity v0 (m/s)", -5.0, 5.0, 0.0, 0.5)
    t, y, info = simple_harmonic_oscillator(mass=mass, k=k, x0=x0, v0=v0)
    params = {"mass (kg)": mass, "spring constant k (N/m)": k, "x0 (m)": x0, "v0 (m/s)": v0}
    ylabel = "Displacement x (m)"

elif system == "Simple Pendulum":
    length = st.sidebar.slider("Length (m)", 0.2, 3.0, 1.0, 0.1)
    theta0 = st.sidebar.slider("Initial angle (degrees)", 1.0, 90.0, 15.0, 1.0)
    small_angle = st.sidebar.checkbox("Use small-angle approximation", value=True)
    t, y, info = simple_pendulum(length=length, theta0_deg=theta0, small_angle=small_angle)
    params = {"length (m)": length, "initial angle (deg)": theta0, "small_angle_approx": small_angle}
    ylabel = "Angle θ (degrees)"

elif system == "Damped Harmonic Oscillator":
    mass = st.sidebar.slider("Mass (kg)", 0.1, 5.0, 1.0, 0.1)
    k = st.sidebar.slider("Spring constant k (N/m)", 1.0, 50.0, 10.0, 1.0)
    c = st.sidebar.slider("Damping coefficient c (Ns/m)", 0.0, 20.0, 1.0, 0.5)
    x0 = st.sidebar.slider("Initial displacement x0 (m)", -2.0, 2.0, 1.0, 0.1)
    t, y, info = damped_harmonic_oscillator(mass=mass, k=k, c=c, x0=x0)
    params = {"mass (kg)": mass, "spring constant k (N/m)": k, "damping c (Ns/m)": c, "x0 (m)": x0}
    ylabel = "Displacement x (m)"

elif system == "Projectile Motion":
    v0 = st.sidebar.slider("Launch speed v0 (m/s)", 1.0, 60.0, 20.0, 1.0)
    angle = st.sidebar.slider("Launch angle (degrees)", 1.0, 89.0, 45.0, 1.0)
    drag_on = st.sidebar.checkbox("Include air resistance", value=False)
    drag_coeff = st.sidebar.slider("Drag coefficient", 0.0, 0.2, 0.05, 0.01) if drag_on else 0.0
    t, x_traj, y_traj, info = projectile_motion(v0=v0, angle_deg=angle, drag_coeff=drag_coeff)
    params = {"v0 (m/s)": v0, "angle (deg)": angle, "drag_coefficient": drag_coeff}
    plot_kind = "trajectory"

elif system == "Double Pendulum":
    L1 = st.sidebar.slider("Length 1 (m)", 0.3, 2.0, 1.0, 0.1)
    L2 = st.sidebar.slider("Length 2 (m)", 0.3, 2.0, 1.0, 0.1)
    theta1_0 = st.sidebar.slider("Initial angle 1 (degrees)", -179.0, 179.0, 120.0, 1.0)
    theta2_0 = st.sidebar.slider("Initial angle 2 (degrees)", -179.0, 179.0, -10.0, 1.0)
    t, th1, th2, x2, y2, info = double_pendulum(
        L1=L1, L2=L2, theta1_0_deg=theta1_0, theta2_0_deg=theta2_0, t_max=15.0
    )
    params = {"L1 (m)": L1, "L2 (m)": L2, "initial angle 1 (deg)": theta1_0, "initial angle 2 (deg)": theta2_0}
    plot_kind = "double_pendulum"

elif system == "1D Wave Equation":
    method = st.sidebar.radio("Method", ["finite_difference", "spectral"], horizontal=True)
    wave_speed = st.sidebar.slider("Wave speed c (m/s)", 0.2, 3.0, 1.0, 0.1)
    pulse_width = st.sidebar.slider("Initial pulse width", 0.02, 0.2, 0.05, 0.01)
    x_arr, t_arr, U, info = wave_equation_1d(
        wave_speed=wave_speed, pulse_width=pulse_width, method=method
    )
    params = {"method": method, "wave_speed (m/s)": wave_speed, "pulse_width": pulse_width}
    plot_kind = "field"

elif system == "1D Heat Diffusion":
    alpha = st.sidebar.slider("Thermal diffusivity α", 0.001, 0.05, 0.01, 0.001)
    hotspot_width = st.sidebar.slider("Initial hot-spot width", 0.02, 0.2, 0.05, 0.01)
    x_arr, t_arr, U, info = heat_diffusion_1d(alpha=alpha, hotspot_width=hotspot_width)
    params = {"diffusivity_alpha": alpha, "hotspot_width": hotspot_width}
    plot_kind = "field"

elif system == "RLC Circuit":
    L = st.sidebar.slider("Inductance L (H)", 0.1, 5.0, 1.0, 0.1)
    C = st.sidebar.slider("Capacitance C (F)", 0.01, 1.0, 0.1, 0.01)
    R = st.sidebar.slider("Resistance R (Ω)", 0.0, 20.0, 1.0, 0.5)
    q0 = st.sidebar.slider("Initial charge q0 (C)", -2.0, 2.0, 1.0, 0.1)
    i0 = st.sidebar.slider("Initial current i0 (A)", -5.0, 5.0, 0.0, 0.5)
    t, y, info = rlc_circuit(L=L, R=R, C=C, q0=q0, i0=i0)
    params = {"L (H)": L, "R (Ω)": R, "C (F)": C, "q0 (C)": q0, "i0 (A)": i0}
    ylabel = "Charge q (C)"

elif system == "Orbital Motion":
    GM = st.sidebar.slider("GM (gravitational parameter)", 0.1, 5.0, 1.0, 0.1)
    r0 = st.sidebar.slider("Initial orbital radius r0", 0.2, 3.0, 1.0, 0.1)
    speed_factor = st.sidebar.slider("Speed ÷ circular speed", 0.3, 1.6, 1.0, 0.05)
    t, x_orb, y_orb, info = orbital_motion(GM=GM, r0=r0, speed_factor=speed_factor)
    params = {"GM": GM, "r0": r0, "speed_factor": speed_factor}
    plot_kind = "orbital"

else:  # Coupled Oscillators
    m1 = st.sidebar.slider("Mass 1 (kg)", 0.2, 3.0, 1.0, 0.1)
    m2 = st.sidebar.slider("Mass 2 (kg)", 0.2, 3.0, 1.0, 0.1)
    k = st.sidebar.slider("Outer spring constant k (N/m)", 1.0, 30.0, 10.0, 1.0)
    k12 = st.sidebar.slider("Coupling spring constant k12 (N/m)", 0.0, 30.0, 5.0, 1.0)
    x1_0 = st.sidebar.slider("Initial displacement x1 (m)", -2.0, 2.0, 1.0, 0.1)
    x2_0 = st.sidebar.slider("Initial displacement x2 (m)", -2.0, 2.0, 0.0, 0.1)
    t, x1_arr, x2_arr, info = coupled_oscillators(m1=m1, m2=m2, k=k, k12=k12, x1_0=x1_0, x2_0=x2_0)
    params = {"m1 (kg)": m1, "m2 (kg)": m2, "k (N/m)": k, "k12 (N/m)": k12, "x1_0 (m)": x1_0, "x2_0 (m)": x2_0}
    plot_kind = "coupled"

col1, col2 = st.columns([2, 1])

with col1:
    if plot_kind == "time_series":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, y, color="#2563eb", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{system}: {ylabel} vs Time")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    elif plot_kind == "trajectory":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_traj, y_traj, color="#2563eb", linewidth=2)
        ax.fill_between(x_traj, y_traj, alpha=0.1, color="#2563eb")
        ax.set_xlabel("Horizontal distance x (m)")
        ax.set_ylabel("Height y (m)")
        ax.set_title("Projectile Trajectory")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    elif plot_kind == "double_pendulum":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(t, th1, label="θ1", color="#2563eb")
        axes[0].plot(t, th2, label="θ2", color="#dc2626")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Angle (degrees)")
        axes[0].set_title("Angles vs Time")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(x2, y2, color="#7c3aed", linewidth=0.7, alpha=0.8)
        axes[1].set_xlabel("x position of bob 2 (m)")
        axes[1].set_ylabel("y position of bob 2 (m)")
        axes[1].set_title("Trajectory of Second Bob (chaos trace)")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)

    elif plot_kind == "field":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1])

        im = ax1.imshow(
            U, aspect="auto", origin="lower",
            extent=[x_arr[0], x_arr[-1], t_arr[0], t_arr[-1]],
            cmap="RdBu_r" if system == "1D Wave Equation" else "inferno",
        )
        ax1.set_xlabel("Position x (m)")
        ax1.set_ylabel("Time (s)")
        ax1.set_title(f"{system}: evolution over space and time")
        fig.colorbar(im, ax=ax1, label="Amplitude" if system == "1D Wave Equation" else "Temperature")

        snap_frac = st.slider("Snapshot time", 0.0, 1.0, 0.3, 0.01, key="snap")
        snap_idx = int(snap_frac * (len(t_arr) - 1))
        ax2.plot(x_arr, U[snap_idx], color="#2563eb", linewidth=2)
        ax2.set_xlabel("Position x (m)")
        ax2.set_ylabel("Amplitude" if system == "1D Wave Equation" else "Temperature")
        ax2.set_title(f"Snapshot at t = {t_arr[snap_idx]:.3f} s")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)

    elif plot_kind == "orbital":
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x_orb, y_orb, color="#2563eb", linewidth=1.5)
        ax.scatter([0], [0], color="#f59e0b", s=250, marker="*", label="Central mass", zorder=5)
        ax.scatter([x_orb[0]], [y_orb[0]], color="#16a34a", s=50, label="Start", zorder=5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Orbital Trajectory")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    elif plot_kind == "coupled":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, x1_arr, label="x1 (mass 1)", color="#2563eb")
        ax.plot(t, x2_arr, label="x2 (mass 2)", color="#dc2626")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_title("Coupled Oscillators: x1(t) and x2(t)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

with col2:
    st.subheader("Derived quantities")
    for key, value in info.items():
        if isinstance(value, (float, np.floating)):
            st.write(f"**{key.replace('_', ' ').title()}:** {value:.4f}")
        else:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

# ---------------------------------------------------------------------------
# Reference info: governing equations, numerical method, applications/notes
# ---------------------------------------------------------------------------
st.divider()
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    with st.expander("📚 Physics Behind This"):
        for eq in SYSTEM_INFO[system]["equations"]:
            st.latex(eq)

with info_col2:
    with st.expander("🔧 Numerical Method"):
        st.write(SYSTEM_INFO[system]["method"])

with info_col3:
    with st.expander("🌍 Applications & Notes"):
        st.write(f"**Difficulty:** {SYSTEM_INFO[system]['difficulty']}")
        st.write("**Prerequisites:** " + ", ".join(SYSTEM_INFO[system]["prerequisites"]))
        st.write("**Real-world applications:**")
        for app_item in SYSTEM_INFO[system]["applications"]:
            st.write(f"- {app_item}")

# ---------------------------------------------------------------------------
# Energy + phase space, for the systems where the decomposition is clean
# ---------------------------------------------------------------------------
if system in ENERGY_SYSTEMS:
    st.divider()
    st.subheader("⚡ Energy & Phase Space")

    if system == "Simple Harmonic Oscillator":
        omega = info["angular_frequency"]
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)  # exact analytic derivative
        KE = 0.5 * mass * v**2
        PE = 0.5 * k * y**2
        x_for_phase, xlabel, vlabel = y, "Position x (m)", "Velocity (m/s)"
        energy_note = "Velocity computed from the exact analytic derivative (no numerical differentiation)."

    elif system == "Damped Harmonic Oscillator":
        v = np.gradient(y, t)
        KE = 0.5 * mass * v**2
        PE = 0.5 * k * y**2
        x_for_phase, xlabel, vlabel = y, "Position x (m)", "Velocity (m/s)"
        energy_note = "Velocity obtained by numerical differentiation (np.gradient). Total energy should visibly decay due to damping."

    elif system == "Simple Pendulum":
        theta_rad = np.radians(y)
        thetadot = np.gradient(theta_rad, t)
        m_nominal = 1.0
        KE = 0.5 * m_nominal * (length**2) * thetadot**2
        PE = m_nominal * 9.81 * length * (1 - np.cos(theta_rad))
        v = thetadot
        x_for_phase, xlabel, vlabel = theta_rad, "Angle θ (rad)", "Angular velocity (rad/s)"
        energy_note = f"Assumes a nominal bob mass of {m_nominal} kg (mass isn't one of the pendulum's parameters above). Angular velocity via numerical differentiation."

    else:  # Coupled Oscillators
        v1 = np.gradient(x1_arr, t)
        v2 = np.gradient(x2_arr, t)
        KE = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
        PE = 0.5 * k * x1_arr**2 + 0.5 * k * x2_arr**2 + 0.5 * k12 * (x1_arr - x2_arr) ** 2
        v = v1
        x_for_phase, xlabel, vlabel = x1_arr, "Position of mass 1 (m)", "Velocity of mass 1 (m/s)"
        energy_note = "Velocities via numerical differentiation. Phase space shown for mass 1 only."

    Total = KE + PE

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        fig_e, ax_e = plt.subplots(figsize=(6, 4))
        ax_e.plot(t, KE, label="Kinetic Energy", color="#16a34a", linewidth=1.8)
        ax_e.plot(t, PE, label="Potential Energy", color="#dc2626", linewidth=1.8)
        ax_e.plot(t, Total, label="Total Energy", color="#2563eb", linewidth=2, linestyle="--")
        ax_e.set_xlabel("Time (s)")
        ax_e.set_ylabel("Energy (arbitrary units)")
        ax_e.set_title("Energy vs Time")
        ax_e.legend()
        ax_e.grid(alpha=0.3)
        fig_e.tight_layout()
        st.pyplot(fig_e)

    with ecol2:
        fig_p, ax_p = plt.subplots(figsize=(6, 4))
        ax_p.plot(x_for_phase, v, color="#7c3aed", linewidth=1.2)
        ax_p.set_xlabel(xlabel)
        ax_p.set_ylabel(vlabel)
        ax_p.set_title("Phase Space (velocity vs position)")
        ax_p.grid(alpha=0.3)
        fig_p.tight_layout()
        st.pyplot(fig_p)

    st.caption(f"ℹ️ {energy_note}")

st.divider()
st.subheader("🤖 AI Tutor")

level = st.radio("Explanation level", ["Beginner", "Intermediate"], horizontal=True)

tab_explain, tab_ask = st.tabs(["Explain current behavior", "Ask a question"])

with tab_explain:
    if st.button("Explain this behavior"):
        with st.spinner("Thinking..."):
            explanation = get_explanation(system, params, info, level)
        st.info(explanation)

with tab_ask:
    question = st.text_input(
        "Ask about this simulation",
        placeholder="e.g. Why does increasing k increase the frequency?",
    )
    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            answer = get_explanation(system, params, info, level, question=question)
        st.info(answer)

st.caption(
    "Built by Tabassum Tariq · Physics Concept Lab · "
    "Simulations solved analytically / numerically in Python (NumPy, SciPy)."
)
