import asyncio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyscript import document, display, when
import js


def initialize_figure():
    """Initialize the figure and set the global variables."""
    global fig, ax, im_psi, im_Q, quiv, slope_line_psi, slope_line_Q, subset, x, z
    global X, Z

    x = np.linspace(-1.75, 1.75, 201)
    z = np.linspace(-1, 3, 401)
    X, Z = np.meshgrid(x, z)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(wspace=0.3)

    # Create placeholder artists.
    kwargs = {"cmap": "RdBu_r", "origin": "lower", "aspect": "auto", "zorder": 0}
    kwargs.update({"extent": [x.min(), x.max(), z.min(), z.max()]})
    cmap = plt.get_cmap(kwargs["cmap"])
    psi_max = 4.0
    psi_levels = np.linspace(-psi_max, psi_max, 21)
    psi_norm = mcolors.BoundaryNorm(psi_levels, ncolors=cmap.N, extend="both")

    im_psi = ax.imshow(np.zeros_like(Z), **kwargs, norm=psi_norm)
    psi_cbar = fig.colorbar(im_psi, ax=ax, extend="both")
    psi_cbar.set_label(r"$\psi$ [-]")
    psi_cbar.set_ticks(np.linspace(-psi_max, psi_max, 11))

    # # Quiver plot
    # step = 20
    # subset = (slice(None, None, step), slice(None, None, step))
    # dummy_wind = np.zeros_like(X)
    # quiv_args = [X[subset], Z[subset], dummy_wind[subset], dummy_wind[subset]]
    # quiv_kwargs = {"color": "k", "scale": 2.5, "width": 0.006, "angles": "xy"}
    # quiv = axes[0].quiver(*quiv_args, **quiv_kwargs, zorder=2)
    # axes[0].quiverkey(
    #     quiv, 0.80, 1.05, 0.2, r"0.2 [-]", labelpos="E", coordinates="axes"
    # )

    # Plot the slope line
    dark_brown = tuple([c * 0.5 for c in mcolors.to_rgb("tab:brown")])
    slope_line_psi = ax.plot(x, x * 0, color=dark_brown, zorder=1)[0]
    slope_line_Q = ax.plot(x, x * 0, color=dark_brown, zorder=1)[0]

    ax.set_ylim(-1, 2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel(r"$x$ [-]")
    ax.set_ylabel(r"$z$ [-]")
    ax.set_facecolor("tab:brown")
    ax.set_aspect("equal")

    ax.set_title(r"$\psi$ [-]")
    fig.patch.set_facecolor("#E6E6E6")
    fig.suptitle(r"", y=0.995)

    display(fig, target="plot-output")


def calculate_constants(f_omega, alpha_omega, N_omega, M):
    """Calculate the constants B, C, A, a_n, a_p used in the solution."""

    B = (
        (1 - 2 * 1j * alpha_omega - alpha_omega**2) * (1 + 1 / N_omega**2 * M**2)
        - f_omega**2
        - M**2
    ) ** (1 / 2)
    C = (1 / N_omega**2 * (-1 + 2 * 1j * alpha_omega + alpha_omega**2) + 1) ** (1 / 2)
    B, C = np.complex128(B), np.complex128(C)
    A = B / C
    a_1 = 1 / A * (-M / A + np.sqrt(M**2 / A**2 + 1))
    a_2 = 1 / A * (-M / A - np.sqrt(M**2 / A**2 + 1))

    if np.imag(a_1) > 0:
        a_n = a_1
        a_p = a_2
    else:
        a_n = a_2
        a_p = a_1

    return B, a_p, a_n


def get_psi_p_tilde(X, Z, M, z_f, B, a_p, a_n):
    """Get the psi_p_tilde solution."""

    Z_sigma = Z - M * X
    cond_1 = (0 < Z_sigma) & (Z_sigma <= z_f)
    cond_2 = Z_sigma > z_f

    psi_p_tilde = np.zeros_like(Z_sigma, dtype=complex)

    t_1 = a_p * (Z_sigma[cond_1] - z_f) + X[cond_1]
    t_2 = a_n * Z_sigma[cond_1] - a_p * z_f + X[cond_1]

    t_3 = a_n * (Z_sigma[cond_2] - z_f) + X[cond_2]
    t_4 = a_n * Z_sigma[cond_2] - a_p * z_f + X[cond_2]

    psi_p_tilde[cond_1] = 1j * (1 / t_1 - 1 / t_2)
    psi_p_tilde[cond_1] += M * a_p / 1j * (1 / t_1 - 1 / t_2)

    psi_p_tilde[cond_2] = 1j * (1 / t_3 - 1 / t_4)
    psi_p_tilde[cond_2] += M / 1j * (a_n / t_3 - a_p / t_4)

    psi_p_tilde = 1 / ((a_n - a_p) * (2 * B**2)) * psi_p_tilde
    return psi_p_tilde


def get_psi_n_tilde(X, Z, M, z_f, B, a_p, a_n):
    """Get the psi_n_tilde solution."""

    Z_sigma = Z - M * X
    cond_1 = (0 < Z_sigma) & (Z_sigma <= z_f)
    cond_2 = Z_sigma > z_f

    psi_n_tilde = np.zeros_like(Z_sigma, dtype=complex)

    t_1 = a_p.conj() * Z_sigma[cond_1] - a_n.conj() * z_f + X[cond_1]
    t_2 = a_n.conj() * (Z_sigma[cond_1] - z_f) + X[cond_1]

    t_3 = a_p.conj() * Z_sigma[cond_2] - a_n.conj() * z_f + X[cond_2]
    t_4 = a_p.conj() * (Z_sigma[cond_2] - z_f) + X[cond_2]

    psi_n_tilde[cond_1] = 1j * (1 / t_1 - 1 / t_2)
    psi_n_tilde[cond_1] += M * a_n.conj() / 1j * (1 / t_1 - 1 / t_2)

    psi_n_tilde[cond_2] = 1j * (1 / t_3 - 1 / t_4)
    psi_n_tilde[cond_2] += M / 1j * (a_n.conj() / t_3 - a_p.conj() / t_4)

    psi_n_tilde = 1 / ((a_n.conj() - a_p.conj()) * (2 * B.conj() ** 2)) * psi_n_tilde
    return psi_n_tilde


sliders = "#f_omega_slider, #alpha_omega_slider, #N_omega_slider, #M_slider, "
sliders += "#z_f_slider, #t_slider"


@when("input", sliders)
def update_params(event):
    """Update the model parameters."""
    global psi_p_tilde, psi_n_tilde, B, a_p, a_n

    # Get slider values
    f_omega = float(document.getElementById("f_omega_slider").value)
    alpha_omega = float(document.getElementById("alpha_omega_slider").value)
    N_omega = float(document.getElementById("N_omega_slider").value)
    M = float(document.getElementById("M_slider").value)
    z_f = float(document.getElementById("z_f_slider").value)

    # Update the text output
    document.getElementById("f_omega_out").innerText = f"{f_omega:.2f}"
    document.getElementById("alpha_omega_out").innerText = f"{alpha_omega:.2f}"
    document.getElementById("N_omega_out").innerText = f"{N_omega:.1f}"
    document.getElementById("M_out").innerText = f"{M:.2f}"
    document.getElementById("z_f_out").innerText = f"{z_f:.2f}"

    # Perform the original physics calculations
    B, a_p, a_n = calculate_constants(f_omega, alpha_omega, N_omega, M)
    psi_p_tilde = get_psi_p_tilde(X, Z, M, z_f, B, a_p, a_n)
    psi_n_tilde = get_psi_n_tilde(X, Z, M, z_f, B, a_p, a_n)

    mask = Z < M * X
    psi_p_tilde[mask] = np.nan
    psi_n_tilde[mask] = np.nan

    update_time(event)


@when("input", "#t_slider")
def update_time(event):
    # Get slider values
    t = float(document.getElementById("t_slider").value)
    M = float(document.getElementById("M_slider").value)
    document.getElementById("t_out").innerText = f"{t:.2f}"

    psi = np.real(psi_p_tilde * np.exp(1j * t) + psi_n_tilde * np.exp(-1j * t))

    # Update artist data
    im_psi.set_data(psi)
    slope_line_psi.set_ydata(x * M)
    slope_line_Q.set_ydata(x * M)
    # quiv.set_UVC(u[subset], w[subset])

    fig.suptitle(rf"$t={t/np.pi:.4f}\pi$ [-]", y=0.995)

    # Redraw fig using display
    display(fig, target="plot-output", append=False)


def hide_loading_screen():
    """Hide loading screen and show main content"""
    loading_screen = document.getElementById("loading-screen")
    main_content = document.getElementById("main-content")

    loading_screen.style.display = "none"
    main_content.style.display = "block"


initialize_figure()
update_params(None)
hide_loading_screen()
