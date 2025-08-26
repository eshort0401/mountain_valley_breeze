import asyncio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyscript import document, display, when
import js

# --- Grid and Figure Setup ---
x = np.linspace(-1.75, 1.75, 201)
z = np.linspace(-1, 3, 401)
X, Z = np.meshgrid(x, z)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(wspace=0.3)

# Create placeholder artists.
kwargs = {"cmap": "RdBu_r", "origin": "lower", "aspect": "auto", "zorder": 0}
kwargs.update({"extent": [x.min(), x.max(), z.min(), z.max()]})
cmap = plt.get_cmap(kwargs["cmap"])
Q_levels = np.linspace(-1, 1, 21)
Q_norm = mcolors.BoundaryNorm(Q_levels, ncolors=cmap.N, extend="both")
psi_max = 0.5
psi_levels = np.linspace(-psi_max, psi_max, 21)
psi_norm = mcolors.BoundaryNorm(psi_levels, ncolors=cmap.N, extend="both")

# Plot psi
im_psi = axes[0].imshow(np.zeros_like(Z), **kwargs, norm=psi_norm)
psi_cbar = fig.colorbar(im_psi, ax=axes[0], extend="both")
psi_cbar.set_label(r"$\psi$ [-]")
psi_cbar.set_ticks(np.linspace(-psi_max, psi_max, 11))

# Plot Q
im_Q = axes[1].imshow(np.zeros_like(Z), **kwargs, norm=Q_norm)
Q_cbar = fig.colorbar(im_Q, ax=axes[1], extend="both")
Q_cbar.set_label(r"$Q$ [-]")
Q_cbar.set_ticks(np.linspace(-1, 1, 11))

# Quiver plot
step = 20
subset = (slice(None, None, step), slice(None, None, step))
dummy_wind = np.zeros_like(X)
quiv_args = [X[subset], Z[subset], dummy_wind[subset], dummy_wind[subset]]
quiv_kwargs = {"color": "k", "scale": 2.5, "width": 0.006, "angles": "xy"}
quiv = axes[0].quiver(*quiv_args, **quiv_kwargs, zorder=2)
axes[0].quiverkey(quiv, 0.80, 1.05, 0.2, r"0.2 [-]", labelpos="E", coordinates="axes")

# Plot the slope line
dark_brown = tuple([c * 0.5 for c in mcolors.to_rgb("tab:brown")])
slope_line_psi = axes[0].plot(x, x * 0, color=dark_brown, zorder=1)[0]
slope_line_Q = axes[1].plot(x, x * 0, color=dark_brown, zorder=1)[0]

for i, ax in enumerate(axes):
    ax.set_ylim(-1, 2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel(r"$x$ [-]")
    if i == 0:
        ax.set_ylabel(r"$z$ [-]")
    ax.set_facecolor("tab:brown")
    ax.set_aspect("equal")

axes[0].set_title(r"$\psi$ [-]")
axes[1].set_title(r"$Q$ [-]")

fig.patch.set_facecolor("#E6E6E6")
# fig.tight_layout()
fig.suptitle(r"", y=0.975)

# --- Display the empty figure template on the page IMMEDIATELY ---
display(fig, target="plot-output")


@when("input", "#f_omega_slider, #alpha_omega_slider, #N_omega_slider, #M_slider")
def update_params(event):
    """Update the model parameters."""
    global psi_base, u_base, w_base, Q_base, B

    # Get slider values
    f_omega = float(document.getElementById("f_omega_slider").value)
    alpha_omega = float(document.getElementById("alpha_omega_slider").value)
    N_omega = float(document.getElementById("N_omega_slider").value)
    M = float(document.getElementById("M_slider").value)

    # Update the text output
    document.getElementById("f_omega_out").innerText = f"{f_omega:.2f}"
    document.getElementById("alpha_omega_out").innerText = f"{alpha_omega:.2f}"
    document.getElementById("N_omega_out").innerText = f"{N_omega:.1f}"
    document.getElementById("M_out").innerText = f"{M:.2f}"

    # Perform the original physics calculations
    Z_sigma = Z - M * X
    B = (
        (1 - 2j * alpha_omega - alpha_omega**2) * (1 + (1 / N_omega**2) * M**2)
        - f_omega**2
        - M**2
    ) ** 0.5
    abs_B_sq = np.abs(B) ** 2
    exp_Z = np.exp(-Z_sigma)

    psi_base = M / abs_B_sq * (exp_Z - 1)
    u_base = -M / abs_B_sq * exp_Z
    w_base = -(M**2) / abs_B_sq * exp_Z
    Q_base = exp_Z

    mask = Z < M * X
    psi_base[mask] = np.nan
    u_base[mask] = np.nan
    w_base[mask] = np.nan
    Q_base[mask] = np.nan

    update_time(event)


@when("input", "#t_slider")
def update_time(event):
    # Get slider values
    t = float(document.getElementById("t_slider").value)
    M = float(document.getElementById("M_slider").value)
    document.getElementById("t_out").innerText = f"{t:.2f}"

    cos_t_B = np.cos(t - 2 * np.angle(B))
    psi = psi_base * cos_t_B
    Q = Q_base * np.cos(t)
    u = u_base * cos_t_B
    w = w_base * cos_t_B

    # Update artist data
    im_psi.set_data(psi)
    im_Q.set_data(Q)
    slope_line_psi.set_ydata(x * M)
    slope_line_Q.set_ydata(x * M)
    quiv.set_UVC(u[subset], w[subset])

    fig.suptitle(rf"$t={t/np.pi:.4f}\pi$ [-]", y=0.975)

    # Redraw fig using display
    display(fig, target="plot-output", append=False)


update_params(None)
