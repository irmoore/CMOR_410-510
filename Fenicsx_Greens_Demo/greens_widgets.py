import io
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    return buf.getvalue()


def sweep_widget(G, u_exact):
    y = np.linspace(-2.5, 2.5, 801)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    f_line, = ax1.plot([], [], 'k--', label='f(y)')
    g_line, = ax1.plot([], [], color='C1', label='G(x - y)')
    p_line, = ax1.plot([], [], color='C0', lw=2, label='G(x - y) f(y)')
    cursor  = ax1.axvline(0, color='C3', lw=1)
    ax1.set_xlim(y[0], y[-1])
    ax1.set_title('Integrand in y')

    ghost,   = ax2.plot([], [], color='lightgray', lw=3)
    run,     = ax2.plot([], [], color='C0', lw=2, label='numerical integral')
    dot,     = ax2.plot([], [], 'o', color='C3')
    exact_ln = ax2.axhline(0, color='k', ls=':', lw=1)
    ax2.set_xlabel('y')
    plt.close(fig)                        # keep the notebook from also showing it as a static image

    img = widgets.Image(format='png')
    s_sigma = widgets.FloatSlider(value=0.5, min=0.05, max=1.0, step=0.05, description='σ')
    s_x     = widgets.FloatSlider(value=0.4, min=-10.0, max=10.0, step=0.05, description='x')
    s_y     = widgets.FloatSlider(value=2.5, min=-2.5, max=2.5, step=0.02, description='sweep y')

    def redraw(change=None):
        sigma, x0, y_cut = s_sigma.value, s_x.value, s_y.value
        f  = norm.pdf(y, scale=sigma)
        Gy = G(x0 - y)
        P  = Gy * f
        I  = cumulative_trapezoid(P, y, initial=0)
        k  = min(np.searchsorted(y, y_cut), len(y) - 1)
        exact = u_exact(x0, sigma)

        f_line.set_data(y, f); g_line.set_data(y, Gy); p_line.set_data(y, P)
        g_line.set_label(f'G(x - y),  x = {x0:.2f}')
        for c in list(ax1.collections):   # remove the previous shaded area
            c.remove()
        ax1.fill_between(y[:k+1], P[:k+1], color='C0', alpha=0.3)
        cursor.set_xdata([y_cut, y_cut])
        ax1.set_ylim(-0.05, max(1.4, 1.05*f.max())); ax1.legend(loc='upper left')

        ghost.set_data(y, I); run.set_data(y[:k+1], I[:k+1]); dot.set_data([y[k]], [I[k]])
        exact_ln.set_ydata([exact, exact]); exact_ln.set_label('exact (G * f)(x)')
        ax2.set_ylim(-0.03, 1.15*max(I[-1], exact)); ax2.legend(loc='upper left')

        img.value = _to_png(fig)

    for s in (s_sigma, s_x, s_y):
        s.observe(redraw, names='value')
    redraw()
    return widgets.VBox([s_sigma, s_x, s_y, img])


def solution_widget(G, u_exact, u_conv):
    x = np.linspace(-3, 3, 301)
    y = np.linspace(-5, 5, 2001)       

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    f_line,  = ax1.plot([], [], 'k--', label='f(x)')
    ax1.set_title('Forcing'); ax1.legend(loc='upper right')
    ax2.plot(x, G(x), color='C1', lw=1, ls='--', label='G(x) = |x|/2')
    ex_line, = ax2.plot([], [], color='lightgray', lw=5, label='exact')
    u_line,  = ax2.plot([], [], color='C0', lw=2, label='numerical computation')
    ax2.set_xlim(x[0], x[-1]); ax2.set_xlabel('x'); ax2.legend(loc='upper center')
    plt.close(fig)

    img = widgets.Image(format='png')
    s_sigma = widgets.FloatSlider(value=0.5, min=0.05, max=1.0, step=0.05, description='σ')

    def redraw(change=None):
        sigma = s_sigma.value
        u  = u_conv(x, sigma, y)
        ue = u_exact(x, sigma)
        f_line.set_data(x, norm.pdf(x, scale=sigma))
        ax1.set_ylim(-0.05, 1.1*norm.pdf(0, scale=sigma))
        ex_line.set_data(x, ue); u_line.set_data(x, u)
        ax2.set_ylim(0, 1.1*max(u.max(), G(x).max()))
        ax2.set_title(f'max |computed - exact| = {np.max(np.abs(u - ue)):.1e}')
        img.value = _to_png(fig)

    s_sigma.observe(redraw, names='value')
    redraw()
    return widgets.VBox([s_sigma, img])
