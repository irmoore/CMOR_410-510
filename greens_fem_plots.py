import numpy as np
import matplotlib.pyplot as plt
import pyvista
import ufl
import dolfinx
from dolfinx import geometry
from viskex.dolfinx_plotter import DolfinxPlotter
from viskex.dolfinx_converter import DolfinxConverter


def eval_points(u, pts):
    """Evaluate a dolfinx Function at an (N, 2) array of points inside its mesh."""
    msh = u.function_space.mesh
    P = np.zeros((len(pts), 3))
    P[:, :2] = pts
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, P)
    colliding = geometry.compute_colliding_cells(msh, candidates, P)
    cells = np.array([colliding.links(i)[0] for i in range(len(P))], dtype=np.int32)
    return u.eval(P, cells).ravel()


def compare_along_x(uh, u_exact, title):
    """Plot the finite element solution and the free-space solution G * f along the line y = 0."""
    L = np.abs(uh.function_space.mesh.geometry.x[:, 0]).max()
    xs = np.linspace(-0.999*L, 0.999*L, 401)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, eval_points(uh, np.column_stack([xs, np.zeros_like(xs)])), lw=2,
            label='finite elements (u = 0 on the boundary)')
    ax.plot(xs, u_exact(np.abs(xs)), '--', lw=2, label='free space  G * f')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('x  (along y = 0)'); ax.set_ylabel('u'); ax.set_title(title); ax.legend()
    fig.tight_layout()
    plt.show()


def velocity(phi):
    """The flow velocity v = ∇φ, as a vector-valued dolfinx Function."""
    W = dolfinx.fem.functionspace(phi.function_space.mesh, ("Lagrange", 1, (2,)))
    v = dolfinx.fem.Function(W)
    v.interpolate(dolfinx.fem.Expression(ufl.grad(phi), W.element.interpolation_points))
    return v


def plot_flow(phi, sources, inflow=False):
    """Speed |∇φ| as colour, with streamlines started around each source (and along the left edge if inflow)."""
    v = velocity(phi)
    L = np.abs(phi.function_space.mesh.geometry.x[:, 0]).max()

    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    seeds = [np.column_stack([sx + 0.3*np.cos(angles), sy + 0.3*np.sin(angles)]) for sx, sy in sources]
    if inflow:
        ys = np.linspace(-0.95*L, 0.95*L, 25)
        seeds.append(np.column_stack([np.full_like(ys, -0.97*L), ys]))
    seeds = np.vstack(seeds)

    plotter = DolfinxPlotter.plot_vector_field(v, "speed", plotter=pyvista.Plotter(window_size=(700, 650)),
                                               show_edges=False, cmap="viridis")
    grid = DolfinxConverter.convert_field(v, "velocity")
    lines = grid.streamlines_from_source(pyvista.PolyData(np.column_stack([seeds, np.zeros(len(seeds))])),
                                         vectors="velocity", surface_streamlines=True,
                                         integration_direction="both", max_length=50.0,
                                         initial_step_length=0.1, max_steps=20000)
    plotter.add_mesh(lines, color="white", line_width=1.5)
    plotter.view_xy()
    plotter.show()
