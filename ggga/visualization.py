import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from concurrent.futures import ProcessPoolExecutor
import typing as t
import functools

from . import SurrogateModel, Space


def partial_dependence(
    space, model, i, j=None, *, samples_transformed, n_points, rng,
):
    def transformed_bounds_linspace(param, n_points):
        return np.linspace(*param.transformed_bounds(), n_points)

    # one-dimensional case
    if j is None:
        xi_transformed = transformed_bounds_linspace(
            space.params[i], n_points)
        yi = np.zeros(n_points)
        stdi = np.zeros(n_points)
        for n, x in enumerate(xi_transformed):
            real_transformed_samples = np.array(samples_transformed)
            real_transformed_samples[:, i] = x
            y, std = model.predict_transformed_a(real_transformed_samples)
            yi[n] = np.mean(y)
            stdi[n] = np.mean(std)
        return xi_transformed, yi, stdi

    # two-dimensional case
    xi_transformed = transformed_bounds_linspace(
            space.params[j], n_points)
    yi_transformed = transformed_bounds_linspace(
            space.params[i], n_points)
    zi = []
    for x in xi_transformed:
        row = []
        for y in yi_transformed:
            real_transformed_samples = np.array(samples_transformed)
            real_transformed_samples[:, (j, i)] = (x, y)
            row.append(np.mean(model.predict_transformed_a(
                real_transformed_samples, return_std=False)))
        zi.append(row)
    return xi_transformed, yi_transformed, np.array(zi).T


async def plot_objective(
    loop, xs, ys, *,
    x_min,
    model: SurrogateModel,
    space: Space,
    contour_levels: int=10,
    n_points: int=40,
    n_samples: int=250,
    subplot_size: float=2,
    rng: np.random.RandomState,
    cmap='viridis_r',
):
    n_dims = space.n_dims

    samples_transformed = [
        space.into_transformed(space.sample(rng=rng))
        for _ in range(n_samples)
    ]

    executor = ProcessPoolExecutor()
    single_dependence = []
    dual_dependence: t.List[t.List[t.Awaitable[tuple]]] = []

    for row in range(n_dims):
        single_dependence.append(loop.run_in_executor(
            executor, functools.partial(
                partial_dependence,
                space, model, row,
                samples_transformed=samples_transformed,
                n_points=n_points,
                rng=rng)))

        dual_dependence.append([])
        for col in range(row):  # leave upper triangle empty
            dual_dependence[row].append(loop.run_in_executor(
                executor, functools.partial(
                    partial_dependence,
                    space, model, row, col,
                    samples_transformed=samples_transformed,
                    n_points=n_points,
                    rng=rng)))

    fig, ax = plt.subplots(
        n_dims, n_dims, figsize=(subplot_size * n_dims, subplot_size * n_dims))
    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95,
        hspace=0.03, wspace=0.03)

    for row in range(n_dims):
        # plot single-variable dependence on diagonal
        param_row = space.params[row]
        print("[INFO] dependence plot 1D {} ({})".format(row, param_row.name))
        xi_transformed, yi, stdi = await single_dependence[row]
        xi = param_row.from_transformed_a(xi_transformed)
        ax_ii = ax[row, row]
        plot_single_variable_dependence(
            ax_ii, row,
            xi=xi, yi=yi, stdi=stdi, xs=xs, ys=ys, x_min=x_min,
            param=param_row)
        ax_ii.set_title(param_row.name + "\n")
        ax_ii.yaxis.tick_right()
        ax_ii.xaxis.tick_top()

        for col in range(row):
            # plot two-variable dependence on lower triangle
            param_col = space.params[col]
            print("[INFO] dependence plot {} x {} ({} x {})".format(
                row, col, param_row.name, param_col.name))
            xi_transformed, yi_transformed, zi = \
                await dual_dependence[row][col]
            xi = param_col.from_transformed_a(xi_transformed)
            yi = param_row.from_transformed_a(yi_transformed)
            ax_ij = ax[row, col]
            plot_dual_variable_dependence(
                ax_ij, col, row,
                xi=xi, yi=yi, zi=zi, xs=xs, x_min=x_min,
                param_x=param_col, param_y=param_row,
                contour_levels=contour_levels, cmap=cmap,
                show_xticks=(row == n_dims - 1),
                show_yticks=(col == 0))

        # hide top right triangle
        for col in range(row + 1, n_dims):
            ax[row, col].axis('off')

    return fig, ax


def plot_single_variable_dependence(
    ax, i, *,
    xi, yi, stdi,
    xs, ys,
    x_min,
    param,
    scatter_args=dict(c='g', s=10, lw=0, alpha=0.5),
    minline_args=dict(linestyle='--', color='r', lw=1),
):
    ax.fill_between(xi, yi - 1.96*stdi, yi + 1.96*stdi, color='b', alpha=0.15)
    ax.scatter([x[i] for x in xs], ys, **scatter_args)
    ax.plot(xi, yi, c='b')
    ax.axvline(x_min[i], **minline_args)
    x_bounds = param.bounds()
    if x_bounds is not None:
        ax.set_xlim(*x_bounds)


def plot_dual_variable_dependence(
    ax, i, j, *,
    xi, yi, zi,
    xs,
    x_min,
    param_x, param_y,
    contour_levels,
    cmap,
    scatter_args=dict(c='k', s=10, lw=0),
    xmin_scatter_args=dict(c='r', s=10, lw=0),
    show_xticks: bool,
    show_yticks: bool,
):
    ax.contourf(xi, yi, zi, contour_levels, locator=None, cmap=cmap)
    ax.scatter([x[i] for x in xs], [x[j] for x in xs], **scatter_args)
    ax.scatter(x_min[i], x_min[j], **xmin_scatter_args)
    x_bounds = param_x.bounds()
    if x_bounds is not None:
        ax.set_xlim(*x_bounds)
    y_bounds = param_y.bounds()
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    if not show_xticks:
        ax.tick_params('x', top=False, bottom=False, labelbottom=False)
    if not show_yticks:
        ax.tick_params('y', left=False, right=False, labelleft=False)
