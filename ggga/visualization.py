import typing as t

import attr
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

from .minimize import Individual
from . import SurrogateModel, Space


def partial_dependence(
    model: SurrogateModel, dim_1: int, dim_2: int = None, *,
    samples_transformed: np.ndarray, n_points: int,
) -> tuple:

    assert samples_transformed is not None
    samples_transformed = np.array(samples_transformed)  # make a copy

    x_transformed = np.linspace(0.0, 1.0, n_points)

    if dim_2 is None:
        ys = np.zeros(n_points)
        ys_std = np.zeros(n_points)
        for i, x in enumerate(x_transformed):
            samples_transformed[:, dim_1] = x
            y, std = model.predict_transformed_a(samples_transformed)
            ys[i] = np.mean(y)
            # IDEALLY: std(a + b) = sqrt(var(a) + var(b) - cov(a, b))
            # NOT: mean of stdev
            # INSTEAD: mean of variance
            ys_std[i] = np.sqrt(np.mean(std**2))
        return x_transformed, ys, ys_std

    ys = np.zeros((n_points, n_points))
    for i, x_i in enumerate(x_transformed):
        for j, x_j in enumerate(x_transformed):
            samples_transformed[:, (dim_1, dim_2)] = (x_i, x_j)
            ys[i, j] = np.mean(model.predict_transformed_a(
                samples_transformed, return_std=False))
    return x_transformed, x_transformed, ys


@attr.s(frozen=True)
class ObjectivePlotStyle:
    contour_levels: int = 10
    subplot_size: float = 2
    cmap: str = 'viridis_r'


@attr.s(frozen=True)
class ObjectivePlotQuality:
    points: int = 40
    samples_per_point: int = 250


def plot_objective(
    x_observed, y_observed, *,
    x_min=None,
    model: SurrogateModel,
    space: Space,
    rng: np.random.RandomState,
    quality: ObjectivePlotQuality = ObjectivePlotQuality(),
    style: ObjectivePlotStyle = ObjectivePlotStyle(),
) -> t.Tuple[t.Any, t.Any]:
    n_dims = space.n_dims

    if x_min is None:
        x_min = x_observed[np.argmin(y_observed)]

    samples_transformed = [
        space.into_transformed(space.sample(rng=rng))
        for _ in range(quality.samples_per_point)
    ]

    fig, axes = plt.subplots(
        n_dims, n_dims,
        figsize=(style.subplot_size * n_dims, style.subplot_size * n_dims),
        squeeze=False)

    for row in range(n_dims):
        # plot single-variable dependence on diagonal
        param_row = space.params[row]
        print("[INFO] dependence plot 1D {} ({})".format(row, param_row.name))
        x_row_transformed, ys, std = partial_dependence(
                model, row,
                samples_transformed=samples_transformed,
                n_points=quality.points)
        x_row = param_row.from_transformed_a(x_row_transformed)
        ax = axes[row, row]
        plot_single_variable_dependence(
            ax, row,
            xs=x_observed, ys=y_observed, x_min=x_min,
            xi=x_row, yi=ys, stdi=std,
            space=space,
        )
        ax.set_title(param_row.name + "\n")
        ax.yaxis.tick_right()
        ax.xaxis.tick_top()

        for col in range(row):
            # plot two-variable dependence on lower triangle
            param_col = space.params[col]
            print("[INFO] dependence plot {} x {} ({} x {})".format(
                row, col, param_row.name, param_col.name))
            x_row_transformed, x_col_transformed, ys = partial_dependence(
                model, row, col,
                samples_transformed=samples_transformed,
                n_points=quality.points)
            x_row = param_row.from_transformed_a(x_row_transformed)
            x_col = param_col.from_transformed_a(x_col_transformed)
            ax_ij = axes[row, col]
            plot_dual_variable_dependence(
                ax_ij, col, row,
                xi=x_col, yi=x_row, zi=ys, xs=x_observed, x_min=x_min,
                param_x=param_col, param_y=param_row,
                contour_levels=style.contour_levels, cmap=style.cmap,
                show_xticks=(row == n_dims - 1),
                show_yticks=(col == 0))

        # hide top right triangle
        for col in range(row + 1, n_dims):
            axes[row, col].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(
        # left=0.05, right=0.95, bottom=0.05, top=0.95,
        hspace=0.03, wspace=0.03)
    return fig, axes


def plot_single_variable_dependence(
    ax, i, *,
    xs, ys,
    xi, yi, stdi,  # pylint: disable=invalid-name
    x_min,
    space,
    scatter_args: dict = dict(c='g', s=10, lw=0, alpha=0.5),
    minline_args: dict = dict(linestyle='--', color='r', lw=1),
):
    param = space.params[i]

    ax.fill_between(xi, yi - 1.96*stdi, yi + 1.96*stdi, color='b', alpha=0.15)
    ax.scatter([x[i] for x in xs], ys, **scatter_args)
    ax.plot(xi, yi, c='b')
    if x_min is not None:
        ax.axvline(x_min[i], **minline_args)
    x_bounds = param.bounds()
    if x_bounds is not None:
        ax.set_xlim(*x_bounds)


def plot_dual_variable_dependence(
    ax, i, j, *,
    xi, yi, zi,  # pylint: disable=invalid-name
    xs,
    x_min,
    param_x, param_y,
    contour_levels,
    cmap,
    scatter_args: dict = dict(c='k', s=10, lw=0),
    xmin_scatter_args: dict = dict(c='r', s=10, lw=0),
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


def plot_convergence(all_evaluations: t.List[Individual]):
    # find the minimum at each generation
    n_generations = max(ind.gen for ind in all_evaluations)
    ind_by_generation: t.List[t.List[Individual]] = \
        [[] for _ in range(n_generations + 1)]
    for ind in all_evaluations:
        ind_by_generation[ind.gen].append(ind)

    prev_min_fitness = np.inf
    min_fitness = []
    for generation in ind_by_generation:
        fitness = min(ind.fitness for ind in generation)
        prev_min_fitness = min(prev_min_fitness, fitness)
        min_fitness.append(prev_min_fitness)

    fig, (utility_ax, ei_ax) = plt.subplots(2, 1, sharex=True)
    palette = sns.color_palette('husl')

    # utility/fitness plot
    sns.stripplot(
        x=[ind.gen for ind in all_evaluations],
        y=[ind.fitness for ind in all_evaluations],
        jitter=True,
        ax=utility_ax,
        palette=palette,
    )

    # min-fitness plot
    utility_ax.plot(
        list(range(len(min_fitness))),
        min_fitness,
        color='blue',
    )

    # EI plot
    sns.stripplot(
        x=[ind.gen for ind in all_evaluations],
        y=[ind.ei for ind in all_evaluations],
        jitter=True,
        ax=ei_ax,
        palette=palette,
    )
    ei_ax.set_yscale('log')  # EI can be very small

    fig.tight_layout()
    return fig, (utility_ax, ei_ax)


def plot_observations_against_model(
    model: SurrogateModel, all_evaluations: t.List[Individual], *,
    ax=None, markersize: int = 10,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    observed_data = np.array([ind.fitness for ind in all_evaluations])
    modelled_data = model.predict_a(
        [ind.sample for ind in all_evaluations], return_std=False)
    # plot the data
    ax.plot(
        observed_data, modelled_data,
        marker='o', alpha=0.5, markersize=markersize, color='g', ls='')
    # plot 45° line
    observed_min, observed_max = np.min(observed_data), np.max(observed_data)
    modelled_min, modelled_max = np.min(modelled_data), np.max(modelled_data)
    the_min = min(observed_min, modelled_min)
    the_max = max(observed_max, modelled_max)
    ax.plot(
        [the_min, the_max], [the_min, the_max],
        marker='', ls='-', color='b')
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')

    if fig is not None:
        fig.tight_layout()
    return fig
