import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import numpy as np  # type: ignore
import typing as t
from .minimize import Individual

from . import SurrogateModel, Space


def partial_dependence(
    space, model, i, j=None, *,
    samples_transformed, n_points, n_samples=None, rng,
) -> tuple:

    if samples_transformed is None:
        assert n_samples is not None, \
            "n_samples required to generate samples_transformed"
        samples_transformed = np.array([
            space.into_transformed(space.sample(rng=rng))
            for _ in range(n_samples)
        ])

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
            # IDEALLY: std(a + b) = sqrt(var(a) + var(b) - cov(a, b))
            # NOT: mean of stdev
            # INSTEAD: mean of variance
            stdi[n] = np.sqrt(np.mean(std**2))
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


def plot_objective(
    xs, ys, *,
    x_min=None,
    model: SurrogateModel,
    space: Space,
    contour_levels: int=10,
    n_points: int=40,
    n_samples: int=250,
    subplot_size: float=2,
    rng: np.random.RandomState,
    cmap='viridis_r',
    contour_filled: bool = True,
    contour_lines: bool = False,
    contour_scatter_args=None,
    contour_xmin_scatter_args=None,
    contour_args=None,
    contour_filled_args=None,
    contour_lines_args=None,
) -> t.Tuple[t.Any, t.Any]:
    n_dims = space.n_dims

    if x_min is None:
        x_min = xs[np.argmin(ys)]

    samples_transformed = [
        space.into_transformed(space.sample(rng=rng))
        for _ in range(n_samples)
    ]

    fig, ax = plt.subplots(
        n_dims, n_dims,
        figsize=(subplot_size * n_dims, subplot_size * n_dims),
        squeeze=False)

    for row in range(n_dims):
        # plot single-variable dependence on diagonal
        param_row = space.params[row]
        print("[INFO] dependence plot 1D {} ({})".format(row, param_row.name))
        ax_ii = ax[row, row]
        plot_single_variable_dependence(
            ax_ii, row,
            xs=xs, ys=ys, x_min=x_min,
            space=space,
            model=model,
            samples_transformed=samples_transformed,
            n_points=n_points,
            n_samples=n_samples,
            rng=rng,
        )
        ax_ii.set_title(param_row.name + "\n")
        ax_ii.yaxis.tick_right()
        ax_ii.xaxis.tick_top()

        for col in range(row):
            # plot two-variable dependence on lower triangle
            param_col = space.params[col]
            print("[INFO] dependence plot {} x {} ({} x {})".format(
                row, col, param_row.name, param_col.name))
            xi_transformed, yi_transformed, zi = partial_dependence(
                space, model, row, col,
                samples_transformed=samples_transformed,
                n_points=n_points,
                rng=rng)
            xi = param_col.from_transformed_a(xi_transformed)
            yi = param_row.from_transformed_a(yi_transformed)
            ax_ij = ax[row, col]
            plot_dual_variable_dependence(
                ax_ij, col, row,
                xi=xi, yi=yi, zi=zi, xs=xs, x_min=x_min,
                param_x=param_col, param_y=param_row,
                contour_levels=contour_levels, cmap=cmap,
                show_xticks=(row == n_dims - 1),
                show_yticks=(col == 0),
                filled=contour_filled,
                lines=contour_lines,
                scatter_args=contour_scatter_args,
                xmin_scatter_args=contour_xmin_scatter_args,
                contour_args=contour_args,
                contour_filled_args=contour_filled_args,
                contour_lines_args=contour_lines_args,
            )


        # hide top right triangle
        for col in range(row + 1, n_dims):
            ax[row, col].axis('off')

    fig.tight_layout()
    fig.subplots_adjust(
        # left=0.05, right=0.95, bottom=0.05, top=0.95,
        hspace=0.03, wspace=0.03)
    return fig, ax


def plot_single_variable_dependence(
    ax, i, *,
    xs, ys,
    x_min,
    space, model,
    samples_transformed,
    n_points: int,
    n_samples: int = None,
    rng: np.random.RandomState,
    scatter_args=dict(c='g', s=10, lw=0, alpha=0.5),
    minline_args=dict(linestyle='--', color='r', lw=1),
):
    param = space.params[i]

    xi_transformed, yi, stdi = partial_dependence(
            space, model, i,
            samples_transformed=None,
            n_points=n_points,
            n_samples=n_samples,
            rng=rng)
    xi = param.from_transformed_a(xi_transformed)

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
    xi, yi, zi,
    xs,
    x_min,
    param_x, param_y,
    contour_levels,
    cmap,
    scatter_args=None,
    xmin_scatter_args=None,
    show_xticks: bool,
    show_yticks: bool,
    filled: bool = True,
    lines: bool = False,
    contour_args=None,
    contour_filled_args=None,
    contour_lines_args=None,
):
    cfa = dict(locator=None, cmap=cmap, alpha=0.8)
    for args in (contour_args, contour_filled_args):
        if args is not None:
            cfa.update(args)

    cla = dict(locator=None, colors='k', linewidths=1)
    for args in (contour_args, contour_lines_args):
        if args is not None:
            cla.update(args)

    sca = dict(c='k', s=10, lw=0)
    if scatter_args is not None:
        sca.update(scatter_args)

    xsca = dict(sca)
    xsca.update(c='r')
    if xmin_scatter_args is not None:
        xsca.update(xmin_scatter_args)

    # start drawing plots

    if filled:
        ax.contourf(xi, yi, zi, contour_levels, **cfa)

    if lines:
        ax.contour(xi, yi, zi, contour_levels, **cla)

    ax.scatter([x[i] for x in xs], [x[j] for x in xs], **sca)
    ax.scatter(x_min[i], x_min[j], **xsca)
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
        f = min(ind.fitness for ind in generation)
        prev_min_fitness = min(prev_min_fitness, f)
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
    ax=None, markersize=10,
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
