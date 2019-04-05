import typing as t

import attr
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

from .minimize import Individual
from . import SurrogateModel, Space

ProgressCB = t.Callable[[str, t.Optional[str]], None]


def _default_progress_cb(_param_1: str, _param_2: t.Optional[str]) -> None:
    pass


@attr.s(frozen=True, auto_attribs=True)
class DualDependenceStyle:
    """Control the appearance of the parameter interaction visualization.

    The interaction (contour) plot has four layers
    that can be configured separately:

    -   filled contour plot: :meth:`get_contour_filled_args`
    -   contour lines: :meth:`get_contour_line_args`
    -   scatter plot of all samples: :meth:`get_scatter_args`
    -   “scatter” plot of the best sample: :meth:`get_xmin_scatter_args`

    Parameters
    ----------
    cmap
        The colour map used for the filled contour plot.
        Defaults to 'viridis_r'.
    contour_args
        Extra arguments for the contour plot (both filled and lines).
    contour_filled
        Whether filled contours are drawn.
        Either this or *countour_lines* should be True.
        Defaults to True.
    contour_filled_args
        Extra arguments for the filled contour plot, overrides *contour_args*.
    contour_levels
        Defaults to 10.
    contour_lines
        Whether contour lines are drawn.
        Either this or *contour_filled* should be True.
    contour_lines_args
        Extra arguments the line contour plot, overrides *contour_args*.
    contour_scatter_args
        Extra arguments for the scatter plot of all samples.
    xmin_scatter_args
        Extra arguments to override the scatter plot appearance
        of the best point.
    subplot_size
        How large each plot in the grid of all parameters should be.
        The whole figure will have size (*n_params* × *subplot_size*)².
    """

    cmap: str = 'viridis_r'
    contour_args: t.Optional[dict] = None
    contour_filled: bool = True
    contour_filled_args: t.Optional[dict] = None
    contour_levels: int = 10
    contour_lines: bool = False
    contour_lines_args: t.Optional[dict] = None
    contour_scatter_args: t.Optional[dict] = None
    contour_xmin_scatter_args: t.Optional[dict] = None
    subplot_size: float = 2
    scatter_args: t.Optional[dict] = None
    xmin_scatter_args: t.Optional[dict] = None

    def get_contour_filled_args(self) -> dict:
        """Filled contour plot arguments.

        1. locator: None, cmap: *cmap*, alpha: 0.8
        2. *contour_args*
        3. *contour_filled_args*
        """
        return _merge_dicts(
            dict(locator=None, cmap=self.cmap, alpha=0.8),
            self.contour_args,
            self.contour_filled_args,
        )

    def get_contour_line_args(self) -> dict:
        """Contour line plot arguments.

        1. locator: None, colors: 'k', linewidths: 1
        2. *contour_args*
        3. *contour_lines_args*
        """
        return _merge_dicts(
            dict(locator=None, colors='k', linewidths=1),
            self.contour_args,
            self.contour_lines_args,
        )

    def get_scatter_args(self) -> dict:
        """Scatter plot arguments.

        1. c: 'k', s: 10, lw: 0
        2. *scatter_args*
        """
        return _merge_dicts(
            dict(c='k', s=10, lw=0),
            self.scatter_args,
        )

    def get_xmin_scatter_args(self) -> dict:
        """“Scatter” plot of the best sample arguments.

        1. :meth:`get_scatter_args`
        2. c: 'r'
        3. *xmin_scatter_args*
        """
        return _merge_dicts(
            self.get_scatter_args(),
            dict(c='r'),
            self.xmin_scatter_args,
        )


class PartialDependence:
    """Visualize and analyze individual contributions of each parameter.

    Parameters
    ----------
    model
    space
    rng
    resolution
        How many samples are used along one parameter.
        Default: 20.
    quality
        How many samples are used along all other parameters
        to get a precise estimate of the average value.
        Default: 250.
    """

    def __init__(
        self, *,
        model: SurrogateModel,
        space: Space,
        rng: RandomState,
        resolution: int = 40,
        quality: int = 250,
    ) -> None:
        self.model: SurrogateModel = model
        self.space: Space = space
        self._resolution: int = resolution
        self._samples_transformed: np.ndarray = np.array([
            space.into_transformed(space.sample(rng=rng))
            for _ in range(quality)
        ])

    def along_one_dimension(
        self, dim: int,
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate contributions along one dimension.

        Returns
        -------
        xs
            Sample locations along this dimension.
        ys_mean
            Mean response at the samples.
        ys_min
            Minimal/optimal response at the samples.
        ys_mean_std
            Mean model uncertainty.
        ys_min_std
            Model uncertainty at the minimum.
        """

        samples_transformed = np.array(self._samples_transformed)
        xs_transformed = np.linspace(0.0, 1.0, self._resolution)

        ys_mean = np.zeros(self._resolution)
        ys_min = np.zeros(self._resolution)
        ys_mean_std = np.zeros(self._resolution)
        ys_min_std = np.zeros(self._resolution)

        for i, x in enumerate(xs_transformed):
            samples_transformed[:, dim] = x
            y, std = self.model.predict_transformed_a(samples_transformed)
            assert std is not None

            ys_mean[i] = np.mean(y)
            # IDEALLY: std(a + b) = sqrt(var(a) + var(b) - cov(a, b))
            # NOT: mean of stdev
            # INSTEAD: mean of variance
            ys_mean_std[i] = np.sqrt(np.mean(std**2))

            y_min_loc = np.argmin(y)
            ys_min[i] = y[y_min_loc]
            ys_min_std[i] = std[y_min_loc]

        xs = self.space.params[dim].from_transformed_a(xs_transformed)
        return xs, ys_mean, ys_min, ys_mean_std, ys_min_std

    def along_two_dimensions(
        self, dim_1: int, dim_2: int,
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate contributions along two dimensions.

        Returns
        -------
        xs_1
            Sample locations along *dim_1*
        xs_2
            Sample locations along *dim_2*.
        ys
            Mean response at the samples.
        """

        samples_transformed = np.array(self._samples_transformed)
        xs_transformed = np.linspace(0.0, 1.0, self._resolution)

        ys = np.zeros((self._resolution, self._resolution))

        for i, x_1 in enumerate(xs_transformed):
            for j, x_2 in enumerate(xs_transformed):
                samples_transformed[:, (dim_1, dim_2)] = (x_1, x_2)
                y, _std = self.model.predict_transformed_a(
                    samples_transformed, return_std=False)
                ys[i, j] = np.mean(y)

        xs_1 = self.space.params[dim_1].from_transformed_a(xs_transformed)
        xs_2 = self.space.params[dim_2].from_transformed_a(xs_transformed)
        return xs_1, xs_2, ys

    def plot_grid(
        self, x_observed: np.ndarray, y_observed: np.ndarray, *,
        x_min=None,
        style: DualDependenceStyle = None,
        progress_cb: ProgressCB = _default_progress_cb,
    ) -> t.Tuple[t.Any, t.Any]:
        """Plot a visualization of parameter influences.

        Parameters
        ----------
        x_observed
        y_observed
        x_min: list or None
            Location of the best sample.
            Defaults to the sample that minimizes *y_observed*.
        style
        progress_cb: ``(dim_1_name, dim_2_name?) -> None``
            Called prior to rendering each sub-plot
            with the names of the parameters in the sub-plot.
            The *dim_2_name* is only provided for interaction plots.

        Returns
        -------
        (fig, axes): tuple
            The plotted figure.
        """

        n_dims = self.space.n_dims

        if style is None:
            style = DualDependenceStyle()
        assert style is not None  # for type checker

        if x_min is None:
            x_min = x_observed[np.argmin(y_observed)]

        fig, axes = plt.subplots(
            n_dims, n_dims,
            figsize=(style.subplot_size * n_dims, style.subplot_size * n_dims),
            squeeze=False)

        for row in range(n_dims):
            # plot single-variable dependence on diagonal
            param_row = self.space.params[row]
            progress_cb(param_row.name, None)

            ax = axes[row, row]
            plot_single_variable_dependence(
                ax, row,
                x_observed=x_observed,
                y_observed=y_observed,
                x_observed_min=x_min,
                partial_dependence=self,
            )

            ax.set_title(param_row.name + "\n")
            ax.yaxis.tick_right()
            ax.xaxis.tick_top()

            for col in range(row):
                # plot two-variable dependence on lower triangle
                param_col = self.space.params[col]
                progress_cb(param_row.name, param_col.name)

                ax = axes[row, col]
                plot_dual_variable_dependence(
                    ax, col, row,
                    partial_dependence=self,
                    x_observed=x_observed, x_observed_min=x_min,
                    style=style,
                )

                if row != n_dims - 1:
                    ax.tick_params(
                        'x', top=False, bottom=False, labelbottom=False)
                if col != 0:
                    ax.tick_params(
                        'y', left=False, right=False, labelleft=False)

            # hide top right triangle
            for col in range(row + 1, n_dims):
                axes[row, col].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(
            # left=0.05, right=0.95, bottom=0.05, top=0.95,
            hspace=0.03, wspace=0.03)
        return fig, axes


def plot_single_variable_dependence(
    ax, dim, *,
    x_observed, y_observed, x_observed_min,
    partial_dependence: PartialDependence,
    scatter_args=dict(c='g', s=10, lw=0, alpha=0.5),
    minline_args=dict(linestyle='--', color='r', lw=1),
    show_mean: bool = True, show_mean_err: bool = True,
    show_min: bool = True, show_min_err: bool = True,
    cmean: str = 'b',
    cmin: str = 'r',
    mean_args: dict = dict(),
    min_args: dict = dict(),
) -> None:

    xs, ys_mean, ys_min, ys_mean_std, ys_min_std = \
        partial_dependence.along_one_dimension(dim)

    ax.scatter([x[dim] for x in x_observed], y_observed, **scatter_args)

    if show_mean and show_mean_err:
        ax.fill_between(
            xs,
            ys_mean - 2*ys_mean_std,
            ys_mean + 2*ys_mean_std,
            color=cmean, alpha=0.15)
    if show_min and show_min_err:
        ax.fill_between(
            xs,
            ys_min - 2*ys_min_std,
            ys_min + 2*ys_min_std,
            color=cmin, alpha=0.15)
    if show_mean:
        ax.plot(xs, ys_mean, c=cmean, **mean_args)
    if show_min:
        ax.plot(xs, ys_min, c=cmin, **min_args)

    if x_observed_min is not None:
        ax.axvline(x_observed_min[dim], **minline_args)

    bounds = partial_dependence.space.params[dim].bounds()
    if bounds is not None:
        ax.set_xlim(*bounds)


def plot_dual_variable_dependence(
    ax, dim_1, dim_2, *,
    partial_dependence: PartialDependence,
    x_observed,
    x_observed_min,
    style: DualDependenceStyle,
) -> None:

    x_2, x_1, ys = partial_dependence.along_two_dimensions(dim_2, dim_1)

    if style.contour_filled:
        ax.contourf(
            x_1, x_2, ys, style.contour_levels,
            **style.get_contour_filled_args())

    if style.contour_lines:
        ax.contour(
            x_1, x_2, ys, style.contour_levels,
            **style.get_contour_line_args())

    ax.scatter(
        [x[dim_1] for x in x_observed],
        [x[dim_2] for x in x_observed],
        **style.get_scatter_args(),
    )
    if x_observed_min is not None:
        ax.scatter(
            x_observed_min[dim_1],
            x_observed_min[dim_2],
            **style.get_xmin_scatter_args(),
        )

    bounds_1 = partial_dependence.space.params[dim_1].bounds()
    bounds_2 = partial_dependence.space.params[dim_2].bounds()
    if bounds_1 is not None:
        ax.set_xlim(*bounds_1)
    if bounds_2 is not None:
        ax.set_ylim(*bounds_2)


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
        observation = min(ind.observation for ind in generation)
        prev_min_fitness = min(prev_min_fitness, observation)
        min_fitness.append(prev_min_fitness)

    fig, (utility_ax, ei_ax) = plt.subplots(2, 1, sharex=True)
    palette = sns.color_palette('husl')

    # utility/fitness plot
    sns.stripplot(
        x=[ind.gen for ind in all_evaluations],
        y=[ind.observation for ind in all_evaluations],
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
        y=[ind.expected_improvement for ind in all_evaluations],
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

    observed_data = np.array([ind.observation for ind in all_evaluations])
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


def _merge_dicts(*optional_dictionaries: t.Optional[dict]) -> dict:
    merged = dict()
    for optional_dict in optional_dictionaries:
        if optional_dict is not None:
            merged.update(optional_dict)
    return merged
