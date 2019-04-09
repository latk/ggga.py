import csv
import json
import operator
import sys
import typing as t

import numpy as np  # type: ignore

from .individual import Individual
from .space import Space, Real
from .surrogate_model import SurrogateModel
from .util import tabularize


class IndividualsToTable:
    def __init__(self, space: Space) -> None:
        self.columns: t.List[str] = []
        self.columns.extend(
            'gen utility prediction ei cost'.split())
        self.columns.extend(
            f"param_{param.name}" for param in space.params)

        self.formats: t.List[str] = []
        self.formats.extend(
            '{:2d} {:.2f} {:.2f} {:.2e} {:.2f}'.split())
        self.formats.extend(
            '{:.5f}' if isinstance(p, Real) else '{}'
            for p in space.params)

    @staticmethod
    def individual_to_row(ind: Individual) -> t.Iterable:
        return IndividualsToTable.observation_to_row(
            gen=ind.gen, observation=ind.observation,
            prediction=ind.prediction,
            expected_improvement=ind.expected_improvement,
            cost=ind.cost, sample=ind.sample,
        )

    @staticmethod
    def observation_to_row(
        *, sample: list, observation: float, prediction: float,
        expected_improvement: float, cost: float, gen: int,
    ) -> t.Iterable:
        yield gen
        yield observation
        yield prediction
        yield expected_improvement
        yield cost
        yield from sample

    def row_to_individual(
            self, row: t.Iterable[str], gen: int = None,
    ) -> Individual:
        row = list(row)
        assert len(row) == len(self.columns)

        (original_gen, observation, prediction,
         expected_improvement, cost, *sample) = row

        if gen is None:
            gen = int(original_gen)

        return Individual(
            [float(x) for x in sample],
            observation=float(observation),
            gen=gen,
            expected_improvement=float(expected_improvement),
            prediction=float(prediction),
            cost=float(cost),
        )


class OutputEventHandler:
    r"""Report progress and save results during optimization process.
    (interface)
    """

    def event_new_generation(
        self, gen: int, *, relscale: t.Tuple[float],
    ) -> None:
        """Called when a new generation is started."""

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *, duration: float,
    ) -> None:
        """Called when evaluations of a generation have completed."""

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *, duration: float,
    ) -> None:
        """Called when a new model has been trained."""

    def event_acquisition_completed(
        self, *, duration: float,
    ) -> None:
        """Called when new samples have been acquired."""


class CompositeOutputEventHandler(OutputEventHandler):

    def __init__(self, *subloggers: OutputEventHandler) -> None:
        self.subloggers = list(subloggers)

    def add(self, logger: OutputEventHandler) -> None:
        self.subloggers.append(logger)

    def event_new_generation(
        self, gen: int, *, relscale: t.Tuple[float],
    ) -> None:
        for logger in self.subloggers:
            logger.event_new_generation(gen, relscale=relscale)

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *, duration: float,
    ) -> None:
        for logger in self.subloggers:
            logger.event_evaluations_completed(individuals, duration=duration)

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *, duration: float,
    ) -> None:
        for logger in self.subloggers:
            logger.event_model_trained(generation, model, duration=duration)

    def event_acquisition_completed(
        self, *, duration: float,
    ) -> None:
        for logger in self.subloggers:
            logger.event_acquisition_completed(duration=duration)


class RecordTrainedModels(OutputEventHandler):
    def __init__(self, model_file: t.TextIO) -> None:
        self._model_file = model_file

        assert hasattr(model_file, 'write'), \
            f"Model output must be a writable file object: {model_file!r}"

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *,
        duration: float,  # pylint: disable=unused-argument
    ) -> None:

        json.dump(
            [generation, model.to_jsonish()],
            self._model_file,
            default=self._coerce_to_jsonish)
        print(file=self._model_file)

    @staticmethod
    def _coerce_to_jsonish(some_object):
        if isinstance(some_object, np.ndarray):
            return list(some_object)
        raise TypeError(f"cannot encode as JSON: {some_object!r}")


class WriteHumanReadableOutput(OutputEventHandler):
    def __init__(
        self, *,
        log_file: t.TextIO,
        individuals_table: IndividualsToTable,
    ) -> None:
        self._file = log_file
        self._individuals_table = individuals_table

    def log(self, msg: str, *, level: str = 'INFO') -> None:
        assert level == 'INFO'
        marker = f"[{level}]"

        first = True
        for line in msg.splitlines():
            if first:
                print(marker, line, file=self._file)
                first = False
            else:
                print(" " * len(marker), line, file=self._file)

    def event_new_generation(
        self, gen: int, *, relscale: t.Tuple[float],
    ) -> None:
        formatted_relscale = ' '.join(format(r, '.5') for r in relscale)
        self.log(f"starting generation #{gen} (relscale {formatted_relscale})")

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *, duration: float,
    ) -> None:
        self.log(f'evaluations ({duration} s):\n' + tabularize(
            header=self._individuals_table.columns,
            formats=self._individuals_table.formats,
            data=[
                list(self._individuals_table.individual_to_row(ind))
                for ind in sorted(individuals,
                                  key=operator.attrgetter('observation'))],
        ))

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *, duration: float,
    ) -> None:
        self.log(
            f"trained new model ({duration} s):\n"
            f"{model!r}")


class RecordCompletedEvaluations(OutputEventHandler):
    def __init__(
        self, csv_file: t.TextIO, *,
        individuals_table: IndividualsToTable,
        write_header: bool = True,
    ) -> None:
        assert hasattr(csv_file, 'write'), \
            f"Evaluation CSV file must be a writable file object: {csv_file!r}"
        self._csv_writer = csv.writer(csv_file)
        self._individuals_table = individuals_table
        if write_header:
            self._csv_writer.writerow(individuals_table.columns)

    @classmethod
    def new(
        cls, csv_file: t.TextIO, *, space: Space, write_header: bool = True,
    ) -> 'RecordCompletedEvaluations':
        return cls(
            csv_file,
            individuals_table=IndividualsToTable(space),
            write_header=write_header,
        )

    @staticmethod
    def load_individuals(
            csv_file: t.TextIO, *,
            space: Space,
            gen: int = None,
    ) -> t.Iterable[Individual]:
        table = IndividualsToTable(space)
        reader = csv.reader(csv_file)
        next(reader, None)  # skip header
        for row in reader:
            yield table.row_to_individual(row, gen=gen)

    def write_result(
        self, *,
        sample: list,
        observation: float,
        gen: int = 0,
        expected_improvement: float = 0.0,
        prediction: float = 0.0,
        cost: float = 0.0,
    ) -> None:
        self._csv_writer.writerow(
            self._individuals_table.observation_to_row(
                sample=sample, observation=observation, gen=gen,
                expected_improvement=expected_improvement,
                prediction=prediction, cost=cost,
            ))

    def write_individual(self, ind: Individual):
        self._csv_writer.writerow(
            self._individuals_table.individual_to_row(ind))

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *,
        duration: float,  # pylint: disable=unused-argument
    ) -> None:
        for ind in individuals:
            self.write_individual(ind)


class Output(CompositeOutputEventHandler):
    r"""
    Control the output during optimization.
    Default implementation of :class:`OutputEventHandler`.

    Parameters
    ----------
    space : Space
        The parameter space.
    evaluation_csv_file : typing.TextIO, optional
        If present, all evaluations are recorded in this file.
    model_file : typing.TextIO, optional
        If present, metadata of the models is recorded in this file,
        using a JSON-per-line format.
    log_file : typing.TextIO, optional
        Where to write human-readable output. Defaults to sys.stdout.
        If set to None, output is suppressed.
    """

    def __init__(
        self, *,
        space: Space,
        evaluation_csv_file: t.Optional[t.TextIO] = None,
        model_file: t.Optional[t.TextIO] = None,
        log_file: t.Optional[t.TextIO] = sys.stdout,
    ) -> None:

        super().__init__()

        individuals_table = IndividualsToTable(space)

        self.space: Space = space

        if log_file is not None:
            self.add(WriteHumanReadableOutput(
                log_file=log_file, individuals_table=individuals_table))

        if model_file is not None:
            self.add(RecordTrainedModels(model_file))

        if evaluation_csv_file is not None:
            self.add(RecordCompletedEvaluations(
                evaluation_csv_file, individuals_table=individuals_table))

        # durations
        self.acquisition_durations: t.List[float] = []
        """list[float]: time spent per acquisition"""
        self.evaluation_durations: t.List[float] = []
        """list[float]: time spent per evaluation"""
        self.training_durations: t.List[float] = []
        """list[float]: time spent per training/fitting"""

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *,
        duration: float,
    ) -> None:
        self.evaluation_durations.append(duration)

        super().event_evaluations_completed(individuals, duration=duration)

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *,
        duration: float,
    ) -> None:
        self.training_durations.append(duration)

        super().event_model_trained(generation, model, duration=duration)

    def event_acquisition_completed(self, *, duration: float) -> None:
        self.acquisition_durations.append(duration)

        super().event_acquisition_completed(duration=duration)


__all__ = [
    Output.__name__,
    OutputEventHandler.__name__,
    RecordCompletedEvaluations.__name__,
]
