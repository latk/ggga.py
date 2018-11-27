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
        yield ind.gen
        yield ind.observation
        yield ind.prediction
        yield ind.expected_improvement
        yield ind.cost
        yield from ind.sample


class OutputEventHandler:
    r"""Interface: Event handlers as the optimization progresses."""

    def event_new_generation(
        self, gen: int, *, relscale: t.Tuple[float],
    ) -> None:
        pass

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *, duration: float,
    ) -> None:
        pass

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *, duration: float,
    ) -> None:
        pass

    def event_acquisition_completed(
        self, *, duration: float,
    ) -> None:
        pass


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
        self, csv_file: t.TextIO, *, individuals_table: IndividualsToTable,
    ) -> None:
        assert hasattr(csv_file, 'write'), \
            f"Evaluation CSV file must be a writable file object: {csv_file!r}"
        self._csv_writer = csv.writer(csv_file)
        self._csv_writer.writerow(individuals_table.columns)
        self._individuals_table = individuals_table

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *,
        duration: float,  # pylint: disable=unused-argument
    ) -> None:
        for ind in individuals:
            self._csv_writer.writerow(
                self._individuals_table.individual_to_row(ind))


class Output(CompositeOutputEventHandler):
    r"""
    Control the output during optimization.

    Attributes:
        space (Space):
            The parameter space.
        evaluation_csv_file (TextIO, optional):
            If present, all evaluations are recorded in this file.
        model_file (TextIO, optional):
            If present, metadata of the models is recorded in this file,
            using a JSON-per-line format.
        log_file (TextIO, optional):
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
        self.evaluation_durations: t.List[float] = []
        self.training_durations: t.List[float] = []

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
]
