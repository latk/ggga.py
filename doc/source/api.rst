API
===

This document describes the public API.

.. autosummary::
   :nosignatures:

   ggga.minimize.Minimizer
   ggga.minimize.OptimizationResult
   ggga.individual.Individual
   ggga.space.Space
   ggga.surrogate_model.SurrogateModel
   ggga.acquisition.AcquisitionStrategy
   ggga.outputs.OutputEventHandler
   ggga.outputs.Output

.. class:: mtrand.RandomState

   see :class:`numpy.random.RandomState`

Minimizer
---------
.. autoclass:: ggga.minimize.Minimizer

OptimizationResult
------------------
.. autoclass:: ggga.minimize.OptimizationResult

Individual
----------
.. autoclass:: ggga.individual.Individual

Space
-----
.. autoclass:: ggga.space.Space

SurrogateModel
--------------
.. autoclass:: ggga.surrogate_model.SurrogateModel

.. class:: ggga.gpr.SurrogateModelGPR

.. class:: ggga.knn.SurrogateModelKNN

AcquisitionStrategy
-------------------
.. autoclass:: ggga.acquisition.AcquisitionStrategy
.. autoclass:: ggga.acquisition.ChainedAcquisition
   :no-members:
.. autoclass:: ggga.acquisition.HedgedAcquisition
   :no-members:
.. autoclass:: ggga.acquisition.RandomReplacementAcquisition
   :no-members:
.. autoclass:: ggga.acquisition.MutationAcquisition
   :no-members:
.. autoclass:: ggga.acquisition.RandomWalkAcquisition
   :no-members:
.. autoclass:: ggga.acquisition.GradientAcquisition
   :no-members:

OutputEventHandler
------------------
.. autoclass:: ggga.outputs.OutputEventHandler

Output
------
.. autoclass:: ggga.outputs.Output

   .. automethod:: add

PartialDependence
-----------------

.. autoclass:: ggga.visualization.PartialDependence

DualDependenceStyle
-------------------

.. autoclass:: ggga.visualization.DualDependenceStyle

SingleDependenceStyle
---------------------

.. autoclass:: ggga.visualization.SingleDependenceStyle

benchmark_functions
-------------------

.. currentmodule:: ggga.benchmark_functions

A collection of optimization benchmark functions
that can be used via the example runner.
Some of them do not have a fixed number of parameters
and can be implicitly used as any n-dimensional version.
Read their docstrings for more information on behaviour, bounds, and optima.

.. function:: goldstein_price(x_1, x_2)

.. function:: easom(x_1, x_2, *, amplitude?)

.. function:: himmelblau(x_1, x_2)

.. function:: rastrigin(*xs, amplitude?)

.. function:: rosenbrock(*xs)

.. function:: sphere(*xs)

.. function:: onemax(*xs)

.. function:: trap(*xs, p_well?)
