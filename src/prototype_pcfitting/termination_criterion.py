from abc import ABC, abstractmethod
from typing import List

import torch


class TerminationCriterion(ABC):

    @abstractmethod
    def may_continue(self, iteration: int, loss: float) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass


class MaxIterationTerminationCriterion(TerminationCriterion):

    def __init__(self, maxiter: int):
        self.maxiter = maxiter

    def may_continue(self, iteration: int, loss: float) -> bool:
        return iteration < self.maxiter

    def reset(self):
        pass


class RelChangeTerminationCriterion(TerminationCriterion):

    def __init__(self, relchange: float, itercount: int):
        self.relchange = relchange
        self.itercount = itercount
        self.last_losses = torch.zeros(itercount)
        self.current_loss_index = -1
        self.current_loss_iteration = -1

    def may_continue(self, iteration: int, loss: float) -> bool:
        # has to be called every iteration!

        # get last loss
        self.current_loss_index += 1
        if self.current_loss_index >= self.itercount:
            self.current_loss_index -= self.itercount

        # before itercount iterations happened
        if self.current_loss_iteration < self.itercount:
            self.current_loss_iteration = iteration
            self.last_losses[self.current_loss_index] = loss
            return True

        # check change
        self.current_loss_iteration = iteration
        if self.last_losses[self.current_loss_index] - loss < self.relchange:
            # change too small!
            self.last_losses[self.current_loss_index] = loss
            return False

        self.last_losses[self.current_loss_index] = loss
        return True

    def reset(self):
        self.last_losses = torch.zeros(self.itercount)
        self.current_loss_index = -1
        self.current_loss_iteration = -1


class AndCombinedTerminationCriterion(TerminationCriterion):

    def __init__(self, criteria: List[TerminationCriterion]):
        self.criteria = criteria

    def may_continue(self, iteration: int, loss: float) -> bool:
        return all([x.may_continue(iteration, loss) for x in self.criteria])

    def reset(self):
        for c in self.criteria:
            c.reset()


class OrCombinedTerminationCriterion(TerminationCriterion):

    def __init__(self, criteria: List[TerminationCriterion]):
        self.criteria = criteria

    def may_continue(self, iteration: int, loss: float) -> bool:
        return any([x.may_continue(iteration, loss) for x in self.criteria])

    def reset(self):
        for c in self.criteria:
            c.reset()
