from abc import ABC, abstractmethod
from typing import List

import torch


class TerminationCriterion(ABC):

    @abstractmethod
    def may_continue(self, iteration: int, loss: float) -> bool:
        pass


class MaxIterationTerminationCriterion(TerminationCriterion):

    def __init__(self, maxiter: int):
        self.maxiter = maxiter

    def may_continue(self, iteration: int, loss: float) -> bool:
        return iteration < self.maxiter


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
        if self.current_loss_index > self.itercount:
            self.current_loss_index -= self.itercount

        if self.current_loss_iteration < self.itercount:
            self.current_loss_iteration = iteration
            self.last_losses[self.current_loss_index] = loss
            return True

        self.current_loss_iteration = iteration
        if self.last_losses[self.current_loss_index] - loss < self.relchange:
            self.last_losses[self.current_loss_index] = loss
            return False

        self.last_losses[self.current_loss_index] = loss
        return True


class CombinedTerminationCriterion(TerminationCriterion):

    def __init__(self, criteria: List[TerminationCriterion]):
        self.criteria = criteria

    def may_continue(self, iteration: int, loss: float) -> bool:
        return all([x.may_continue(iteration, loss) for x in self.criteria])