from abc import ABC, abstractmethod
from typing import List

import torch


class TerminationCriterion(ABC):

    @abstractmethod
    def may_continue(self, iteration: int, losses: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self):
        pass


class MaxIterationTerminationCriterion(TerminationCriterion):

    def __init__(self, maxiter: int):
        self.maxiter = maxiter

    def may_continue(self, iteration: int, losses: torch.Tensor) -> torch.Tensor:
        return torch.tensor([iteration < self.maxiter]).repeat(losses.shape[0])

    def reset(self):
        pass


class RelChangeTerminationCriterion(TerminationCriterion):
    # Please be aware that this criterion operates on the batch-loss and not
    # on the individual losses. So this will behave differently with different batch sizes.

    def __init__(self, relchange: float, itercount: int):
        self.relchange = relchange
        self.itercount = itercount
        self.last_losses = torch.zeros(1, itercount, device='cuda')
        self.current_loss_index = -1
        self.current_loss_iteration = -1
        self.running = False
        self.continuing = torch.ones(1, dtype=torch.bool, device='cuda')

    def may_continue(self, iteration: int, losses: torch.Tensor) -> bool:
        # has to be called every iteration!
        if not self.running:
            self.last_losses = self.last_losses.repeat(losses.shape[0], 1)
            self.continuing = self.continuing.repeat(losses.shape[0])
            self.running = True

        # get last loss
        self.current_loss_index += 1
        if self.current_loss_index >= self.itercount:
            self.current_loss_index -= self.itercount

        # before itercount iterations happened
        if self.current_loss_iteration < self.itercount:
            self.current_loss_iteration = iteration
            self.last_losses[:, self.current_loss_index] = losses
            return self.continuing

        # check change
        self.current_loss_iteration = iteration
        neg = (self.last_losses[self.continuing, self.current_loss_index] - losses[self.continuing]) < self.relchange
        self.continuing[self.continuing] &= ~neg

        self.last_losses[:, self.current_loss_index] = losses
        return self.continuing

    def reset(self):
        self.last_losses = torch.zeros(1, self.itercount, device='cuda')
        self.continuing = torch.ones(1, dtype=torch.bool, device='cuda')
        self.current_loss_index = -1
        self.current_loss_iteration = -1
        self.running = False


class AndCombinedTerminationCriterion(TerminationCriterion):

    def __init__(self, criteria: List[TerminationCriterion]):
        self.criteria = criteria

    def may_continue(self, iteration: int, losses: torch.Tensor) -> torch.Tensor:
        result = torch.ones(losses.shape[0], dtype=torch.bool)
        for x in self.criteria:
            result &= x.may_continue(iteration, losses)
        return result

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
