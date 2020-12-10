"""Base Metric class to be derived from."""

from functools import total_ordering


@total_ordering
class Metric:
    """A base class that will be inherited by evaluation metrics.

    Args:
        name: A name for the metric that will be kept internally after upper-casing
        score: A floating point score
        detailed_score: A custom, more detailed string
            representing the score given above
        higher_better: If `False`, the smaller the better
    """
    def __init__(self, name: str, score: float,
                 detailed_score: str = "", higher_better: bool = True):
        self.name = name.upper()
        self.score = score
        self.detailed_score = detailed_score
        self.higher_better = higher_better

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        rhs = (self.detailed_score if self.detailed_score
               else "%.2f" % self.score)
        return self.name + ' = ' + rhs
