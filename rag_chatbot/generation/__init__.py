"""Grounded answer generation package."""

__all__ = ["GroundedGenerator", "answer_question"]


def __getattr__(name: str):
    """Lazily expose generation helpers without heavy import-time dependencies."""

    if name == "GroundedGenerator":
        from .service import GroundedGenerator

        return GroundedGenerator
    if name == "answer_question":
        from .pipeline import answer_question

        return answer_question
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
