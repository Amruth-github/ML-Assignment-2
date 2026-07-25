from __future__ import annotations


def build_models() -> dict:
    raise NotImplementedError


def evaluate(model, X_test, y_test) -> dict:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
