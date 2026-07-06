import sys

from .cli import run


def main() -> None:
    run("edit", sys.argv[1:])


if __name__ == "__main__":
    main()
