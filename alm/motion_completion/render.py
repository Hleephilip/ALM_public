import sys

from .cli import run


def main() -> None:
    run("render", sys.argv[1:])


if __name__ == "__main__":
    main()
