import sys

from .cli import run


def main() -> None:
    run("complete", sys.argv[1:])


if __name__ == "__main__":
    main()
