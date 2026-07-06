"""Short project entrypoint.

Main command:

python auto.py pull
"""

from __future__ import annotations

import argparse
import sys

from scripts.autonomous_loop_runner import build_arg_parser as build_loop_arg_parser
from scripts.autonomous_loop_runner import run_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    pull = subparsers.add_parser("pull", help="Run the PullCube xarm6 autonomous loop.")
    loop_parser = build_loop_arg_parser()
    for action in loop_parser._actions:
        if not action.option_strings:
            continue
        if isinstance(action, argparse._HelpAction):
            continue
        kwargs = {
            "default": action.default,
            "help": action.help,
        }
        if getattr(action, "type", None) is not None:
            kwargs["type"] = action.type
        if getattr(action, "choices", None) is not None:
            kwargs["choices"] = action.choices
        if action.const is not None:
            kwargs["const"] = action.const
        if action.nargs is not None:
            kwargs["nargs"] = action.nargs
        if isinstance(action, argparse._StoreTrueAction):
            pull.add_argument(*action.option_strings, action="store_true", default=action.default, help=action.help)
        elif isinstance(action, argparse._StoreFalseAction):
            pull.add_argument(*action.option_strings, action="store_false", default=action.default, help=action.help)
        else:
            pull.add_argument(*action.option_strings, **kwargs)

    args = parser.parse_args()
    if args.command == "pull":
        run_loop(args)
        return
    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
