import argparse

from ms_agent.cli.app import AppCMD
from ms_agent.cli.run import RunCMD


def run_cmd():
    """This is the entrance of the all the cli commands.

    This cmd imports all other sub commands, for example, `run` and `app`.
    """
    parser = argparse.ArgumentParser(
        'ModelScope-agent Command Line tool',
        usage='ms-agent <command> [<args>]')

    subparsers = parser.add_subparsers(
        help='ModelScope-agent commands helpers')

    RunCMD.define_args(subparsers)
    AppCMD.define_args(subparsers)

    # unknown args will be handled in config.py
    args, _ = parser.parse_known_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()
