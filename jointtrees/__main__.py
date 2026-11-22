"""Command line, from the folder that contains jointtrees/ (or anywhere after `pip install`):

    python3 -m jointtrees verify FILES_OR_FOLDERS [--verbose]     re-evaluation and independent checks (verify.py)
    python3 -m jointtrees evaluate FILE                           the full report of a witness

Both take --import FILE.py (or a module name, repeatable): your own filter families or problems, loaded first.
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys

from .verify import verify
from .witness import Witness, witness_files


def load_extension(name):
    """Run FILE.py, or import the module `name`, so that the filter families and problems it defines are known."""
    if not name.endswith('.py'):
        importlib.import_module(name)
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(name)))         # the modules next to it
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(name))[0], name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                     # dataclasses look their module up (string annotations)
    spec.loader.exec_module(module)


def files_of(targets):
    """The witness files of the given files and folders."""
    missing = [t for t in targets if not os.path.exists(t)]
    if missing:
        sys.exit(f'no such file or folder: {", ".join(missing)}')
    files = [f for t in targets for f in witness_files(t)]
    if not files:
        sys.exit(f'no witness files (*.json) in {", ".join(targets)}')
    return files


def verify_files(targets, verbose):
    """Verify every witness file, print the result of each, and exit with 1 if one fails."""
    files = files_of(targets)
    failed = 0
    bounds = []                                         # files with a variable at a search bound
    for path in files:
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            passed, messages = False, [('FAIL', f'not a readable JSON file: {error}')]
        else:
            passed, messages = verify(data)
        if not passed:
            failed += 1
        if any(level == 'bound' for level, _ in messages):
            bounds.append(path)
        print(f"{'PASS' if passed else 'FAIL'}  {path}")
        for level, text in messages:
            if len(files) == 1 or level == 'FAIL' or verbose:
                print(f'      [{level}] {text}')
    print(f'\n{len(files) - failed} of {len(files)} witnesses verified, {len(bounds)} with a variable at a search '
          'bound' + (' (the optimum may lie beyond it, --verbose shows which)' if bounds else ''))
    if bounds and len(bounds) <= 10:
        print('  ' + '\n  '.join(bounds))
    sys.exit(1 if failed else 0)


def evaluate_file(path):
    """Print the full report of a witness."""
    if not os.path.isfile(path):
        sys.exit(f'evaluate: {path} is not a witness file')
    try:
        print(Witness.load(path).report())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        sys.exit(f'evaluate: {path}: {error}')


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--import', dest='imports', action='append', default=[], metavar='FILE.py',
                        help='a file or module with your own filter families or problems, loaded first (repeatable)')
    parser = argparse.ArgumentParser(prog='python3 -m jointtrees', description='Check and show witness files.')
    commands = parser.add_subparsers(dest='command', required=True)
    checks = commands.add_parser('verify', parents=[common], help='re-evaluation and independent checks (verify.py)')
    checks.add_argument('targets', nargs='+', metavar='FILE_OR_FOLDER')
    checks.add_argument('--verbose', action='store_true', help='every check of every file')
    report = commands.add_parser('evaluate', parents=[common], help='the full report of a witness')
    report.add_argument('file')
    args = parser.parse_args(argv)
    for name in args.imports:
        load_extension(name)
    if args.command == 'evaluate':
        evaluate_file(args.file)
    else:
        verify_files(args.targets, args.verbose)


if __name__ == '__main__':
    main()
