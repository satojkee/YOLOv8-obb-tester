import os
import json
import click
from munch import DefaultMunch

from src import *


@click.command()
@click.option(
    '-s',
    '--setup',
    'setup_file',
    default=DEFAULT_SETUP,
    help=f'Where is setup file located? [default: {DEFAULT_SETUP}]'
)
def main(setup_file: str) -> None:
    """YOLOv8-obb tester."""
    if os.path.exists(setup_file):
        with open(setup_file, 'r') as handler:
            setup = DefaultMunch.fromDict(json.load(handler))

            TestUtil(setup).run()
    else:
        click.echo('"{}" not found!'.format(setup_file), err=True)


if __name__ == '__main__':
    main()
