from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="uic.dsprov",
    package_names=['uic', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    package_data={'': ['*.conf', '*.json', '*.yml']},
    description='Contains classes for creating and accessing discharge summary *provenience of data* annotations.',
    user='plandes',
    project='dsprov',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
