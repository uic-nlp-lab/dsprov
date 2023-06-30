#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness, ProgramNameConfigurator

if (__name__ == '__main__'):
    cctx = ProgramNameConfigurator(None, default='dsprov').create_section()
    harness = ConfigurationImporterCliHarness(
        src_dir_name='src/python',
        package_resource='uic.dsprov',
        app_config_context=cctx,
        proto_args='proto',
        proto_factory_kwargs={'reload_pattern': r'^zensols\.dsprov\.(?!domain)'},
    )
    harness.run()
