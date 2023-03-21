import argparse
import subprocess
import sys

nuitka_args = [
    '--standalone',
    '--assume-yes-for-downloads',
    '--plugin-enable=numpy',
    #'--enable-plugin=pyqt5',
    #'--noinclude-numba-mode',
    '--include-data-dir=models=models',
    '--include-data-dir=JSON Images Filestore=JSON Images Filestore',
    '--include-data-dir=PNG Images Filestore=PNG Images Filestore',
    '--include-data-dir=Story Library=Story Library',
    '--include-data-files=configuration.json=configuration.json',
    '--output-dir=Release',
    '--user-plugin=nuitka_plugins/MediapipePlugin.py',
    '--user-plugin=nuitka_plugins/OpenvinoPlugin.py',
    '--user-plugin=nuitka_plugins/SounddevicePlugin.py'
]


def build():
    parser = argparse.ArgumentParser(description='Build motion input with Nuitka.')
    parser.add_argument('--lto', action='store_true',
                        help='Use link time optimisation. Default: False')
    parser.add_argument('--console', action='store_true',
                        help='Show console when running the built executable. Default: False')
    parser.add_argument('target_file', type=str,
                        help='Target file to build.')
    args = parser.parse_args()
    print(args)
    if args.lto:
        nuitka_args.append('--lto=yes')
    else:
        nuitka_args.append('--lto=no')
    #if not args.console:
    #    nuitka_args.append('--windows-disable-console')
    subprocess.call([sys.executable, "-m", "nuitka"] + nuitka_args + [args.target_file])

if __name__ == "__main__":
    build()
    