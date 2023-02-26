""" Details see below in class definition.
"""
import os
import re

from nuitka import Options
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.PythonVersions import getSystemPrefixPath
from nuitka.utils.FileOperations import listDir
from nuitka.utils.Utils import isMacOS, isWin32Windows

class NuitkaPluginSouddevice(NuitkaPluginBase):
    """This class represents the main logic of the plugin.
    This is a plugin to ensure scripts using numpy, scipy, pandas,
    scikit-learn, etc. work well in standalone mode.
    While there already are relevant entries in the "ImplicitImports.py" plugin,
    this plugin copies any additional binary or data files required by many
    installations.
    """
    plugin_name = "sounddevice"
    plugin_desc = "Required for sounddevice"
    reason = "Dynamic import compilation needed"
    def __init__(self):
        pass

    def getExtraDlls(self, module):
        """Copy extra shared libraries or data for this installation.
        Args:
            module: module object
        Yields:
            DLL entry point objects
        """

        full_name = module.getFullName()
        if full_name == "sounddevice":
            sounddevice_binaries = tuple(
                self._getSounddeviceBinaries(site_packages=module.getCompileTimeDirectory())
            )

            for full_path, target_filename in sounddevice_binaries:
                yield self.makeDllEntryPoint(
                    source_path=full_path,
                    dest_path=target_filename,
                    package_name=full_name,
                    reason=self.reason
                )

            self.reportFileCount(full_name, len(sounddevice_binaries))


    @staticmethod
    def _getSounddeviceBinaries(site_packages):
        """Return any binaries in openvino package.

        Returns:
            tuple of abspaths of binaries.
        """

        # look in openvino/libs for binaries
        libdir = os.path.join(site_packages, "_sounddevice_data", "portaudio-binaries")
        if os.path.isdir(libdir):
            for full_path, filename in listDir(libdir):
                if "libportaudio" in filename:
                    yield full_path, os.path.join("_sounddevice_data", "portaudio-binaries", filename)

