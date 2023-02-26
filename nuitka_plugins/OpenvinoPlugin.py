""" Details see below in class definition.
"""
import os
import re

from nuitka import Options
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.PythonVersions import getSystemPrefixPath
from nuitka.utils.FileOperations import listDir
from nuitka.utils.Utils import isMacOS, isWin32Windows

class NuitkaPluginOpenvino(NuitkaPluginBase):
    """This class represents the main logic of the plugin.
    This is a plugin to ensure scripts using numpy, scipy, pandas,
    scikit-learn, etc. work well in standalone mode.
    While there already are relevant entries in the "ImplicitImports.py" plugin,
    this plugin copies any additional binary or data files required by many
    installations.
    """
    plugin_name = "openvino"
    plugin_desc = "Required for openvino"
    reason = "Dynamic import compilation needed"
    def __init__(self):
        pass

    def getImplicitImports(self, module):
        full_name = module.getFullName()
        if full_name == "openvino":
            yield "openvino.inference_engine.constants"

    def getExtraDlls(self, module):
        """Copy extra shared libraries or data for this installation.
        Args:
            module: module object
        Yields:
            DLL entry point objects
        """

        full_name = module.getFullName()
        if full_name == "openvino":
            openvino_binaries = tuple(
                self._getOpenvinoBinaries(openvino_dir=module.getCompileTimeDirectory())
            )

            for full_path, target_filename in openvino_binaries:
                yield self.makeDllEntryPoint(
                    source_path=full_path,
                    dest_path=target_filename,
                    package_name=full_name,
                    reason=self.reason
                )

            self.reportFileCount(full_name, len(openvino_binaries))


    @staticmethod
    def _getOpenvinoBinaries(openvino_dir):
        """Return any binaries in openvino package.

        Returns:
            tuple of abspaths of binaries.
        """

        # look in openvino/libs for binaries
        libdir = os.path.join(openvino_dir, "libs" if not isMacOS() else "dylibs")
        if os.path.isdir(libdir):
            for full_path, filename in listDir(libdir):
                yield full_path, filename

