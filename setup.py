"""
@author: phdenzel
Usage:
    python setup.py py2app
"""
import os
from setuptools import setup

APP = ['modelzapper.py']
PLIST = dict(CFBundleName="GLEAM",
             CFBundleDisplayName="GLEAM",
             CFBundleGetInfoString="Analyze photometric gravitational lens data" \
             + " using the GLEAM",
             CFBundleIdentifier="org.pythonmac.gleam",
             author_email="phdenzel@gmail.com",
             CFBundleVersion="0.1.0",
             CFBundleShortVersionString="0.1.0",
             NSHumanReadableCopyright=u"Copyright \u00A9 2018, Philipp Denzel," \
             + " All Rights Reserved",
             LSBackgroundOnly=False,
)
DATAFILES = [('', ['']),
             ('', ['lib']),
             ('', ['include']),
]
PACKAGES = [
    'numpy',
    'matplotlib',
    'scipy',
    'PIL',
            
]
OPTIONS = {'iconfile': "imgs/gleam.icns",
           'plist': PLIST,
           'packages': PACKAGES,
}
setup(
    app=APP,
    data_files=DATAFILES,
    options={'py2app': OPTIONS},
    setup_requires=["py2app"],
)
