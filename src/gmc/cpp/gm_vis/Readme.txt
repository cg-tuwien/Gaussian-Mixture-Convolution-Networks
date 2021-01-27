In this directory, there needs to be a "build" folder, that should contain the following folders and files from the visualizer-build-directory:
- res
- shaders
- pygmvis[...].pyd
On Windows, you might also need to add Qt5Core.dll, Qt5Gui.dll, Qt5Widgets.dll

On Windows: env-variable QT_PLUGIN_PATH needs to be set to <Qt-Folder>\5.12.6\msvc2017_64\plugins.