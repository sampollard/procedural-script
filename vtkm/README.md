# Checking whether inside or outside a curve using VTK-m

This little worklet takes in a file of control points (a sample is shown in cs.txt) and returns for each point whether it is within or without of the curve, according to some user-specified functor (default width is 0.08).

In order for this to work you must install VTK-m somewhere on your computer. It can be downloaded from the
(https://gitlab.kitware.com/vtk/vtk-m)[VTK-m repository] and be built using `ccmake`.

## To Build
* change the line `set(VTKm_DIR "/Users/spollard/Documents/uo/cis607vtkm/vtk-m/build/lib")` in CMakeLists.txt as appropriate.
* cmake .
* ./bezier

