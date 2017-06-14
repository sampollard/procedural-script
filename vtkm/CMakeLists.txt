##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2015 Sandia Corporation.
##  Copyright 2015 UT-Battelle, LLC.
##  Copyright 2015 Los Alamos National Security.
##
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
##  Laboratory (LANL), the U.S. Government retains certain rights in
##  this software.
##
##=============================================================================

cmake_minimum_required(VERSION 3.6)

set(VTKm_DIR "/Users/spollard/Documents/uo/cis607vtkm/vtk-m/build/lib")

#Find the VTK-m package
find_package(VTKm REQUIRED QUIET
  OPTIONAL_COMPONENTS Serial CUDA TBB)

find_package(Boost REQUIRED)

add_executable(bezier bezier.cxx)

target_include_directories(bezier PRIVATE ${VTKm_INCLUDE_DIRS})
target_link_libraries(bezier PRIVATE ${VTKm_LIBRARIES})
target_compile_options(bezier PRIVATE ${VTKm_COMPILE_OPTIONS})

