# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki

# Include any dependencies generated for this target.
include CMakeFiles/Sinteticki.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Sinteticki.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Sinteticki.dir/flags.make

CMakeFiles/Sinteticki.dir/sinteticki.cpp.o: CMakeFiles/Sinteticki.dir/flags.make
CMakeFiles/Sinteticki.dir/sinteticki.cpp.o: sinteticki.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/Grimnir/AugmentedRealityPetnica2017/Sinteticki/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Sinteticki.dir/sinteticki.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Sinteticki.dir/sinteticki.cpp.o -c /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki/sinteticki.cpp

CMakeFiles/Sinteticki.dir/sinteticki.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Sinteticki.dir/sinteticki.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki/sinteticki.cpp > CMakeFiles/Sinteticki.dir/sinteticki.cpp.i

CMakeFiles/Sinteticki.dir/sinteticki.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Sinteticki.dir/sinteticki.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki/sinteticki.cpp -o CMakeFiles/Sinteticki.dir/sinteticki.cpp.s

CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.requires:

.PHONY : CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.requires

CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.provides: CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.requires
	$(MAKE) -f CMakeFiles/Sinteticki.dir/build.make CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.provides.build
.PHONY : CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.provides

CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.provides.build: CMakeFiles/Sinteticki.dir/sinteticki.cpp.o


# Object files for target Sinteticki
Sinteticki_OBJECTS = \
"CMakeFiles/Sinteticki.dir/sinteticki.cpp.o"

# External object files for target Sinteticki
Sinteticki_EXTERNAL_OBJECTS =

Sinteticki: CMakeFiles/Sinteticki.dir/sinteticki.cpp.o
Sinteticki: CMakeFiles/Sinteticki.dir/build.make
Sinteticki: /usr/local/lib/libopencv_videostab.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_ts.a
Sinteticki: /usr/local/lib/libopencv_superres.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_stitching.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_contrib.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_nonfree.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_ocl.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_gpu.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_photo.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_objdetect.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_legacy.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_video.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_ml.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_calib3d.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_features2d.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_highgui.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_imgproc.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_flann.so.2.4.13
Sinteticki: /usr/local/lib/libopencv_core.so.2.4.13
Sinteticki: CMakeFiles/Sinteticki.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/Grimnir/AugmentedRealityPetnica2017/Sinteticki/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Sinteticki"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Sinteticki.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Sinteticki.dir/build: Sinteticki

.PHONY : CMakeFiles/Sinteticki.dir/build

CMakeFiles/Sinteticki.dir/requires: CMakeFiles/Sinteticki.dir/sinteticki.cpp.o.requires

.PHONY : CMakeFiles/Sinteticki.dir/requires

CMakeFiles/Sinteticki.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Sinteticki.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Sinteticki.dir/clean

CMakeFiles/Sinteticki.dir/depend:
	cd /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki /home/Grimnir/AugmentedRealityPetnica2017/Sinteticki/CMakeFiles/Sinteticki.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Sinteticki.dir/depend

