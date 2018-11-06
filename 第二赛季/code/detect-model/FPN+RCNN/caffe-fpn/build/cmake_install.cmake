# Install script for directory: /workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Caffe" TYPE FILE FILES "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/cmake/CaffeConfig.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Caffe/CaffeTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Caffe/CaffeTargets.cmake"
         "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/CMakeFiles/Export/share/Caffe/CaffeTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Caffe/CaffeTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Caffe/CaffeTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Caffe" TYPE FILE FILES "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/CMakeFiles/Export/share/Caffe/CaffeTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Caffe" TYPE FILE FILES "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/CMakeFiles/Export/share/Caffe/CaffeTargets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/src/gtest/cmake_install.cmake")
  include("/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/src/caffe/cmake_install.cmake")
  include("/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/tools/cmake_install.cmake")
  include("/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/examples/cmake_install.cmake")
  include("/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/python/cmake_install.cmake")
  include("/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/matlab/cmake_install.cmake")
  include("/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/docs/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
