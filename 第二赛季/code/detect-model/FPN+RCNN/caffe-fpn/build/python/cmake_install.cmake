# Install script for directory: /workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/classify.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/detect.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/draw_net.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/requirements.txt"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/pycaffe.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/detector.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/classifier.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/__init__.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/draw.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/net_spec.py"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/io.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         RPATH "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/install/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda/lib64:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/lib/_caffe.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         OLD_RPATH "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda/lib64:/usr/local/lib::::::::"
         NEW_RPATH "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/build/install/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/usr/local/cuda/lib64:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/imagenet"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/proto"
    "/workspace/mnt/group/ocr/xieyufei/tianchi/season2/code/FPN/caffe-fpn/python/caffe/test"
    )
endif()

