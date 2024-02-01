# See https://pytorch.org/tutorials/advanced/cpp_frontend.html
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(benchmark)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)   # needed by export-method=tracing/scripting

add_executable(benchmark benchmark.cpp)
target_link_libraries(
  torchscript_mask_rcnn
  -Wl,--no-as-needed TorchVision::TorchVision -Wl,--as-needed
  "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET benchmark PROPERTY CXX_STANDARD 14)