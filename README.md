# monte-carlo-multi-system

Simple Monte carlo simulation API running on CPU/GPU using Thrust library.

#### CPU

  - Standard threaed implementation (CPP)
  - OpenMP implementation
  - TBB implementation

### GPU
  - CUDA implementation

The current implementation uses the execution policy mechanism of the Thrust library to run the simulation on the wanted
implementation. All CPU implementations have own methods since currently Thrust doesn't seem to support dynamic execution policy choosing. This leads to more boilerplate code.

## Building

Since this uses CUDA, its advised to import this in CUDA projects capable IDE.
I used Eclipse C++ IDE with the Nsight Eclipse Plugin Edition.

https://docs.nvidia.com/cuda/nsight-eclipse-plugins-guide/index.html

##### Dependencies

  - cuda
  - thrust
  - gomp
  - tbb
  - pthread
  - pistache
  - jsoncpp
