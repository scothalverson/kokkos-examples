CXX = g++
CXXFLAGS = -I/home/scot/software/spack/opt/spack/linux-ubuntu18.10-x86_64/gcc-8.2.0/kokkos-2.7.00-aeuk5hmtz3peiknngmxvo4dpntb6574l/include/
CXXLDFLAGS = -L/home/scot/software/spack/opt/spack/linux-ubuntu18.10-x86_64/gcc-8.2.0/kokkos-2.7.00-aeuk5hmtz3peiknngmxvo4dpntb6574l/lib/ -lkokkos -ldl


all:
	$(CXX) -O3 $(CXXFLAGS) ../src/mandelbrot.cpp -o mandelbrot.x $(CXXLDFLAGS)

clean:
	rm *.o *.x
