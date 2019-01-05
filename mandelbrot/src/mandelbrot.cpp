#include <complex>
#include <iostream>
#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"

struct cplxCount {
	std::complex<float> z,c;
	size_t i;
} ;

typedef Kokkos::DualView<cplxCount**> view_type;
void printMandelbrot(view_type data, size_t rows, size_t columns, size_t max_iteration){
	for(int x = 0; x < rows; x++){
		for(int y = 0; y < columns; y++){
			std::cout << (data.h_view(x,y).i == max_iteration ? '*' : ' ');
		}
		std::cout << '\n';
	}
}

void doMandelbrot(size_t max_row, size_t max_column, size_t max_iteration){
	view_type data("data", max_row, max_column);
	data.modify<view_type::host_mirror_space>();
	Kokkos::parallel_for(max_row*max_column, KOKKOS_LAMBDA(int cell) {
			size_t row = cell / max_column;
			size_t column =  cell - (row * max_column);
			data.d_view(row,column).z = {
				(float)column * 2 / max_column - 1.5f,
				(float)row * 2 / max_row - 1				
			};
			data.d_view(row,column).c = data.d_view(row,column).z;
			data.d_view(row,column).i = 0;
			while(abs(data.d_view(row,column).z) < 2 && ++data.d_view(row,column).i < max_iteration){
				data.d_view(row,column).z = pow(data.d_view(row,column).z, 2) + data.d_view(row,column).c;
			}
	});
	data.sync<view_type::host_mirror_space>();
	printMandelbrot(data, max_row, max_column, max_iteration);

}

int main(){
	Kokkos::initialize();
	doMandelbrot(80,190,100);
	Kokkos::finalize();
}
