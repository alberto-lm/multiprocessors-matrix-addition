#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

#define runs 5

int rowsA = 1024;
int colsA = 1024;
int rowsB = 1024;
int colsB = 1024;
int numThreads = 8;
int ompCorrect = 1;
int cudaCorrect = 1;
int threadsCorrect = 1;
int i;
int j;
int k;
int count;
int x;

char fileA[] = "matrizA.txt";
char fileB[] = "matrizB.txt";
char fileC[] = "matrizC.txt";

double *a, *b, *c, *r, *h_r, start, end;
double executionTimes[runs + 2][3];
double accumTime, temp, precision = 0.0000000001;

int getMatrixesDimensions() {
	printf("Filas Matriz A = ");
	scanf("%d", &rowsA);
	printf("Columnas Matriz A = ");
	scanf("%d", &colsA);
	printf("Filas Matriz B = ");
	scanf("%d", &rowsB);
	printf("Columnas Matriz B = ");
	scanf("%d", &colsB);
	if (rowsA <= 0 || rowsB <= 0 || colsA <= 0 || colsB <= 0 || colsA != rowsB) {
		printf("La operacion no se puede realizar con las dimensiones especificadas.\n");
		return -1;
	}
	return 0;
}

void writeDataIntoFile() {
	FILE* file = fopen(fileC, "w");
	if (file == NULL) {
		printf("El archivo %s no se crea correctamente.", fileC);
	}
	for (int i = 0; i < rowsA * colsB; i++) {
		fprintf(file, "%5.10lf\n", c[i]);
	}
	fclose(file);
}

int readDataFromFile(double* matrixContainer, int expectedSize, char *fileName) {
	FILE* file = fopen(fileName, "r");
	if (file == NULL) {
		printf("El archivo %s no abria correctamente.", fileName);
		return -3;
	}
	int i = 0;
	while (i < expectedSize && fscanf(file, "%lf", &matrixContainer[i]) != EOF){	
		i++;
	}
	printf("\n");
	fclose(file);
	if (i < expectedSize) {
		printf("La cantidad de elementos leidos del archivo %s no permite construir la matriz especificada.", fileName);
		return -3;
	}
	return 0;
}

int buildMatrixes() {
	size_t aSize, bSize, outSize;
	aSize = rowsA * colsA * sizeof(double);
	bSize = rowsB * colsB * sizeof(double);
	outSize = rowsA * colsB * sizeof(double);
	a = (double*)malloc(aSize);
	b = (double*)malloc(bSize);
	c = (double*)malloc(outSize);
	r = (double*)malloc(outSize);
	h_r = (double*)malloc(outSize);
	if (errno == ENOMEM || a == NULL || b == NULL || c == NULL) {
		printf("Error en memory allocation o especio insuficiente en el heap.\n");
		return -2;
	}
	return 0;
}

void freeMemory() {
	free(a);
	free(b);
	free(c);
	free(r);
	free(h_r);
}

int verifyResults(double *outputMatrix) {
	for (i = 0; i < rowsA * colsB; i++) {
		if (abs(c[i] - outputMatrix[i]) > precision) {
			printf("Wroonggg!!\n");
			printf("c = %.10lf\tout = %.10lf\n", c[i], outputMatrix[i]);
			return 0;
		}
	}
	return 1;
}

void serialMul() {
	accumTime = 0;
	for (x = 0; x < runs; x++) {
		start = omp_get_wtime();
		for (i = 0; i < rowsA; i++) {
			for (j = 0; j < colsB; j++) {
				c[i * colsB + j] = 0;
				for (k = 0; k < colsA; k++) {
					c[i * colsB + j] += a[i * colsA + k] * b[k * colsB + j];;
				}
			}
		}
		end = omp_get_wtime();
		accumTime += (end - start);
		executionTimes[x][0] = (end - start) * 1000;
	}
	executionTimes[runs + 1][0] = accumTime / runs  * 1000;
	writeDataIntoFile();
	printf("OMP finished\n");

}

void parallelOMP() {
	accumTime = 0;
	omp_set_num_threads(numThreads);
	for (x = 0; x < runs; x++) {
		#pragma omp parallel shared (a, b, c, r) private (count, i, j, k)
		{
			start = omp_get_wtime();
				#pragma omp for
				for (count = 0; count < rowsA * colsB; count++) {
						i = count / colsB;
						j = count % colsB;
						r[count] = 0;
						for (k = 0; k < colsA; k++) {
							r[count] += a[i * colsA + k] * b[k * colsB + j];;
						}
					
				}
			}
			end = omp_get_wtime();
		  	accumTime += (end - start);
		  	executionTimes[x][1] = (end - start) * 1000;
	}
	executionTimes[runs + 1][1] = accumTime / runs * 1000;
	ompCorrect = verifyResults(r);
	printf("OMP finished\n");

}

__global__ void cudaMul(int *cudaIterations, double * d_a, double * d_b, double * d_r){
	double accum = 0.0;
	int n = *cudaIterations;
	for (int i = 0; i < n; i++){
		//printf("blockIdx = %d\tthreadIdx = %d\ta = %lf\tb = %lf\n", blockIdx.x, threadIdx.x, a[blockIdx.x * n + i], b[i * n + threadIdx.x]);
		accum += d_a[blockIdx.x * n + i] * d_b[i * n + threadIdx.x];
	}
	d_r[blockIdx.x * n + threadIdx.x] = accum;
}

void cudaWrapper(){
	accumTime = 0;
	int *cudaIterations;
	double *d_r, *d_a, *d_b;
	cudaMalloc((int**)& cudaIterations, sizeof(int));
	cudaMemcpy(cudaIterations, &colsA, sizeof(int) , cudaMemcpyHostToDevice);
	cudaMalloc((double**)& d_a, rowsA * colsA * sizeof(double));
	cudaMalloc((double**)& d_b, rowsB * colsB * sizeof(double));
	cudaMalloc((double**)& d_r, rowsA * colsB * sizeof(double));
	cudaMemcpy(d_a, a, rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, rowsB * colsB * sizeof(double), cudaMemcpyHostToDevice);
	for (x = 0; x < runs; x++) {
		start = omp_get_wtime();
		cudaMul <<< rowsA, colsB >>> (cudaIterations, d_a, d_b, d_r);
		cudaDeviceSynchronize();
		end = omp_get_wtime();
		accumTime += (end - start);
		executionTimes[x][2] = (end - start) * 1000;
	}
	cudaMemcpy(h_r, d_r, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);
	executionTimes[runs + 1][2] = accumTime / runs * 1000;
	cudaCorrect = verifyResults(h_r);
	cudaFree(cudaIterations);
	cudaFree(d_r);
	cudaFree(d_a);
	cudaFree(d_b);
}

void printTable() {
	printf("%-20s%-20s%-20s%-20s\n","Corrida", "Serial", "Paralelo1", "Paralelo2");
	for (i = 0; i < runs; i++) {
		printf ("%-20d%-20.10lf%-20.10lf%-20.10lf\n", i+1, executionTimes[i][0], executionTimes[i][1], executionTimes[i][2]);
	}
	printf ("%-20s%-20.10lf%-20.10lf%-20.10lf\n", "Promedio", executionTimes[runs + 1][0], executionTimes[runs + 1][1], executionTimes[runs + 1][2]);
	printf ("%-20s%-20s%-20.10lf%-20.10lf\n", "Speedup", "-",  executionTimes[runs + 1][1] / executionTimes[runs + 1][0], executionTimes[runs + 1][2] / executionTimes[runs + 1][0]);
	printf ("%-20s%-20s%-20d%-20d\n", "Correct", "-", ompCorrect, cudaCorrect);
}

int main() {
	/*if (getMatrixesDimensions()) {
		return -1;
	}*/
	if (buildMatrixes()) {
		return -2;
	}
	if (readDataFromFile(a, rowsA * colsA, fileA) || readDataFromFile(b, rowsB * colsB, fileB)) {
		freeMemory();
		return -3;
	}
	serialMul();
	parallelOMP();
	cudaWrapper();
	printTable();
	freeMemory();
	return 0;
}
