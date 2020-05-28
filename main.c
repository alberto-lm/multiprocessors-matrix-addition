#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

#define runs 5

int rowsA = 50;
int colsA = 50;
int rowsB = 50;
int colsB = 50;
int numThreads = 8;
int ompCorrect = 1;
int threadsCorrect = 1;
int i;
int j;
int k;
int count;
int x;

char fileA[] = "matrizA.txt";
char fileB[] = "matrizB.txt";
char fileC[] = "matrizC.txt";

double *a, *b, *c, *r, start, end;
double executionTimes[runs + 2][3];
double accumTime;

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
		printf("El archivo %s no se creó correctamente.", fileC);
	}
	for (int i = 0; i < rowsA * colsB; i++) {
		fprintf(file, "%5.10lf\n", c[i]);
	}
	fclose(file);
}

int readDataFromFile(double* matrixContainer, int expectedSize, char *fileName) {
	FILE* file = fopen(fileName, "r");
	if (file == NULL) {
		printf("El archivo %s no abrió correctamente.", fileName);
		return -3;
	}
	int i = 0;
	while (i < expectedSize && fscanf(file, "%lf", &matrixContainer[i]) != EOF){	
		i++;
	}
	printf("\n");
	fclose(file);
	if (i < expectedSize) {
		printf("La cantidad de elementos leídos del archivo %s no permite construir la matriz especificada.", fileName);
		return -3;
	}
	return 0;
}

int buildMatrixes() {
	size_t alignment = 32;
	size_t aSize, bSize, outSize;
	aSize = rowsA * colsA * sizeof(double);
	bSize = rowsB * colsB * sizeof(double);
	outSize = rowsA * colsB * sizeof(double);
	a = (double*)_aligned_malloc(aSize, alignment);
	b = (double*)_aligned_malloc(bSize, alignment);
	c = (double*)_aligned_malloc(outSize, alignment);
	r = (double*)_aligned_malloc(outSize, alignment);
	if (errno == ENOMEM || a == NULL || b == NULL || c == NULL) {
		printf("Error en memory allocation o especio insuficiente en el heap.\n");
		return -2;
	}
	return 0;
}

void freeMemory() {
	_aligned_free(a);
	_aligned_free(b);
	_aligned_free(c);
	_aligned_free(r);
}

int verifyResults() {
	for (i = 0; i < rowsA * colsB; i++) {
		if (c[i] != r[i]) {
			printf("Wronnnngg\n");
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
					c[i * colsB + j] += /*c[i * colsB + j] +*/ a[i * colsA + k] * b[k * colsB + j];
				}
			}
		}
		end = omp_get_wtime();
		accumTime += (end - start);
		executionTimes[x][0] = (end - start);
	}
	executionTimes[runs + 1][0] = accumTime / runs;
	writeDataIntoFile();
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
							r[count] += a[i * colsA + k] * b[k * colsB + j];
						}
					
				}
			}
			end = omp_get_wtime();
		  accumTime += (end - start);
		  executionTimes[x][1] = (end - start);
	}
	executionTimes[runs + 1][1] = accumTime / runs;
	ompCorrect = verifyResults();
}

void printTable() {
	printf("%-15s\t%-15s\t%-15s\n","Corrida", "Serial", "Paralelo");
	for (i = 0; i < runs; i++) {
		printf ("%-15d\t%5.10lf\t%5.10lf\n", i+1, executionTimes[i][0], executionTimes[i][1]);
	}
	printf ("%-15s\t%5.10lf\t%5.10lf\n", "Promedio", executionTimes[runs + 1][0], executionTimes[runs + 1][1]);
	printf ("%-15s\t%-15s\t%5.10lf\n", "Speedup", "-", executionTimes[runs + 1][0] / executionTimes[runs + 1][1]);

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
	printTable();
	freeMemory();
	return 0;
}
