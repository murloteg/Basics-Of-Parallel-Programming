#include <iostream>
#include <cmath>
#include <mpi.h>
#define EPSILON = 0.00001;
#define TAU = 0.01;

double* GenerateVectorOfSolution(int size) {
    auto vector = new double[size];
    for (int i = 0; i < size; ++i) {
        vector[i] = 0;
    }
    return vector;
}

double* GenerateVectorOfRightSide(int size) {
    auto vector = new double[size];
    for (int i = 0; i < size; ++i) {
        vector[i] = size + 1;
    }
    return vector;
}

double* GeneratePartOfMatrix(int rank, int numberOfProcesses, int size) {
    int matrixSize = size * (size / numberOfProcesses);
    auto matrix = new double[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        matrix[i] = 1;
    }
    int rows = (size / numberOfProcesses) + 1 * (size % numberOfProcesses != 0);
    int startPosition = rows * rank;
    for (int i = 0; i < rows; ++i) {
        if (startPosition > matrixSize) {
            break;
        }
        matrix[startPosition] = 2;
        startPosition = startPosition + size + 1;
    }
    return matrix;
}

void CalculateFirstApproximation(double* vectorOfSolution, const double* vectorOfRightSide, int size) {
    for (int i = 0; i < size; ++i) {
        vectorOfSolution[i] = vectorOfRightSide[i];
    }
}

double GetSecondNormOfVector(const double* vector, int size) {
    double totalSum = 0;
    for (int i = 0; i < size; ++i) {
        totalSum += vector[i] * vector[i];
    }
    return sqrt(totalSum);
}

void DebugPrint(double* matrix, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << matrix[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
}

double* MultipleMatrixByVector(double* vector, int size, int numberOfProcesses) {
    auto vectorOfResult = new double[size];
    for (int rank = 0; rank < numberOfProcesses; ++rank) {
        double* partOfMatrix = GeneratePartOfMatrix(rank, numberOfProcesses, size);
        DebugPrint(partOfMatrix, size/numberOfProcesses, size); // TODO: remove later.
        int columns = size / numberOfProcesses;
        for (int i = 0; i < columns; ++i) {
            vectorOfResult[i + rank * columns] = 0;
            for (int j = 0; j < size; ++j) {
                vectorOfResult[i + rank * columns] += partOfMatrix[i * size + j] * vector[j];
            }
        }
    }
    return vectorOfResult;
}

bool CheckStopCondition() {

}

void MethodOfSimpleIteration(const double* partOfMatrix, double* vectorOfSolution, const double* vectorOfRightSide, int size) {

}

void SimpleMultipleMatrixByVector(double* vector, double* matrix, int size) {
    auto vectorOfResult = new double[size];
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < size; ++k) {
            vectorOfResult[k] += matrix[k * size + j] * vector[j];
        }
    }
    for (int i = 0; i < size; ++i) {
        std::cout << vectorOfResult[i] << std::endl;
    }
}

using namespace std;
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Incorrect input: matrix size" << endl;
        return 0;
    }

    int matrixSize = strtol(argv[1], nullptr, 10);
    int numberOfProcesses = 0;
//    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    auto vector = new double[3] {0, 5, 10};
    auto matrix = new double[9] {1, 2, 3, 1, 2, 3, 0, 2, 0};
//    SimpleMultipleMatrixByVector(vector, matrix, 3);
    auto res = MultipleMatrixByVector(GenerateVectorOfRightSide(4), 4, 4);
    for (int i = 0; i < 4; ++i) {
        cout << res[i] << " ";
    }

    return 0;
}
