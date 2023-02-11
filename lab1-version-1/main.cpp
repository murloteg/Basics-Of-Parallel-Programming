#include <iostream>
#include <cmath>
#include <mpi.h>

const int SIZE = 10;
const double EPSILON = 0.00001;
const double TAU = 0.01;

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
    int matrixSize = 0;
    if (size % numberOfProcesses != 0) {
        // ... TODO: do it later.
    }
    else {
        matrixSize = size * (size / numberOfProcesses);
    }
    auto matrix = new double[matrixSize];
    for (int i = 0; i < matrixSize; ++i) {
        matrix[i] = 1;
    }
    int rows = (size / numberOfProcesses);
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

void SetFirstApproximation(double* vectorOfSolution, const double* vectorOfRightSide, int size) {
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

double* MultiplyMatrixByVector(double* vector, int size, int numberOfProcesses) {
    auto vectorOfResult = new double[size];
    for (int rank = 0; rank < numberOfProcesses; ++rank) {
        double* partOfMatrix = GeneratePartOfMatrix(rank, numberOfProcesses, size);
        int columns = size / numberOfProcesses;
        for (int i = 0; i < columns; ++i) {
            vectorOfResult[i + rank * columns] = 0;
            for (int j = 0; j < size; ++j) {
                vectorOfResult[i + rank * columns] += partOfMatrix[i * size + j] * vector[j];
            }
        }
        delete[] partOfMatrix;
    }
    return vectorOfResult;
}

double* SubtractionOfVectors(double* first, double* second, int size) {
    auto result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] - second[i];
    }
    return result;
}

bool CheckStopCondition(double* vectorOfSolution, double* vectorOfRightSide, int size, int numberOfProcesses) {
    double firstNorm = GetSecondNormOfVector(SubtractionOfVectors(MultiplyMatrixByVector(vectorOfSolution, size, numberOfProcesses), vectorOfRightSide, size), size);
    double secondNorm = GetSecondNormOfVector(vectorOfRightSide, size);
    return (firstNorm / secondNorm) < EPSILON;
}

double* MultiplyVectorByConst(double* first, int size, double constValue) {
    auto result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] * constValue;
    }
    return result;
}

void CopyVectorToVector(double* source, double* destination, int size) {
    for (int i = 0; i < size; ++i) {
        destination[i] = source[i];
    }
}

void PrintVector(double* vector, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void CleanUp(double* vectorOfSolution, double* vectorOfRightSide) {
    delete[] vectorOfSolution;
    delete[] vectorOfRightSide;
}

double* MethodOfSimpleIteration(double* vectorOfSolution, double* vectorOfRightSide, int size, int numberOfProcesses) {
    auto totalResult = new double[size];
    while (!CheckStopCondition(vectorOfSolution, vectorOfRightSide, size, numberOfProcesses)) {
        double* resultOfMultiplicationByMatrix = MultiplyMatrixByVector(vectorOfSolution, size, numberOfProcesses);
        double* resultOfSubtraction = SubtractionOfVectors(resultOfMultiplicationByMatrix, vectorOfRightSide, size);
        double* resultOfMultiplicationByConst = MultiplyVectorByConst(resultOfSubtraction, size, TAU);
        totalResult = SubtractionOfVectors(vectorOfSolution, resultOfMultiplicationByConst, size);
        CopyVectorToVector(totalResult, vectorOfSolution, size);
        delete[] resultOfMultiplicationByMatrix;
        delete[] resultOfMultiplicationByConst;
        delete[] resultOfSubtraction;
    }
    CleanUp(vectorOfSolution, vectorOfRightSide);
    return totalResult;
}

int main(int argc, char** argv) {
    auto vectorOfSolution = GenerateVectorOfSolution(SIZE);
    auto vectorOfRightSide = GenerateVectorOfRightSide(SIZE);
    SetFirstApproximation(vectorOfSolution, vectorOfRightSide, SIZE);

    int numberOfProcesses = 0;
    int currentRank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    vectorOfSolution = MethodOfSimpleIteration(vectorOfSolution, vectorOfRightSide, SIZE, numberOfProcesses);
    if (currentRank == numberOfProcesses - 1) {
        PrintVector(vectorOfSolution, SIZE);
    }
    MPI_Finalize();

    return 0;
}
