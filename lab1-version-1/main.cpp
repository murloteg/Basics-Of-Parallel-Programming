#include <iostream>
#include <cmath>
#include <mpi.h>

const int SIZE = 4;
const double EPSILON = 0.00001;
const double TAU = 0.01;

double* GenerateVectorOfSolution(int size) {
    auto vector = new double[size];
    for (int i = 0; i < size; ++i) {
        vector[i] = 0;
    }
    return vector;
}

double* GenerateVectorOfRightSide(int size, bool statusOfChangeOfSize, int originSize) {
    auto vector = new double[size];
    for (int i = 0; i < originSize; ++i) {
        vector[i] = size + 1;
    }
    if (statusOfChangeOfSize) {
        for (int i = originSize; i < size; ++i) {
            vector[i] = 0;
        }
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

double* MultiplyMatrixByVector(double* vector, int size, int rank, int numberOfProcesses) {
    double* partOfMatrix = GeneratePartOfMatrix(rank, numberOfProcesses, size);
    int columns = size / numberOfProcesses;
    auto vectorOfResult = new double[columns];
    for (int i = 0; i < columns; ++i) {
        vectorOfResult[i] = 0;
        for (int j = 0; j < size; ++j) {
            vectorOfResult[i] += partOfMatrix[i * size + j] * vector[j];
        }
    }
    delete[] partOfMatrix;
    return vectorOfResult;
}

double* SubtractionOfVectors(double* first, double* second, int size) {
    auto result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] - second[i];
    }
    return result;
}

bool CheckStopCondition(double firstNorm, double secondNorm) {
//    double firstNorm = GetSecondNormOfVector(SubtractionOfVectors(MultiplyMatrixByVector(vectorOfSolution, size, rank, numberOfProcesses), vectorOfRightSide, size), size);
//    double secondNorm = GetSecondNormOfVector(vectorOfRightSide, size);
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

double* MethodOfSimpleIteration(double* vectorOfSolution, double* vectorOfRightSide, int rank, int size, int numberOfProcesses, bool statusOfChangeOfSize, int originSize) {
    auto totalResult = new double[size];
    auto tempVector = new double[size];
    double lowerPartSecondNorm = GetSecondNormOfVector(vectorOfRightSide, SIZE);
    double upperPartSecondNorm = GetSecondNormOfVector(MultiplyVectorByConst(vectorOfRightSide, SIZE, -1), SIZE);
    int columns = SIZE / numberOfProcesses;
    while (!CheckStopCondition(upperPartSecondNorm, lowerPartSecondNorm)) {
        double* partOfResultOfMultiplication = MultiplyMatrixByVector(vectorOfSolution, size, rank, numberOfProcesses);
        MPI_Allgather(partOfResultOfMultiplication, columns, MPI_DOUBLE, tempVector, columns, MPI_DOUBLE, MPI_COMM_WORLD);
//        PrintVector(tempVector, size);
        double* resultOfSubtraction = SubtractionOfVectors(tempVector, vectorOfRightSide, size);
        double* resultOfMultiplicationByConst = MultiplyVectorByConst(resultOfSubtraction, size, TAU);
        totalResult = SubtractionOfVectors(vectorOfSolution, resultOfMultiplicationByConst, size);
        CopyVectorToVector(totalResult, vectorOfSolution, size);
        delete[] partOfResultOfMultiplication;
        delete[] resultOfMultiplicationByConst;
        delete[] resultOfSubtraction;
        upperPartSecondNorm = GetSecondNormOfVector(SubtractionOfVectors(tempVector, vectorOfRightSide, size), size);
    }
    delete[] tempVector;
    CleanUp(vectorOfSolution, vectorOfRightSide);
    return totalResult;
}

bool IsCorrectSize(int size, int numberOfProcesses) {
    return (size % numberOfProcesses) == 0;
}

int FindCorrectSize(int size, int numberOfProcesses) {
    int result = size;
    while (result % numberOfProcesses != 0) {
        ++result;
    }
    return result;
}

int main(int argc, char** argv) {
    int numberOfProcesses = 0;
    int currentRank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    bool statusOfChangeOfSize = false;
    int matrixSize = SIZE;
    if (!IsCorrectSize(SIZE, numberOfProcesses)) {
        matrixSize = FindCorrectSize(SIZE, numberOfProcesses);
        statusOfChangeOfSize = true;
    }
    auto vectorOfSolution = GenerateVectorOfSolution(matrixSize);
    auto vectorOfRightSide = GenerateVectorOfRightSide(matrixSize, statusOfChangeOfSize, SIZE);
    SetFirstApproximation(vectorOfSolution, vectorOfRightSide, matrixSize);
    vectorOfSolution = MethodOfSimpleIteration(vectorOfSolution, vectorOfRightSide, currentRank, matrixSize, numberOfProcesses, statusOfChangeOfSize, SIZE);
    if (currentRank == numberOfProcesses - 1) {
        PrintVector(vectorOfSolution, matrixSize);
    }
    delete[] vectorOfSolution;
    MPI_Finalize();

    return 0;
}
