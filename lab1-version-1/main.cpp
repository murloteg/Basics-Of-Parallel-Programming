#include <iostream>
#include <cmath>
#include <mpi.h>

const int ORIGIN_SIZE = 3000;
const double EPSILON = 1e-5;
const double TAU = 1e-5;

double* GenerateVectorOfSolution(int size) {
    auto vector = new double[size];
    for (int i = 0; i < size; ++i) {
        vector[i] = 0;
    }
    return vector;
}

double* GenerateVectorOfRightSide(int size) {
    auto vector = new double[size];
    for (int i = 0; i < ORIGIN_SIZE; ++i) {
        vector[i] = ORIGIN_SIZE + 1;
    }
    if (size != ORIGIN_SIZE) {
        for (int i = ORIGIN_SIZE; i < size; ++i) {
            vector[i] = 0;
        }
    }
    return vector;
}

void PrintVector(double* vector, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

int GetNumberOfExtraRows(int size) {
    return (size - ORIGIN_SIZE);
}

int GetLastElementFromArray(int* array, int size) {
    return array[size - 1];
}

double* GeneratePartOfMatrix(int rank, int numberOfProcesses, int size) {
    int matrixSize = size * (size / numberOfProcesses);
    int numberOfRows = (size / numberOfProcesses);
    auto matrix = new double[matrixSize];
    for (int i = 0; i < numberOfRows; ++i) {
        for (int j = 0; j < ORIGIN_SIZE; ++j) {
            matrix[i * size + j] = 1;
        }
    }
    int startPosition = numberOfRows * rank;
    for (int i = 0; i < numberOfRows; ++i) {
        if (startPosition > matrixSize) {
            break;
        }
        matrix[startPosition] = 2;
        startPosition = startPosition + size + 1;
    }
    auto arrayWithRowNumbers = new int[numberOfRows];
    for (int i = 0; i < numberOfRows; ++i) {
        arrayWithRowNumbers[i] = i + rank * numberOfRows + 1;
    }

    int numberOfExtraRows = GetNumberOfExtraRows(size);
    int extraRowPosition = size - numberOfExtraRows;
    if (GetLastElementFromArray(arrayWithRowNumbers, numberOfRows) > extraRowPosition) {
        for (int i = 0; i < numberOfRows; ++i) {
            if (arrayWithRowNumbers[i] > extraRowPosition) {
                for (int j = 0; j < size; ++j) {
                    matrix[i * size + j] = 0;
                }
            }
        }
    }
    delete[] arrayWithRowNumbers;
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

void MultiplyMatrixByVector(double* vectorOfResult, const double* vector, int size, int rank, int numberOfProcesses) {
    double* partOfMatrix = GeneratePartOfMatrix(rank, numberOfProcesses, size);
    int numberOfRows = size / numberOfProcesses;
    for (int i = 0; i < numberOfRows; ++i) {
        vectorOfResult[i] = 0;
        for (int j = 0; j < size; ++j) {
            vectorOfResult[i] += partOfMatrix[i * size + j] * vector[j];
        }
    }
    delete[] partOfMatrix;
}

void SubtractionOfVectors(double* result, const double* first, const double* second, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] - second[i];
    }
}

bool CheckStopCondition(double firstNorm, double secondNorm) {
    return (firstNorm / secondNorm) < EPSILON;
}

void MultiplyVectorByConst(double* result, const double* first, int size, double constValue) {
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] * constValue;
    }
}

void CopyVectorToVector(const double* source, double* destination, int size) {
    for (int i = 0; i < size; ++i) {
        destination[i] = source[i];
    }
}

void CleanUp(const double* vectorOfSolution, const double* vectorOfRightSide, const double* storageVector,
             const double* resultOfMultiplicationByConst, const double* partOfMultiplication,
             const double* resultOfSubtraction) {
    delete[] vectorOfSolution;
    delete[] vectorOfRightSide;
    delete[] storageVector;
    delete[] resultOfMultiplicationByConst;
    delete[] partOfMultiplication;
    delete[] resultOfSubtraction;
}

double* MethodOfSimpleIteration(double* vectorOfSolution, int rank, int size, int numberOfProcesses) {
    auto vectorOfRightSide = GenerateVectorOfRightSide(size);
    SetFirstApproximation(vectorOfSolution, vectorOfRightSide, size);

    int numberOfRows = size / numberOfProcesses;
    auto totalResult = new double[size];
    auto storageVector = new double[size];
    auto resultOfMultiplicationByConst = new double[size];
    auto partOfMultiplication = new double[numberOfRows];
    double lowerPartSecondNorm = GetSecondNormOfVector(vectorOfRightSide, size);
    MultiplyVectorByConst(resultOfMultiplicationByConst, vectorOfRightSide, size, -1);
    double upperPartSecondNorm = GetSecondNormOfVector(resultOfMultiplicationByConst, size);

    auto resultOfSubtraction = new double[size];
    while (!CheckStopCondition(upperPartSecondNorm, lowerPartSecondNorm)) {
        MultiplyMatrixByVector(partOfMultiplication, vectorOfSolution, size, rank, numberOfProcesses);
        MPI_Allgather(partOfMultiplication, numberOfRows, MPI_DOUBLE, storageVector, numberOfRows, MPI_DOUBLE, MPI_COMM_WORLD);
        SubtractionOfVectors(resultOfSubtraction, storageVector, vectorOfRightSide, size);
        MultiplyVectorByConst(resultOfMultiplicationByConst, resultOfSubtraction, size, TAU);
        SubtractionOfVectors(totalResult, vectorOfSolution, resultOfMultiplicationByConst, size);
        CopyVectorToVector(totalResult, vectorOfSolution, size);
        upperPartSecondNorm = GetSecondNormOfVector(resultOfSubtraction, size);
    }
    CleanUp(vectorOfSolution, vectorOfRightSide, storageVector, resultOfMultiplicationByConst, partOfMultiplication, resultOfSubtraction);
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
    int numberOfProcesses;
    int currentRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    int matrixSize = ORIGIN_SIZE;
    if (!IsCorrectSize(ORIGIN_SIZE, numberOfProcesses)) {
        matrixSize = FindCorrectSize(ORIGIN_SIZE, numberOfProcesses);
    }
    auto vectorOfSolution = GenerateVectorOfSolution(matrixSize);
    double start = MPI_Wtime();
    vectorOfSolution = MethodOfSimpleIteration(vectorOfSolution, currentRank, matrixSize, numberOfProcesses);
    double end = MPI_Wtime();
    if (currentRank == numberOfProcesses - 1) {
        PrintVector(vectorOfSolution, ORIGIN_SIZE);
        std::cout << "[Time]: " << end - start << " (sec)." << std::endl;
    }
    delete[] vectorOfSolution;
    MPI_Finalize();

    return 0;
}
