#include <iostream>
#include <cmath>
#include <mpi.h>

const int ORIGIN_SIZE = 4096;
const double EPSILON = 1e-8;
const double TAU = 1e-5;

double* GeneratePartOfVectorOfSolution(int size, int numberOfProcesses) {
    int columns = size / numberOfProcesses;
    auto vector = new double[columns];
    for (int i = 0; i < columns; ++i) {
        vector[i] = 0;
    }
    return vector;
}

int GetNumberOfExtraRows(int size) {
    return (size - ORIGIN_SIZE);
}

int GetLastElementFromArray(int* array, int size) {
    return array[size - 1];
}

double* GeneratePartOfVectorOfRightSide(int size, int rank, int numberOfProcesses) {
    int columns = size / numberOfProcesses;
    auto vector = new double[columns];
    for (int i = 0; i < columns; ++i) {
        vector[i] = ORIGIN_SIZE + 1;
    }

    auto arrayWithRowNumbers = new int[columns];
    for (int i = 0; i < columns; ++i) {
        arrayWithRowNumbers[i] = i + rank * columns + 1;
    }
    int numberOfExtraRows = GetNumberOfExtraRows(size);
    int extraRowPosition = size - numberOfExtraRows;
    if (GetLastElementFromArray(arrayWithRowNumbers, columns) > extraRowPosition) {
        for (int i = 0; i < columns; ++i) {
            if (arrayWithRowNumbers[i] > extraRowPosition) {
                vector[i] = 0;
            }
        }
    }
    delete[] arrayWithRowNumbers;
    return vector;
}

void PrintVector(double* vector, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

double* GeneratePartOfMatrix(int rank, int numberOfProcesses, int size, bool statusOfChangeOfSize) {
    int matrixSize;
    if (statusOfChangeOfSize) {
        matrixSize = size * (size / numberOfProcesses);
    }
    else {
        matrixSize = ORIGIN_SIZE * (ORIGIN_SIZE / numberOfProcesses);
    }
    int columns = (size / numberOfProcesses);
    auto matrix = new double[matrixSize];
    for (int i = 0; i < columns; ++i) {
        for (int j = 0; j < ORIGIN_SIZE; ++j) {
            matrix[i * size + j] = 1;
        }
    }
    int startPosition = columns * rank;
    for (int i = 0; i < columns; ++i) {
        if (startPosition > matrixSize) {
            break;
        }
        matrix[startPosition] = 2;
        startPosition = startPosition + size + 1;
    }
    auto arrayWithRowNumbers = new int[columns];
    for (int i = 0; i < columns; ++i) {
        arrayWithRowNumbers[i] = i + rank * columns + 1;
    }

    int numberOfExtraRows = GetNumberOfExtraRows(size);
    int extraRowPosition = size - numberOfExtraRows;
    if (GetLastElementFromArray(arrayWithRowNumbers, columns) > extraRowPosition) {
        for (int i = 0; i < columns; ++i) {
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

double* MultiplyMatrixByVector(const double* vector, int size, int rank, int numberOfProcesses, bool statusOfChangeOfSize) {
    double* partOfMatrix = GeneratePartOfMatrix(rank, numberOfProcesses, size, statusOfChangeOfSize);
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

double* SubtractionOfVectors(const double* first, const double* second, int size) {
    auto result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] - second[i];
    }
    return result;
}

bool CheckStopCondition(double firstNorm, double secondNorm) {
    return (firstNorm / secondNorm) < EPSILON;
}

double* MultiplyVectorByConst(const double* first, int size, double constValue) {
    auto result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] * constValue;
    }
    return result;
}

void CopyVectorToVector(const double* source, double* destination, int size) {
    for (int i = 0; i < size; ++i) {
        destination[i] = source[i];
    }
}

void InnerCleanUp(const double* partOfMultiplication, const double* resultOfMultiplicationByConst, const double* resultOfSubtraction) {
    delete[] partOfMultiplication;
    delete[] resultOfMultiplicationByConst;
    delete[] resultOfSubtraction;
}

void OuterCleanUp(const double* partOfVectorOfSolution, const double* partOfVectorOfRightSide,
                  const double* vectorOfSolution, const double* vectorOfRightSide, const double* storageVector) {
    delete[] partOfVectorOfSolution;
    delete[] partOfVectorOfRightSide;
    delete[] vectorOfSolution;
    delete[] vectorOfRightSide;
    delete[] storageVector;
}

double* MethodOfSimpleIteration(double* partOfVectorOfSolution, int rank, int size, int numberOfProcesses, bool statusOfChangeOfSize) {
    auto partOfVectorOfRightSide = GeneratePartOfVectorOfRightSide(size, rank, numberOfProcesses);
    auto vectorOfRightSide = new double[size];
    int columns = size / numberOfProcesses;
    MPI_Allgather(partOfVectorOfRightSide, columns, MPI_DOUBLE, vectorOfRightSide, columns, MPI_DOUBLE, MPI_COMM_WORLD);
    auto vectorOfSolution = new double[size];
    SetFirstApproximation(partOfVectorOfSolution, vectorOfRightSide, columns);
    MPI_Allgather(partOfVectorOfSolution, columns, MPI_DOUBLE, vectorOfSolution, columns, MPI_DOUBLE, MPI_COMM_WORLD);

    auto totalResult = new double[size];
    auto storageVector = new double[size];
    double lowerPartSecondNorm = GetSecondNormOfVector(vectorOfRightSide, size);
    double upperPartSecondNorm = GetSecondNormOfVector(MultiplyVectorByConst(vectorOfRightSide, size, -1), size);
    while (!CheckStopCondition(upperPartSecondNorm, lowerPartSecondNorm)) {
        double* partOfMultiplication = MultiplyMatrixByVector(vectorOfSolution, size, rank, numberOfProcesses, statusOfChangeOfSize);
        MPI_Allgather(partOfMultiplication, columns, MPI_DOUBLE, storageVector, columns, MPI_DOUBLE, MPI_COMM_WORLD);
        double* resultOfSubtraction = SubtractionOfVectors(storageVector, vectorOfRightSide, size);
        double* resultOfMultiplicationByConst = MultiplyVectorByConst(resultOfSubtraction, size, TAU);
        totalResult = SubtractionOfVectors(vectorOfSolution, resultOfMultiplicationByConst, size);
        CopyVectorToVector(totalResult, vectorOfSolution, size);
        upperPartSecondNorm = GetSecondNormOfVector(resultOfSubtraction, size);
        InnerCleanUp(partOfMultiplication, resultOfMultiplicationByConst, resultOfSubtraction);
    }
    OuterCleanUp(partOfVectorOfSolution, partOfVectorOfRightSide, vectorOfSolution, vectorOfRightSide, storageVector);
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

    bool statusOfChangeOfSize = false;
    int matrixSize = ORIGIN_SIZE;
    if (!IsCorrectSize(ORIGIN_SIZE, numberOfProcesses)) {
        matrixSize = FindCorrectSize(ORIGIN_SIZE, numberOfProcesses);
        statusOfChangeOfSize = true;
    }
    auto vectorOfSolution = GeneratePartOfVectorOfSolution(matrixSize, numberOfProcesses);
    double start = MPI_Wtime();
    vectorOfSolution = MethodOfSimpleIteration(vectorOfSolution, currentRank, matrixSize, numberOfProcesses, statusOfChangeOfSize);
    double end = MPI_Wtime();
    if (currentRank == numberOfProcesses - 1) {
        PrintVector(vectorOfSolution, ORIGIN_SIZE);
        std::cout << "[Time]: " << end - start << " (sec)." << std::endl;
    }
    delete[] vectorOfSolution;
    MPI_Finalize();

    return 0;
}
