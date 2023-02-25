#include <iostream>
#include <cmath>
#include <mpi.h>

const int ORIGIN_SIZE = 3000;
const double EPSILON = 1e-5;
const double TAU = 1e-5;

double* GeneratePartOfVectorOfSolution(int size, int numberOfProcesses) {
    int numberOfRows = size / numberOfProcesses;
    auto vector = new double[numberOfRows];
    for (int i = 0; i < numberOfRows; ++i) {
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
    int numberOfRows = size / numberOfProcesses;
    auto vector = new double[numberOfRows];
    for (int i = 0; i < numberOfRows; ++i) {
        vector[i] = ORIGIN_SIZE + 1;
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

void CopyVectorToVector(const double* source, double* destination, int size) {
    for (int i = 0; i < size; ++i) {
        destination[i] = source[i];
    }
}

double GetSecondNormOfVector(double* vector, int size, int numberOfProcesses) {
    double totalSum = 0;
    int currentRank = 0;
    auto bufferOfPartOfVector = new double [size];
    CopyVectorToVector(vector, bufferOfPartOfVector, size);
    while (currentRank < numberOfProcesses) {
        MPI_Bcast(bufferOfPartOfVector, size, MPI_DOUBLE, currentRank,
                  MPI_COMM_WORLD);
        for (int i = 0; i < size; ++i) {
            totalSum += bufferOfPartOfVector[i] * bufferOfPartOfVector[i];
        }
        ++currentRank;
    }
    return sqrt(totalSum);
}

void MultiplyMatrixByVector(double* vectorOfResult, const double* partOfVector, int size, int rank, int numberOfProcesses) {
    double* partOfMatrix = GeneratePartOfMatrix(rank, numberOfProcesses, size);
    int numberOfRows = size / numberOfProcesses;
    auto bufferOfPartOfVector = new double[numberOfRows];
    CopyVectorToVector(partOfVector, bufferOfPartOfVector, numberOfRows);
    int currentRank = 0;
    for (int i = 0; i < numberOfRows; ++i) {
        vectorOfResult[i] = 0;
        MPI_Bcast(bufferOfPartOfVector, numberOfRows, MPI_DOUBLE, currentRank, MPI_COMM_WORLD);
        for (int j = 0; j < size; ++j) {
            vectorOfResult[i] += partOfMatrix[i * numberOfRows + j] * bufferOfPartOfVector[j % numberOfRows];
        }
        if (i > numberOfRows) {
            ++currentRank;
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

void CleanUp(const double* partOfVectorOfSolution, const double* partOfVectorOfRightSide,
                  const double* partOfResultOfMultiplicationByConst, const double* partOfMultiplication,
                  const double* partOfResultOfSubtraction) {
    delete[] partOfVectorOfSolution;
    delete[] partOfVectorOfRightSide;
    delete[] partOfResultOfMultiplicationByConst;
    delete[] partOfMultiplication;
    delete[] partOfResultOfSubtraction;
}

double* MethodOfSimpleIteration(double* partOfVectorOfSolution, int rank, int size, int numberOfProcesses) {
    auto partOfVectorOfRightSide = GeneratePartOfVectorOfRightSide(size, rank, numberOfProcesses);
    int numberOfRows = size / numberOfProcesses;
    SetFirstApproximation(partOfVectorOfSolution, partOfVectorOfRightSide, numberOfRows);

    auto partOfTotalResult = new double[numberOfRows];
    auto partOfResultOfMultiplicationByConst = new double[numberOfRows];
    auto partOfMultiplication = new double[numberOfRows];
    double lowerPartSecondNorm = GetSecondNormOfVector(partOfVectorOfRightSide, numberOfRows, numberOfProcesses);
    MultiplyVectorByConst(partOfResultOfMultiplicationByConst, partOfVectorOfRightSide, numberOfRows, -1);
    double upperPartSecondNorm = GetSecondNormOfVector(partOfResultOfMultiplicationByConst, numberOfRows, numberOfProcesses);

    auto partOfResultOfSubtraction = new double[numberOfRows];
    while (!CheckStopCondition(upperPartSecondNorm, lowerPartSecondNorm)) {
        MultiplyMatrixByVector(partOfMultiplication, partOfVectorOfSolution, size, rank, numberOfProcesses);
        SubtractionOfVectors(partOfResultOfSubtraction, partOfMultiplication, partOfVectorOfRightSide, numberOfRows);
        MultiplyVectorByConst(partOfResultOfMultiplicationByConst, partOfResultOfSubtraction, numberOfRows, TAU);
        SubtractionOfVectors(partOfTotalResult, partOfVectorOfSolution, partOfResultOfMultiplicationByConst, numberOfRows);
        CopyVectorToVector(partOfTotalResult, partOfVectorOfSolution, numberOfRows);
        upperPartSecondNorm = GetSecondNormOfVector(partOfResultOfSubtraction, numberOfRows, numberOfProcesses);
    }
    auto totalResult = new double[size];
    MPI_Allgather(partOfVectorOfSolution, numberOfRows, MPI_DOUBLE, totalResult, numberOfRows, MPI_DOUBLE, MPI_COMM_WORLD);
    CleanUp(partOfVectorOfSolution, partOfVectorOfRightSide, partOfResultOfMultiplicationByConst, partOfMultiplication, partOfResultOfSubtraction);
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
    auto vectorOfSolution = GeneratePartOfVectorOfSolution(matrixSize, numberOfProcesses);
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
