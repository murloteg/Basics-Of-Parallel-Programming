#include <iostream>
#include <cmath>
#include <sys/time.h>

const int ORIGIN_SIZE = 2049;
const double EPSILON = 1e-5;
const double TAU = 1e-5;

double* GenerateVectorOfSolution(int size) {
    double* vector = new double[size];
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        vector[i] = 0;
    }
    return vector;
}

double* GenerateVectorOfRightSide(int size) {
    double* vector = new double[size];
#pragma omp parallel for
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

double* GenerateMatrix(int size) {
    double* matrix = new double[size * size];
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = 1;
        }
    }
    int startPosition = 0;
    for (int i = 0; i < size; ++i) {
        matrix[startPosition] = 2;
        startPosition = startPosition + size + 1;
    }
    return matrix;
}

void SetFirstApproximation(double* vectorOfSolution, const double* vectorOfRightSide, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        vectorOfSolution[i] = vectorOfRightSide[i];
    }
}

double GetSecondNormOfVector(const double* vector, int size) {
    double totalSum = 0;
#pragma omp parallel for reduction(+: totalSum)
    for (int i = 0; i < size; ++i) {
        totalSum += vector[i] * vector[i];
    }
    return sqrt(totalSum);
}

void MultiplyMatrixByVector(double* vectorOfResult, const double* matrix, const double* vector, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        vectorOfResult[i] = 0;
        for (int j = 0; j < size; ++j) {
            vectorOfResult[i] += matrix[i * size + j] * vector[j];
        }
    }
}

void SubtractionOfVectors(double* result, const double* first, const double* second, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] - second[i];
    }
}

bool CheckStopCondition(double firstNorm, double secondNorm) {
    return (firstNorm / secondNorm) < EPSILON;
}

void MultiplyVectorByConst(double* result, const double* first, int size, double constValue) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        result[i] = first[i] * constValue;
    }
}

void CopyVectorToVector(const double* source, double* destination, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        destination[i] = source[i];
    }
}

void CleanUp(const double* matrix, const double* vectorOfSolution, const double* vectorOfRightSide,
             const double* resultOfMultiplicationByConst,
             const double* resultOfMultiplication, const double* resultOfSubtraction) {
    delete[] matrix;
    delete[] vectorOfSolution;
    delete[] vectorOfRightSide;
    delete[] resultOfMultiplicationByConst;
    delete[] resultOfMultiplication;
    delete[] resultOfSubtraction;
}

double* MethodOfSimpleIteration(double* vectorOfSolution, int size) {
    double* vectorOfRightSide = GenerateVectorOfRightSide(size);
    SetFirstApproximation(vectorOfSolution, vectorOfRightSide, size);

    double* matrix = GenerateMatrix(size);
    double* totalResult = new double[size];
    double* resultOfMultiplicationByConst = new double[size];
    double* resultOfMultiplication = new double[size];
    double lowerPartSecondNorm = GetSecondNormOfVector(vectorOfRightSide, size);
    MultiplyVectorByConst(resultOfMultiplicationByConst, vectorOfRightSide, size, -1);
    double upperPartSecondNorm = GetSecondNormOfVector(resultOfMultiplicationByConst, size);

    double* resultOfSubtraction = new double[size];
    while (!CheckStopCondition(upperPartSecondNorm, lowerPartSecondNorm)) {
        MultiplyMatrixByVector(resultOfMultiplication, matrix, vectorOfSolution, size);
        SubtractionOfVectors(resultOfSubtraction, resultOfMultiplication, vectorOfRightSide, size);
        MultiplyVectorByConst(resultOfMultiplicationByConst, resultOfSubtraction, size, TAU);
        SubtractionOfVectors(totalResult, vectorOfSolution, resultOfMultiplicationByConst, size);
        CopyVectorToVector(totalResult, vectorOfSolution, size);
        upperPartSecondNorm = GetSecondNormOfVector(resultOfSubtraction, size);
    }
    CleanUp(matrix, vectorOfSolution, vectorOfRightSide, resultOfMultiplicationByConst, resultOfMultiplication, resultOfSubtraction);
    return totalResult;
}

int main(int argc, char** argv) {
    int size = ORIGIN_SIZE;
    double* vectorOfSolution = GenerateVectorOfSolution(size);
    struct timeval start, end;
    gettimeofday(&start, nullptr);
    vectorOfSolution = MethodOfSimpleIteration(vectorOfSolution, size);
    gettimeofday(&end, nullptr);

    PrintVector(vectorOfSolution, size);
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double elapsedTime = seconds + microseconds * 1e-6;
    std::cout << "[Time]: " << elapsedTime << " (sec)." << std::endl;

    delete[] vectorOfSolution;

    return 0;
}
