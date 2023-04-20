#include <iostream>
#include <cfloat>
#include <cmath>
#include <mpi.h>

#define CHECK_PARAMETER 10
#define EPSILON 1e-8
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 513,
    NUMBER_OF_POINTS_Y = 319,
    NUMBER_OF_POINTS_Z = 387
};

/* f(x, y, z) = x^2 + y^2 + z^2 */
float CalculateFunctionValue(float x, float y, float z) {
    return (x * x) + (y * y) + (z * z);
}

/* p(x, y, z) = 6 - EQUATION_PARAMETER * f(x, y, z) */
float CalculateRightPartOfEquationValue(float x, float y, float z) {
    return (6 - EQUATION_PARAMETER * CalculateFunctionValue(x, y, z));
}

float GetAreaLength() {
    return AREA_END_COORDINATE - AREA_START_COORDINATE;
}

float CalculateDistanceBetweenPoints(int numberOfPoints) {
    return GetAreaLength() / (numberOfPoints - 1);
}

bool IsBoundaryPositionInXYPlane(int x, int y, int numberOfPointsX, int numberOfPointsY) {
    return (x == 0 || (x == numberOfPointsX - 1) || y == 0 || (y == numberOfPointsY - 1));
}

bool IsBoundaryPositionInXZPlane(int x, int z, int numberOfPointsX, int numberOfPointsZ) {
    return (x == 0 || (x == numberOfPointsX - 1) || z == 0 || (z == numberOfPointsZ - 1));
}

bool IsBoundaryPositionInYZPlane(int y, int z, int numberOfPointsY, int numberOfPointsZ) {
    return (y == 0 || (y == numberOfPointsY - 1) || z == 0 || (z == numberOfPointsZ - 1));
}

bool IsBoundaryPosition(int x, int y, int z, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    bool IsBoundaryPosition = IsBoundaryPositionInXYPlane(x, y, numberOfPointsX, numberOfPointsY) ||
            IsBoundaryPositionInXZPlane(x, z, numberOfPointsX, numberOfPointsZ) ||
            IsBoundaryPositionInYZPlane(y, z, numberOfPointsY, numberOfPointsZ);
    return IsBoundaryPosition;
}

float GetCurrentCoordinateValue(int startPointCoordinate, int index, float distanceBetweenPoints) {
    return startPointCoordinate + index * distanceBetweenPoints;
}

void SetInitialApproximationToPartOfGrid(float* partOfGridWithPoints, int numberOfPlanesXY) {
    int position = 0;
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                position = i * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + j * NUMBER_OF_POINTS_X + k;
                partOfGridWithPoints[position] = INITIAL_APPROXIMATION;
            }
        }
    }
}

/* position formula: (i * N_y * N_x + j * N_x + k) */
void SetBoundaryConditionals(float* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
    int position = 0;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                position = i * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + j * NUMBER_OF_POINTS_X + k;
                int handledBlocks = 0;
                for (int count = 0; count < currentRank; ++count) {
                    handledBlocks += countOfPlanes[count];
                }

                if (IsBoundaryPosition(k, j, i + handledBlocks, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
                    float distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
                    float distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
                    float distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

                    float localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, k, distanceBetweenCoordinateX);
                    float localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, j, distanceBetweenCoordinateY);
                    float localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, i, distanceBetweenCoordinateZ);
                    partOfGridWithPoints[position] = CalculateFunctionValue(localX, localY, localZ);
                }
            }
        }
    }
}

float* CreatePartOfGridWithPoints(int* countOfPlanes, int currentRank) {
    int numberOfPlanesXY = countOfPlanes[currentRank];
    float* partOfGridWithPoints = new float[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y * numberOfPlanesXY];
    SetInitialApproximationToPartOfGrid(partOfGridWithPoints, numberOfPlanesXY);
    SetBoundaryConditionals(partOfGridWithPoints, countOfPlanes, currentRank);
    return partOfGridWithPoints;
}

float CalculateDifferenceBetweenPreviousValueAndNewValue(float previousValue, float newValue) {
    return fabs(newValue - previousValue);
}

float FindLeftMultiplier(float distanceBetweenCoordinateX, float distanceBetweenCoordinateY, float distanceBetweenCoordinateZ) {
    return 1 / (2 / (distanceBetweenCoordinateX * distanceBetweenCoordinateX) + 2 / (distanceBetweenCoordinateY * distanceBetweenCoordinateY)
                + 2 / (distanceBetweenCoordinateZ * distanceBetweenCoordinateZ) + EQUATION_PARAMETER);
}

bool IsThisPointTopNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
    int position = indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK;
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int firstIndex = handledBlocks * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
    return (position >= firstIndex) && (position < firstIndex + NUMBER_OF_POINTS_Y);
}

bool IsThisPointBottomNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
    int position = indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK;
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int lastIndex = (handledBlocks + countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
    return (position > lastIndex - NUMBER_OF_POINTS_Y) && (position <= lastIndex);
}

void SendBoundaryPoint(float* sendValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        int tag = destinationRank * 100;
        MPI_Isend(sendValue, 1, MPI_FLOAT, destinationRank, tag, MPI_COMM_WORLD, request);
    }
}

void ReceiveBoundaryPoint(float* receivedValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        int tag = destinationRank * 100;
        MPI_Irecv(receivedValue, 1, MPI_FLOAT, destinationRank, tag, MPI_COMM_WORLD, request);
    }
}

float FindRightMultiplier(float* gridOfPoints, int indexI, int indexJ, int indexK, float distanceBetweenCoordinateX,
                           float distanceBetweenCoordinateY, float distanceBetweenCoordinateZ,
                           int* countOfPlanes, float* currentValue, float* bottomValue, float* topValue,
                           int currentRank, int numberOfProcesses) {
    MPI_Request bottomNeighborRequest;
    MPI_Request topNeighborRequest;


    currentValue[0] = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];

    /* bottom --> for CURRENT process; top --> for CURRENT process! */


    if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank) &&
            IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        /* send this point to NEXT process */
        SendBoundaryPoint(currentValue, currentRank + 1, numberOfProcesses, &topNeighborRequest);
        /* send this point to PREVIOUS process */
        SendBoundaryPoint(currentValue, currentRank - 1, numberOfProcesses, &bottomNeighborRequest);

        /* receive bottom boundary point FROM previous process TO current process! */
        ReceiveBoundaryPoint(bottomValue, currentRank - 1, numberOfProcesses, &topNeighborRequest);
        /* receive top boundary point FROM next process TO current process! */
        ReceiveBoundaryPoint(bottomValue, currentRank + 1, numberOfProcesses, &bottomNeighborRequest);
    } else if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        /* send this point to NEXT process */
        SendBoundaryPoint(currentValue, currentRank + 1, numberOfProcesses, &topNeighborRequest);
        topValue[0] = gridOfPoints[(indexI - 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];
        /* receive top boundary point FROM next process TO current process! */
        ReceiveBoundaryPoint(bottomValue, currentRank + 1, numberOfProcesses, &bottomNeighborRequest);
    } else if (IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        /* send this point to PREVIOUS process */
        SendBoundaryPoint(currentValue, currentRank - 1, numberOfProcesses, &bottomNeighborRequest);
        bottomValue[0] = gridOfPoints[(indexI + 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];
        /* receive bottom boundary point FROM previous process TO current process! */
        ReceiveBoundaryPoint(bottomValue, currentRank - 1, numberOfProcesses, &topNeighborRequest);
    } else {
        /* getting values from local (in current process) neighbors */
        bottomValue[0] = gridOfPoints[(indexI) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ + 1) * NUMBER_OF_POINTS_X + indexK];
        topValue[0] = gridOfPoints[(indexI) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ - 1) * NUMBER_OF_POINTS_X + indexK];
    }

    float firstNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ + 1) * NUMBER_OF_POINTS_X + indexK];
    float secondNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ - 1) * NUMBER_OF_POINTS_X + indexK];

    float firstNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK + 1];
    float secondNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK - 1];

    float localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    float localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    float localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI, distanceBetweenCoordinateZ);
    float rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

    /* wait all sent boundary points from another processes */
    if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&topNeighborRequest, MPI_STATUS_IGNORE);
    }
    if (IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&bottomNeighborRequest, MPI_STATUS_IGNORE);
    }

    float firstNeighborZ = bottomValue[0];
    float secondNeighborZ = topValue[0];

    return (firstNeighborZ + secondNeighborZ) / distanceBetweenCoordinateZ * distanceBetweenCoordinateZ
            + (firstNeighborY * secondNeighborY) / distanceBetweenCoordinateY * distanceBetweenCoordinateY
            + (firstNeighborX * secondNeighborX) / distanceBetweenCoordinateX * distanceBetweenCoordinateX
            - rightPartValue;
}

float GetCheckValue(int indexI, int indexJ, int indexK, float distanceBetweenCoordinateX,
                     float distanceBetweenCoordinateY, float distanceBetweenCoordinateZ) {
    float localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    float localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    float localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI, distanceBetweenCoordinateZ);
    return CalculateFunctionValue(localX, localY, localZ);
}

bool ThisProcessHasOnlyBoundedPositions(int* countOfPlanes, int currentRank) {
    int expectedPositions = countOfPlanes[currentRank] * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
    int counter = 0;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int handledBlocks = 0;
                for (int count = 0; count < currentRank; ++count) {
                    handledBlocks += countOfPlanes[count];
                }

                if (IsBoundaryPosition(k, j, i + handledBlocks, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
                    ++counter;
                }
            }
        }
    }
    return (counter == expectedPositions);
}

bool DoAllProcessesHaveOnlyBoundedPositions(int* countOfPlanes, int numberOfProcesses) {
    int boundedProcessesCounter = 0;
    for (int i = 0; i < numberOfProcesses; ++i) {
        if (ThisProcessHasOnlyBoundedPositions(countOfPlanes, i)) {
            ++boundedProcessesCounter;
        }
    }
    return boundedProcessesCounter == numberOfProcesses;
}

float GetMinDifferenceFromArray(float* arrayWithMaxDifferences, int numberOfProcesses) {
    float minDifference = FLT_MAX;
    for (int i = 0; i < numberOfProcesses; ++i) {
        minDifference = std::min(minDifference, arrayWithMaxDifferences[i]);
    }
    return minDifference;
}

bool CompareResult(float* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
    if (ThisProcessHasOnlyBoundedPositions(countOfPlanes, currentRank)) {
        return true;
    }

    float distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    float distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    float distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    float maxDifference = FLT_MIN;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int position = i * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + j * NUMBER_OF_POINTS_X + k;
                float checkValue = GetCheckValue(i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                maxDifference = std::max(maxDifference, (float) (fabs(partOfGridWithPoints[position] - checkValue)));
            }
        }
    }

    std::cout << "MAX DIFFERENCE: " << maxDifference << " FROM PROCESS: \"" << currentRank << "\"\n";
    return (maxDifference > FLT_MIN && maxDifference < CHECK_PARAMETER);
}

void IterativeProcessOfJacobiAlgorithm(float* partOfGridWithPoints, int* countOfPlanes, int currentRank, int numberOfProcesses) {
    if (DoAllProcessesHaveOnlyBoundedPositions(countOfPlanes, numberOfProcesses)) {
        if (currentRank == 0) {
            std::cout << "ONLY PROCESSES WITH BOUNDED BLOCKS!\n";
        }
        return;
    }

    float distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    float distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    float distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    float* currentMaxDifference = new float[1];
    currentMaxDifference[0] = FLT_MAX;

    float* arrayWithMaxDifferences = new float[numberOfProcesses];
    for (int i = 0; i < numberOfProcesses; ++i) {
        arrayWithMaxDifferences[i] = FLT_MAX;
    }

    float* currentValue = new float[1];
    float* bottomValue = new float[1];
    float* topValue = new float[1];

    int numberOfPlanesXY = countOfPlanes[currentRank];
    float maxDifference = FLT_MAX;
    while (GetMinDifferenceFromArray(arrayWithMaxDifferences, numberOfProcesses) >= EPSILON) {
        for (int i = 0; i < numberOfPlanesXY; ++i) {
            for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
                for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                    int position = i * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + j * NUMBER_OF_POINTS_X + k;
                    int handledBlocks = 0;
                    for (int count = 0; count < currentRank; ++count) {
                        handledBlocks += countOfPlanes[count];
                    }

                    if (!IsBoundaryPosition(k, j, i + handledBlocks, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
                        float previousValue = partOfGridWithPoints[position];
                        float leftMultiplier = FindLeftMultiplier(distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                        float rightMultiplier = FindRightMultiplier(partOfGridWithPoints, i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ, countOfPlanes, currentValue, bottomValue, topValue, currentRank, numberOfProcesses);
                        partOfGridWithPoints[position] = leftMultiplier * rightMultiplier;
                        maxDifference = std::min(maxDifference, CalculateDifferenceBetweenPreviousValueAndNewValue(previousValue, partOfGridWithPoints[position]));
                        currentMaxDifference[0] = maxDifference;
                    }
                }
            }
        }
        MPI_Allgather(currentMaxDifference, 1, MPI_FLOAT, arrayWithMaxDifferences, 1, MPI_FLOAT, MPI_COMM_WORLD);
    }

    if (CompareResult(partOfGridWithPoints, countOfPlanes, currentRank)) {
        std::cout << "CORRECT\n";
    } else {
        std::cout << "INCORRECT!\n";
    }

    delete[] bottomValue;
    delete[] topValue;
    delete[] currentValue;
    delete[] currentMaxDifference;
    delete[] arrayWithMaxDifferences;
}

void DebugPrintOfGrid(float* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    int position = 0;
    for (int i = 0; i < numberOfPointsZ; ++i) {
        for (int j = 0; j < numberOfPointsY; ++j) {
            for (int k = 0; k < numberOfPointsX; ++k) {
                position = i * numberOfPointsX * numberOfPointsY + j * numberOfPointsX + k;
                std::cout << gridOfPoints[position] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void FillCountOfPlanesArray(int* countOfPlanes, int numberOfProcesses) {
    int counter = 0;
    while (counter < NUMBER_OF_POINTS_Z) {
        int currentIndex = counter % numberOfProcesses;
        countOfPlanes[currentIndex] += 1;
        ++counter;
    }
}

void PrintCountOfPlanesArray(int* countOfPlanes, int numberOfProcesses) {
    std::cout << "DISTRIBUTION OF PLANES XY: ";
    for (int i = 0; i < numberOfProcesses; ++i) {
        std::cout << countOfPlanes[i] << " ";
    }
    std::cout << "\n";
}

void CleanUp(float* gridOfPoints, int* countOfPlanes) {
    delete[] gridOfPoints;
    delete[] countOfPlanes;
}

int main(int argc, char** argv) {
    int numberOfProcesses = 0;
    int currentRank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    int* countOfPlanes = new int[numberOfProcesses];
    FillCountOfPlanesArray(countOfPlanes, numberOfProcesses);
    if (currentRank == 0) {
        PrintCountOfPlanesArray(countOfPlanes, numberOfProcesses);
    }

    float* partOfGridWithPoints = CreatePartOfGridWithPoints(countOfPlanes, currentRank);
    double start = MPI_Wtime();
    IterativeProcessOfJacobiAlgorithm(partOfGridWithPoints, countOfPlanes, currentRank, numberOfProcesses);
    double end = MPI_Wtime();
    if (currentRank == 0) {
        std::cout << "Elapsed time: " << end - start << " [sec]\n";
    }

    CleanUp(partOfGridWithPoints, countOfPlanes);

    MPI_Finalize();

    return 0;
}
