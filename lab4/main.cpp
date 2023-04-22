#include <iostream>
#include <cmath>
#include <mpi.h>
#include <cfloat>

#define CHECK_PARAMETER 1
#define EPSILON 1e-8
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 4,
    NUMBER_OF_POINTS_Y = 4,
    NUMBER_OF_POINTS_Z = 4
};

/* f(x, y, z) = x^2 + y^2 + z^2 */
double CalculateFunctionValue(double x, double y, double z) {
    return (x * x) + (y * y) + (z * z);
}

/* p(x, y, z) = 6 - EQUATION_PARAMETER * f(x, y, z) */
double CalculateRightPartOfEquationValue(double x, double y, double z) {
    return (6 - EQUATION_PARAMETER * CalculateFunctionValue(x, y, z));
}

double GetAreaLength() {
    return AREA_END_COORDINATE - AREA_START_COORDINATE;
}

double CalculateDistanceBetweenPoints(int numberOfPoints) {
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

double GetCurrentCoordinateValue(int startPointCoordinate, int index, double distanceBetweenPoints) {
    return (double) startPointCoordinate + index * distanceBetweenPoints;
}

int GetGridPosition(int indexI, int indexJ, int indexK) {
    return indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK;
}

void SetInitialApproximationToPartOfGrid(double* partOfGridWithPoints, int numberOfPlanesXY) {
    int position = 0;
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                position = GetGridPosition(i, j, k);
                partOfGridWithPoints[position] = INITIAL_APPROXIMATION;
            }
        }
    }
}

/* position formula: (i * N_y * N_x + j * N_x + k) */
void SetBoundaryConditionals(double* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
    int position = 0;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                position = GetGridPosition(i, j, k);
                int handledBlocks = 0;
                for (int count = 0; count < currentRank; ++count) {
                    handledBlocks += countOfPlanes[count];
                }

                if (IsBoundaryPosition(k, j, i + handledBlocks, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
                    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
                    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
                    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

                    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, k, distanceBetweenCoordinateX);
                    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, j, distanceBetweenCoordinateY);
                    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, i, distanceBetweenCoordinateZ);
                    partOfGridWithPoints[position] = CalculateFunctionValue(localX, localY, localZ);
                }
            }
        }
    }
}

double* CreatePartOfGridWithPoints(int* countOfPlanes, int currentRank) {
    int numberOfPlanesXY = countOfPlanes[currentRank];
    double* partOfGridWithPoints = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y * numberOfPlanesXY];
    SetInitialApproximationToPartOfGrid(partOfGridWithPoints, numberOfPlanesXY);
    SetBoundaryConditionals(partOfGridWithPoints, countOfPlanes, currentRank);
    return partOfGridWithPoints;
}

double FindLeftMultiplier(double distanceBetweenCoordinateX, double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ) {
    return 1 / (2 / (distanceBetweenCoordinateX * distanceBetweenCoordinateX) + 2 / (distanceBetweenCoordinateY * distanceBetweenCoordinateY)
                + 2 / (distanceBetweenCoordinateZ * distanceBetweenCoordinateZ) + EQUATION_PARAMETER);
}

bool IsThisPointTopNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
    int position = GetGridPosition(indexI, indexJ, indexK);
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int firstIndex = handledBlocks * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
//    int firstIndex = countOfPlanes[currentRank] * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
    std::cout << "top boundaries: [" << firstIndex << " " << firstIndex + NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1 << "]\n";
    return (position >= firstIndex) && (position < firstIndex + NUMBER_OF_POINTS_Y * NUMBER_OF_POINTS_X);
}

bool IsThisPointBottomNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
    int position = GetGridPosition(indexI, indexJ, indexK);
    std::cout << "(x, y, z): (" << indexK << " " << indexJ << " " << indexI << ") " << " pos: " << position << " rank: " <<  currentRank << "\n";
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int lastIndex = (handledBlocks + countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
//    int lastIndex = (countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
    std::cout << "bottom boundaries: [" << lastIndex - NUMBER_OF_POINTS_Y * NUMBER_OF_POINTS_X + 1 << " " << lastIndex << "]\n";
    return (position > lastIndex - NUMBER_OF_POINTS_Y * NUMBER_OF_POINTS_X) && (position <= lastIndex);
}

void SendBoundaryPoint(double* sendValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        MPI_Isend(sendValue, 1, MPI_DOUBLE, destinationRank, 1, MPI_COMM_WORLD, request);
    }
}

void ReceiveBoundaryPoint(double* receivedValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        MPI_Irecv(receivedValue, 1, MPI_DOUBLE, destinationRank, MPI_ANY_TAG, MPI_COMM_WORLD, request);
    }
}

double FindRightMultiplier(double* gridOfPoints, int indexI, int indexJ, int indexK, double distanceBetweenCoordinateX,
                           double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ,
                           int* countOfPlanes, int handledBlocks, int currentRank, int numberOfProcesses) {
    MPI_Request bottomNeighborRequest;
    MPI_Request topNeighborRequest;

    double currentValue = gridOfPoints[GetGridPosition(indexI, indexJ, indexK)];
    /* bottom --> for CURRENT process; top --> for CURRENT process! */
    double topValue = 0;
    double bottomValue = 0;
    
    if (IsThisPointBottomNeighbor(indexI + handledBlocks, indexJ, indexK, countOfPlanes, currentRank) &&
            IsThisPointTopNeighbor(indexI + handledBlocks, indexJ, indexK, countOfPlanes, currentRank)) {
        /* send this point to NEXT process */
        SendBoundaryPoint(&currentValue, currentRank + 1, numberOfProcesses, &topNeighborRequest);
        /* send this point to PREVIOUS process */
        SendBoundaryPoint(&currentValue, currentRank - 1, numberOfProcesses, &bottomNeighborRequest);

        /* receive bottom boundary point FROM previous process TO current process! */
        ReceiveBoundaryPoint(&bottomValue, currentRank - 1, numberOfProcesses, &topNeighborRequest);
        /* receive top boundary point FROM next process TO current process! */
        ReceiveBoundaryPoint(&bottomValue, currentRank + 1, numberOfProcesses, &bottomNeighborRequest);
    } else if (IsThisPointBottomNeighbor(indexI + handledBlocks, indexJ, indexK, countOfPlanes, currentRank)) {
        /* send this point to NEXT process */
        SendBoundaryPoint(&currentValue, currentRank + 1, numberOfProcesses, &topNeighborRequest);
        topValue = gridOfPoints[GetGridPosition(indexI - 1, indexJ, indexK)];
        /* receive top boundary point FROM next process TO current process! */
        ReceiveBoundaryPoint(&bottomValue, currentRank + 1, numberOfProcesses, &bottomNeighborRequest);
    } else if (IsThisPointTopNeighbor(indexI + handledBlocks, indexJ, indexK, countOfPlanes, currentRank)) {
        /* send this point to PREVIOUS process */
        SendBoundaryPoint(&currentValue, currentRank - 1, numberOfProcesses, &bottomNeighborRequest);
        bottomValue = gridOfPoints[GetGridPosition(indexI + 1, indexJ, indexK)];
        /* receive bottom boundary point FROM previous process TO current process! */
        ReceiveBoundaryPoint(&bottomValue, currentRank - 1, numberOfProcesses, &topNeighborRequest);
    } else {
        /* getting values from local (in current process) neighbors */
        int maxPos = (countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
        int position = (indexI + 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ) * NUMBER_OF_POINTS_X + indexK;
        if (position >= maxPos) {
            std::cout << "Overflow! position: " << position << "; max pos: " << maxPos << "; index I: " << indexI << "; rank: " << currentRank << "\n";
        }
        bottomValue = gridOfPoints[GetGridPosition(indexI + 1, indexJ, indexK)];
        topValue = gridOfPoints[GetGridPosition(indexI - 1, indexJ, indexK)];
    }

    double firstNeighborY = gridOfPoints[GetGridPosition(indexI, indexJ + 1, indexK)];
    double secondNeighborY = gridOfPoints[GetGridPosition(indexI, indexJ - 1, indexK)];

    double firstNeighborX = gridOfPoints[GetGridPosition(indexI, indexJ, indexK + 1)];
    double secondNeighborX = gridOfPoints[GetGridPosition(indexI, indexJ, indexK - 1)];

    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI + handledBlocks, distanceBetweenCoordinateZ);
    double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

    /* wait all sent boundary points from another processes */
    if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&topNeighborRequest, MPI_STATUS_IGNORE);
    }
    if (IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&bottomNeighborRequest, MPI_STATUS_IGNORE);
    }

    double firstNeighborZ = bottomValue;
    double secondNeighborZ = topValue;

    if (currentRank == 0) {
        printf("TOTAL LOG: rank: %d \n (x, y, z): (%d %d %d) \n rank's value: %lf \n left X value: %lf \n right X value: %lf \n lower Y value: %lf \n upper Y value: %lf \n prev Z value: %lf \n next Z value: %lf \n\n", currentRank, indexK, indexJ, indexI, currentValue, secondNeighborX, firstNeighborX, secondNeighborY, firstNeighborY, topValue, bottomValue);
    }

    return (firstNeighborZ + secondNeighborZ) / distanceBetweenCoordinateZ * distanceBetweenCoordinateZ
            + (firstNeighborY * secondNeighborY) / distanceBetweenCoordinateY * distanceBetweenCoordinateY
            + (firstNeighborX * secondNeighborX) / distanceBetweenCoordinateX * distanceBetweenCoordinateX
            - rightPartValue;
}

double GetCheckValue(int indexI, int indexJ, int indexK, double distanceBetweenCoordinateX,
                     double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ) {
    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI, distanceBetweenCoordinateZ);
    return CalculateFunctionValue(localX, localY, localZ);
}

double GetMaxDifferenceFromArray(double* arrayWithMaxDifferences, int numberOfProcesses) {
    double maxDifference = DBL_MIN;
    for (int i = 0; i < numberOfProcesses; ++i) {
        maxDifference = std::max(maxDifference, arrayWithMaxDifferences[i]);
//        std::cout << "value: " << arrayWithMaxDifferences[i] << " ";
    }
    std::cout << "\n";
    return maxDifference;
}

bool CompareResult(double* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double maxDifference = DBL_MIN;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int position = GetGridPosition(i, j, k);
                double checkValue = GetCheckValue(i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                maxDifference = std::max(maxDifference, fabs(partOfGridWithPoints[position] - checkValue));
                if (currentRank == 0 && (fabs(partOfGridWithPoints[position] - checkValue)) >= maxDifference) {
                    std::cout << "real pos: " << partOfGridWithPoints[position] << " expected: " << checkValue << "\n";
                }
            }
        }
    }

    std::cout << "MAX DIFFERENCE: " << maxDifference << " FROM PROCESS: \"" << currentRank << "\"\n";
    return maxDifference < CHECK_PARAMETER;
}

void IterativeProcessOfJacobiAlgorithm(double* partOfGridWithPoints, int* countOfPlanes, int currentRank, int numberOfProcesses) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double* arrayWithMaxDifferences = new double[numberOfProcesses];
    for (int i = 0; i < numberOfProcesses; ++i) {
        arrayWithMaxDifferences[i] = DBL_MIN;
    }

    int numberOfPlanesXY = countOfPlanes[currentRank];
    while (true) {
        double maxDifference = DBL_MIN;
        for (int i = 0; i < numberOfPlanesXY; ++i) {
            for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
                for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                    int position = GetGridPosition(i, j, k);
                    int handledBlocks = 0;
                    for (int count = 0; count < currentRank; ++count) {
                        handledBlocks += countOfPlanes[count];
                    }
                    if (!IsBoundaryPosition(k, j, i + handledBlocks, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
//                        std::cout << "(x, y, z): (" << k << " " << j << " " << i << ")\n";
                        double previousValue = partOfGridWithPoints[position];
                        double leftMultiplier = FindLeftMultiplier(distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                        double rightMultiplier = FindRightMultiplier(partOfGridWithPoints, i, j, k, distanceBetweenCoordinateX,
                                                                     distanceBetweenCoordinateY, distanceBetweenCoordinateZ,
                                                                     countOfPlanes, handledBlocks, currentRank, numberOfProcesses);
                        partOfGridWithPoints[position] = leftMultiplier * rightMultiplier; // TODO
                        maxDifference = std::max(maxDifference, fabs(partOfGridWithPoints[position] - previousValue));
                    }
//                    if (currentRank == 0) {
//                        std::cout << "new value: " << partOfGridWithPoints[position] << " old value: " << previousValue << "\n";
//                    }
                }
            }
        }
//        std::cout << currentRank << " proc: " << maxDifference << " planes: " << numberOfPlanesXY << "\n";
        MPI_Allgather(&maxDifference, 1, MPI_DOUBLE, arrayWithMaxDifferences, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        if (currentRank == 0) {
            std::cout << "diff: " << GetMaxDifferenceFromArray(arrayWithMaxDifferences, numberOfProcesses) << "\n";
        }
        if (GetMaxDifferenceFromArray(arrayWithMaxDifferences, numberOfProcesses) < EPSILON) {
            break;
        }
    }
    if (CompareResult(partOfGridWithPoints, countOfPlanes, currentRank)) {
        std::cout << "CORRECT\n";
    } else {
        std::cout << "INCORRECT!\n";
    }

    delete[] arrayWithMaxDifferences;
}

void DebugPrintOfGrid(double* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    int position = 0;
    for (int i = 0; i < numberOfPointsZ; ++i) {
        for (int j = 0; j < numberOfPointsY; ++j) {
            for (int k = 0; k < numberOfPointsX; ++k) {
                position = GetGridPosition(i, j, k);
                std::cout << gridOfPoints[position] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int CalculateProcessNumberOfPlanes(int currentRank, int numberOfProcesses) {
    int numberOfPlanes = NUMBER_OF_POINTS_Z / numberOfProcesses;
    if (currentRank < NUMBER_OF_POINTS_Z % numberOfProcesses) {
        ++numberOfPlanes;
    }
    return numberOfPlanes;
}

void PrintCountOfPlanesArray(int* countOfPlanes, int numberOfProcesses) {
    std::cout << "DISTRIBUTION OF PLANES XY: ";
    for (int i = 0; i < numberOfProcesses; ++i) {
        std::cout << countOfPlanes[i] << " ";
    }
    std::cout << "\n";
}

void CleanUp(double* gridOfPoints, int* countOfPlanes) {
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
    int numberOfPlanes = CalculateProcessNumberOfPlanes(currentRank, numberOfProcesses);
    MPI_Allgather(&numberOfPlanes, 1, MPI_INT, countOfPlanes, 1, MPI_INT, MPI_COMM_WORLD);
    if (currentRank == 0) {
        PrintCountOfPlanesArray(countOfPlanes, numberOfProcesses);
    }

    double* partOfGridWithPoints = CreatePartOfGridWithPoints(countOfPlanes, currentRank);
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
