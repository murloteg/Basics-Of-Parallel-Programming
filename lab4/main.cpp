#include <iostream>
#include <cmath>
#include <mpi.h>
#include <cfloat>

#define CHECK_PARAMETER 3
#define EPSILON 1e-18
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 213,
    NUMBER_OF_POINTS_Y = 127,
    NUMBER_OF_POINTS_Z = 109
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
    return GetAreaLength() / (double) (numberOfPoints - 1);
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

int CalculatePosition(int indexI, int indexJ, int indexK) {
    return indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK;
}

void SetInitialApproximationToPartOfGrid(double* partOfGridWithPoints, int numberOfPlanesXY) {
    int position = 0;
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                position = CalculatePosition(i, j, k);
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
                position = CalculatePosition(i, j, k);
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
    int position = CalculatePosition(indexI, indexJ, indexK);
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int firstIndex = handledBlocks * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
    return (position >= firstIndex) && (position < firstIndex + NUMBER_OF_POINTS_Y);
}

bool IsThisPointBottomNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
    int position = CalculatePosition(indexI, indexJ, indexK);
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int lastIndex = (handledBlocks + countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
    return (position > lastIndex - NUMBER_OF_POINTS_Y) && (position <= lastIndex);
}

void SendBoundaryPoint(double* sendValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        int tag = destinationRank * 100;
        MPI_Isend(sendValue, 1, MPI_FLOAT, destinationRank, tag, MPI_COMM_WORLD, request);
    }
}

void ReceiveBoundaryPoint(double* receivedValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        int tag = destinationRank * 100;
        MPI_Irecv(receivedValue, 1, MPI_FLOAT, destinationRank, tag, MPI_COMM_WORLD, request);
    }
}

double FindRightMultiplier(double* gridOfPoints, int indexI, int indexJ, int indexK, double distanceBetweenCoordinateX,
                           double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ,
                           int* countOfPlanes, double* currentValue, double* bottomValue, double* topValue,
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
        int maxPos = (countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
        if ((indexI + 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ) * NUMBER_OF_POINTS_X + indexK >= maxPos) {
            std::cout << "Overflow!\n";
        }
        
        bottomValue[0] = gridOfPoints[(indexI + 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ) * NUMBER_OF_POINTS_X + indexK];
        topValue[0] = gridOfPoints[(indexI - 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ) * NUMBER_OF_POINTS_X + indexK];
    }

    double firstNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ + 1) * NUMBER_OF_POINTS_X + indexK];
    double secondNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ - 1) * NUMBER_OF_POINTS_X + indexK];

    double firstNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK + 1];
    double secondNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK - 1];

    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI, distanceBetweenCoordinateZ);
    double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

    /* wait all sent boundary points from another processes */
    if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&topNeighborRequest, MPI_STATUS_IGNORE);
    }
    if (IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&bottomNeighborRequest, MPI_STATUS_IGNORE);
    }

    double firstNeighborZ = bottomValue[0];
    double secondNeighborZ = topValue[0];

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

bool ThisProcessHasOnlyBoundedPositions() {
    return NUMBER_OF_POINTS_X == 2 || NUMBER_OF_POINTS_Y == 2;
}

bool DoAllProcessesHaveOnlyBoundedPositions(int numberOfProcesses) {
    int boundedProcessesCounter = 0;
    for (int i = 0; i < numberOfProcesses; ++i) {
        if (ThisProcessHasOnlyBoundedPositions()) {
            ++boundedProcessesCounter;
        }
    }
    return boundedProcessesCounter == numberOfProcesses;
}

double GetMaxDifferenceFromArray(double* arrayWithMaxDifferences, int numberOfProcesses) {
    double maxDifference = DBL_MIN;
    for (int i = 0; i < numberOfProcesses; ++i) {
        maxDifference = std::max(maxDifference, arrayWithMaxDifferences[i]);
    }
    return maxDifference;
}

bool CompareResult(double* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
    if (ThisProcessHasOnlyBoundedPositions()) {
        return true;
    }

    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double maxDifference = DBL_MIN;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int position = CalculatePosition(i, j, k);
                double checkValue = GetCheckValue(i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                maxDifference = std::max(maxDifference, (double) (fabs(partOfGridWithPoints[position] - checkValue)));
//                if (currentRank == 0) {
//                    std::cout << "real pos: " << partOfGridWithPoints[position] << " expected: " << checkValue << "\n";
//                }
            }
        }
    }

    std::cout << "MAX DIFFERENCE: " << maxDifference << " FROM PROCESS: \"" << currentRank << "\"\n";
    return (maxDifference > DBL_MIN && maxDifference < CHECK_PARAMETER);
}

void IterativeProcessOfJacobiAlgorithm(double* partOfGridWithPoints, int* countOfPlanes, int currentRank, int numberOfProcesses) {
    if (DoAllProcessesHaveOnlyBoundedPositions(numberOfProcesses)) {
        if (currentRank == 0) {
            std::cout << "ONLY PROCESSES WITH BOUNDED BLOCKS!\n";
        }
        return;
    }

    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double* currentMaxDifference = new double[1];
    currentMaxDifference[0] = DBL_MIN;

    double* arrayWithMaxDifferences = new double[numberOfProcesses];
    for (int i = 0; i < numberOfProcesses; ++i) {
        arrayWithMaxDifferences[i] = DBL_MIN;
    }

    double* currentValue = new double[1];
    double* bottomValue = new double[1];
    double* topValue = new double[1];

    int numberOfPlanesXY = countOfPlanes[currentRank];
    while (true) {
        double maxDifference = DBL_MIN;
        for (int i = 1; i < numberOfPlanesXY - 1; ++i) {
            for (int j = 1; j < NUMBER_OF_POINTS_Y - 1; ++j) {
                for (int k = 1; k < NUMBER_OF_POINTS_X - 1; ++k) {
                    int position = CalculatePosition(i, j, k);
                    int handledBlocks = 0;
                    for (int count = 0; count < currentRank; ++count) {
                        handledBlocks += countOfPlanes[count];
                    }
                    double previousValue = partOfGridWithPoints[position];
                    double leftMultiplier = FindLeftMultiplier(distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                    double rightMultiplier = FindRightMultiplier(partOfGridWithPoints, i, j, k, distanceBetweenCoordinateX,
                                                                distanceBetweenCoordinateY, distanceBetweenCoordinateZ,
                                                                countOfPlanes, currentValue, bottomValue, topValue,
                                                                currentRank, numberOfProcesses);
                    partOfGridWithPoints[position] = leftMultiplier * rightMultiplier;
                    maxDifference = std::max(maxDifference, fabs(partOfGridWithPoints[position] - previousValue));
                    currentMaxDifference[0] = maxDifference;
                }
            }
        }
        MPI_Allgather(currentMaxDifference, 1, MPI_FLOAT, arrayWithMaxDifferences, 1, MPI_FLOAT, MPI_COMM_WORLD);
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

    delete[] bottomValue;
    delete[] topValue;
    delete[] currentValue;
    delete[] currentMaxDifference;
    delete[] arrayWithMaxDifferences;
}

void DebugPrintOfGrid(double* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    int position = 0;
    for (int i = 0; i < numberOfPointsZ; ++i) {
        for (int j = 0; j < numberOfPointsY; ++j) {
            for (int k = 0; k < numberOfPointsX; ++k) {
                position = CalculatePosition(i, j, k);
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
