#include <iostream>
#include <cmath>
#include <mpi.h>
#include <cfloat>

#define CHECK_PARAMETER 0.5
#define EPSILON 1e-8
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 303,
    NUMBER_OF_POINTS_Y = 451,
    NUMBER_OF_POINTS_Z = 781
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
                    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, i + handledBlocks, distanceBetweenCoordinateZ);
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

//bool IsThisPointTopNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
//    int position = GetGridPosition(indexI, indexJ, indexK);
//    int handledBlocks = 0;
//    for (int count = 0; count < currentRank; ++count) {
//        handledBlocks += countOfPlanes[count];
//    }
//
//    int firstIndex = handledBlocks * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
//    return (position >= firstIndex) && (position < firstIndex + NUMBER_OF_POINTS_Y * NUMBER_OF_POINTS_X);
//}
//
//bool IsThisPointBottomNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
//    int position = GetGridPosition(indexI, indexJ, indexK);
//    int handledBlocks = 0;
//    for (int count = 0; count < currentRank; ++count) {
//        handledBlocks += countOfPlanes[count];
//    }
//
//    int lastIndex = (handledBlocks + countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
//    return (position > lastIndex - NUMBER_OF_POINTS_Y * NUMBER_OF_POINTS_X) && (position <= lastIndex);
//}

bool IsPositionBottomNeighbor(int position, int* countOfPlanes, int currentRank) {
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int lastIndex = (handledBlocks + countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
    return (position > lastIndex - NUMBER_OF_POINTS_Y * NUMBER_OF_POINTS_X) && (position <= lastIndex);
}

bool IsPositionTopNeighbor(int position, int* countOfPlanes, int currentRank) {
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int firstIndex = handledBlocks * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
    return (position >= firstIndex) && (position < firstIndex + NUMBER_OF_POINTS_Y * NUMBER_OF_POINTS_X);
}

bool IsFirstRequestFromPlane(int position, int handledBlocks, int currentRank) {
    if (currentRank == 0) {
        return position == NUMBER_OF_POINTS_X + 2; /* avoid border position */
    }
    return position == (handledBlocks * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + 1);
}

//void SendBoundaryPoint(double* sendValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
//    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
//        MPI_Isend(sendValue, 1, MPI_DOUBLE, destinationRank, 1, MPI_COMM_WORLD, request);
//    }
//}
//
//void ReceiveBoundaryPoint(double* receivedValue, int destinationRank, int numberOfProcesses, MPI_Request* request) {
//    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
//        MPI_Irecv(receivedValue, 1, MPI_DOUBLE, destinationRank, MPI_ANY_TAG, MPI_COMM_WORLD, request);
//    }
//}

void SendBoundaryPlane(double* sendPlane, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        MPI_Isend(sendPlane, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, destinationRank, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_COMM_WORLD, request);
    }
}

void ReceiveBoundaryPlane(double* receivedPlane, int destinationRank, int numberOfProcesses, MPI_Request* request) {
    if (destinationRank >= 0 && destinationRank < numberOfProcesses) {
        MPI_Irecv(receivedPlane, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, destinationRank, MPI_ANY_TAG, MPI_COMM_WORLD, request);
    }
}

double FindRightMultiplier(double* gridOfPoints, int indexI, int indexJ, int indexK, double distanceBetweenCoordinateX,
                           double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ, int* countOfPlanes,
                           int handledBlocks, int currentRank, int numberOfProcesses, double* topBorderPlane,
                           double* bottomBorderPlane, double* receivedTopBorderPlane, double* receivedBottomBorderPlane) {
    MPI_Request bottomNeighborRequest;
    MPI_Request topNeighborRequest;

    /* bottom --> for CURRENT process; top --> for CURRENT process! */

    int position = GetGridPosition(indexI + handledBlocks, indexJ, indexK);
    if (IsFirstRequestFromPlane(position, handledBlocks, currentRank)) {
        if (IsPositionTopNeighbor(position, countOfPlanes, currentRank) && IsPositionBottomNeighbor(position, countOfPlanes, currentRank)) {
            SendBoundaryPlane(topBorderPlane, currentRank - 1, numberOfProcesses, &topNeighborRequest);
            SendBoundaryPlane(bottomBorderPlane, currentRank + 1, numberOfProcesses, &bottomNeighborRequest);

            ReceiveBoundaryPlane(receivedBottomBorderPlane, currentRank - 1, numberOfProcesses, &bottomNeighborRequest);
            ReceiveBoundaryPlane(receivedTopBorderPlane, currentRank + 1, numberOfProcesses, &topNeighborRequest);
        } else if (IsPositionTopNeighbor(position, countOfPlanes, currentRank)) {
            SendBoundaryPlane(bottomBorderPlane, currentRank + 1, numberOfProcesses, &bottomNeighborRequest);
            ReceiveBoundaryPlane(receivedTopBorderPlane, currentRank + 1, numberOfProcesses, &topNeighborRequest);
        } else if (IsPositionBottomNeighbor(position, countOfPlanes, currentRank)) {
            SendBoundaryPlane(topBorderPlane, currentRank - 1, numberOfProcesses, &topNeighborRequest);
            ReceiveBoundaryPlane(receivedBottomBorderPlane, currentRank - 1, numberOfProcesses, &bottomNeighborRequest);
        }
    }

    double firstNeighborY = gridOfPoints[GetGridPosition(indexI, indexJ + 1, indexK)];
    double secondNeighborY = gridOfPoints[GetGridPosition(indexI, indexJ - 1, indexK)];

    double firstNeighborX = gridOfPoints[GetGridPosition(indexI, indexJ, indexK + 1)];
    double secondNeighborX = gridOfPoints[GetGridPosition(indexI, indexJ, indexK - 1)];

    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI + handledBlocks, distanceBetweenCoordinateZ);
    double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

    /* wait all sent boundary planes from another processes */
    if (IsFirstRequestFromPlane(position, handledBlocks, currentRank) && IsPositionTopNeighbor(position, countOfPlanes, currentRank)) {
        MPI_Wait(&topNeighborRequest, MPI_STATUS_IGNORE);
    }
    if (IsFirstRequestFromPlane(position, handledBlocks, currentRank) && IsPositionBottomNeighbor(position, countOfPlanes, currentRank)) {
        MPI_Wait(&bottomNeighborRequest, MPI_STATUS_IGNORE);
    }

    double firstNeighborZ = topBorderPlane[indexJ * NUMBER_OF_POINTS_X + indexK];
    double secondNeighborZ = bottomBorderPlane[indexJ * NUMBER_OF_POINTS_X + indexK];

    double sqrDistanceX = distanceBetweenCoordinateX * distanceBetweenCoordinateX;
    double sqrDistanceY = distanceBetweenCoordinateY * distanceBetweenCoordinateY;
    double sqrDistanceZ = distanceBetweenCoordinateZ * distanceBetweenCoordinateZ;

    double sumX = firstNeighborX + secondNeighborX;
    double sumY = firstNeighborY + secondNeighborY;
    double sumZ = firstNeighborZ + secondNeighborZ;

    return (sumZ / sqrDistanceZ + sumY / sqrDistanceY + sumX / sqrDistanceX) - rightPartValue;
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
    }
    return maxDifference;
}

bool CompareResult(double* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double maxDifference = 0;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int position = GetGridPosition(i, j, k);
                int handledBlocks = 0;
                for (int count = 0; count < currentRank; ++count) {
                    handledBlocks += countOfPlanes[count];
                }
                double checkValue = GetCheckValue(i + handledBlocks, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                maxDifference = std::max(maxDifference, fabs(partOfGridWithPoints[position] - checkValue));
            }
        }
    }

    std::cout << "MAX DIFFERENCE: " << maxDifference << " FROM PROCESS: \"" << currentRank << "\"\n";
    return maxDifference < CHECK_PARAMETER;
}

void DebugPrintOfGrid(double* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    int position = 0;
    for (int i = 0; i < numberOfPointsZ; ++i) {
        std::cout << "plane " << i << "\n";
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

void CopyBorderPlane(double* borderPlane, double* partOfGridWithPoints, int indexI) {
    for (int indexJ = 0; indexJ < NUMBER_OF_POINTS_Y; ++indexJ) {
        for (int indexK = 0; indexK < NUMBER_OF_POINTS_X; ++indexK) {
            borderPlane[indexJ * NUMBER_OF_POINTS_X + indexK] = partOfGridWithPoints[GetGridPosition(indexI, indexJ, indexK)];
        }
    }
}

void UpdateBorderPlane(double* partOfGridWithPoints, double* borderPlane, int numberOfPlanesXY, int indexI) {
    if (indexI == 0 || indexI == numberOfPlanesXY - 1) {
        CopyBorderPlane(borderPlane, partOfGridWithPoints, indexI);
    }
}

void IterativeProcessOfJacobiAlgorithm(double* partOfGridWithPoints, int* countOfPlanes, int currentRank, int numberOfProcesses) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

//    double* arrayWithMaxDifferences = new double[numberOfProcesses];
//    for (int i = 0; i < numberOfProcesses; ++i) {
//        arrayWithMaxDifferences[i] = DBL_MIN;
//    }

    double* topBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];
    double* bottomBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];

    double* receivedTopBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];
    double* receivedBottomBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];

    int numberOfPlanesXY = countOfPlanes[currentRank];
//    MPI_Request allProcessRequest;
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
                    double previousValue;
                    if (!IsBoundaryPosition(k, j, i + handledBlocks, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
                        previousValue= partOfGridWithPoints[position];
                        double leftMultiplier = FindLeftMultiplier(distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                        double rightMultiplier = FindRightMultiplier(partOfGridWithPoints, i, j, k, distanceBetweenCoordinateX,
                                                                     distanceBetweenCoordinateY, distanceBetweenCoordinateZ,
                                                                     countOfPlanes, handledBlocks, currentRank, numberOfProcesses,
                                                                     topBorderPlane, bottomBorderPlane, receivedTopBorderPlane, receivedBottomBorderPlane);
                        partOfGridWithPoints[position] = leftMultiplier * rightMultiplier;
                        maxDifference = std::max(maxDifference, fabs(partOfGridWithPoints[position] - previousValue));
                    }
                }
            }
            UpdateBorderPlane(partOfGridWithPoints, topBorderPlane, numberOfPlanesXY, (i - 1));
            UpdateBorderPlane(partOfGridWithPoints, bottomBorderPlane, numberOfPlanesXY, (i + 1));
        }
//        MPI_Iallgather(&maxDifference, 1, MPI_DOUBLE, arrayWithMaxDifferences, 1, MPI_DOUBLE, MPI_COMM_WORLD, &allProcessRequest);
//        MPI_Wait(&allProcessRequest, MPI_STATUSES_IGNORE);
//        if (currentRank == 0) {
//            DebugPrintOfGrid(partOfGridWithPoints, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z);
//        }

//        MPI_Allgather(&maxDifference, 1, MPI_DOUBLE, arrayWithMaxDifferences, 1, MPI_DOUBLE, MPI_COMM_WORLD);
//        std::cout << "rank: " << currentRank << " max diff: " << GetMaxDifferenceFromArray(arrayWithMaxDifferences, numberOfProcesses) << "\n";
//        if (GetMaxDifferenceFromArray(arrayWithMaxDifferences, numberOfProcesses) < EPSILON) {
//            break;
//        }

        double maxDiffFromAllProcesses = DBL_MIN;
        MPI_Allreduce(&maxDifference, &maxDiffFromAllProcesses, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (currentRank == 0) {
            std::cout << "max diff: " << maxDiffFromAllProcesses << "\n";
        }
        if (maxDiffFromAllProcesses < EPSILON) {
            break;
        }
    }
    if (CompareResult(partOfGridWithPoints, countOfPlanes, currentRank)) {
        std::cout << "CORRECT\n";
    } else {
        std::cout << "INCORRECT!\n";
    }

//    delete[] arrayWithMaxDifferences;
    delete[] topBorderPlane;
    delete[] bottomBorderPlane;
    delete[] receivedTopBorderPlane;
    delete[] receivedBottomBorderPlane;
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
