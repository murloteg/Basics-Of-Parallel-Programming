#include <iostream>
#include <cmath>
#include <cfloat>
#include <mpi.h>

#define CHECK_PARAMETER 1
#define EPSILON 1e-8
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 315,
    NUMBER_OF_POINTS_Y = 315,
    NUMBER_OF_POINTS_Z = 315
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
    return startPointCoordinate + index * distanceBetweenPoints;
}

int Get2DGridPosition(int indexJ, int indexK) {
    return indexJ * NUMBER_OF_POINTS_X + indexK;
}

int Get3DGridPosition(int indexI, int indexJ, int indexK) {
    return indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK;
}

void SetInitialApproximationToPartOfGrid(double* partOfGridWithPoints, int numberOfPlanesXY) {
    int position = 0;
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                position = Get3DGridPosition(i, j, k);
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
                position = Get3DGridPosition(i, j, k);
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

void InvokeCommunications(double* partOfGridWithPoints, double* topBorderPlane, double* bottomBorderPlane, int numberOfPlanesXY, int currentRank, int numberOfProcesses, MPI_Request* topNeighborRequest, MPI_Request* bottomNeighborRequest) {
    if (currentRank != 0) {
        MPI_Isend(partOfGridWithPoints, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, currentRank - 1, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_COMM_WORLD, topNeighborRequest);
        MPI_Irecv(bottomBorderPlane, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, currentRank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, bottomNeighborRequest);
    }
    if (currentRank != numberOfProcesses - 1) {
        MPI_Isend(partOfGridWithPoints + (numberOfPlanesXY - 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, currentRank + 1, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_COMM_WORLD, bottomNeighborRequest);
        MPI_Irecv(topBorderPlane, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, currentRank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, topNeighborRequest);
    }
}

double FindRightMultiplier(double* gridOfPoints, int indexI, int indexJ, int indexK, double distanceBetweenCoordinateX,
                           double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ,
                           int handledBlocks, double* topBorderPlane, double* bottomBorderPlane) {
    double firstNeighborY = gridOfPoints[Get3DGridPosition(indexI, indexJ + 1, indexK)];
    double secondNeighborY = gridOfPoints[Get3DGridPosition(indexI, indexJ - 1, indexK)];

    double firstNeighborX = gridOfPoints[Get3DGridPosition(indexI, indexJ, indexK + 1)];
    double secondNeighborX = gridOfPoints[Get3DGridPosition(indexI, indexJ, indexK - 1)];

    double firstNeighborZ = topBorderPlane[Get2DGridPosition(indexJ, indexK)];
    double secondNeighborZ = bottomBorderPlane[Get2DGridPosition(indexJ, indexK)];

    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI + handledBlocks, distanceBetweenCoordinateZ);
    double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

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

bool CompareResult(double* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double maxDifference = 0;
    int numberOfPlanesXY = countOfPlanes[currentRank];
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int position = Get3DGridPosition(i, j, k);
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

double GetMaxDifferenceFromArray(double* arrayWithMaxDifferences, int numberOfProcesses) {
    double maxDifference = DBL_MIN;
    for (int i = 0; i < numberOfProcesses; ++i) {
        maxDifference = std::max(maxDifference, arrayWithMaxDifferences[i]);
    }
    return maxDifference;
}

void IterativeProcessOfJacobiAlgorithm(double* partOfGridWithPoints, int* countOfPlanes, int currentRank, int numberOfProcesses) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);
    double leftMultiplier = FindLeftMultiplier(distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);

    double* arrayWithMaxDifferences = new double[numberOfProcesses];
    for (int i = 0; i < numberOfProcesses; ++i) {
        arrayWithMaxDifferences[i] = DBL_MIN;
    }

    double* topBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];
    double* bottomBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];

    MPI_Request topNeighborRequest;
    MPI_Request bottomNeighborRequest;

    int numberOfPlanesXY = countOfPlanes[currentRank];
    int iterationCounter = 0;
    while (true) {
        double maxDifference = DBL_MIN;
        InvokeCommunications(partOfGridWithPoints, topBorderPlane, bottomBorderPlane, numberOfPlanesXY, currentRank, numberOfProcesses, &topNeighborRequest, &bottomNeighborRequest);
        for (int i = 0; i < numberOfPlanesXY; ++i) {
            for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
                for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                    int position = Get3DGridPosition(i, j, k);
                    int handledBlocks = 0;
                    for (int count = 0; count < currentRank; ++count) {
                        handledBlocks += countOfPlanes[count];
                    }
                    double previousValue;
                    if (!IsBoundaryPosition(k, j, i + handledBlocks, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
                        previousValue = partOfGridWithPoints[position];
                        if (currentRank != 0) {
                            MPI_Wait(&topNeighborRequest, MPI_STATUS_IGNORE);
                        }
                        if (currentRank != numberOfProcesses - 1) {
                            MPI_Wait(&bottomNeighborRequest, MPI_STATUS_IGNORE);
                        }

                        double rightMultiplier = FindRightMultiplier(partOfGridWithPoints, i, j, k, distanceBetweenCoordinateX,
                                                                     distanceBetweenCoordinateY, distanceBetweenCoordinateZ,
                                                                     handledBlocks, topBorderPlane, bottomBorderPlane);
                        partOfGridWithPoints[position] = leftMultiplier * rightMultiplier;
                        maxDifference = std::max(maxDifference, fabs(partOfGridWithPoints[position] - previousValue));
                    }
                }
            }
        }

        MPI_Allgather(&maxDifference, 1, MPI_DOUBLE, arrayWithMaxDifferences, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        if (currentRank == 0) {
            ++iterationCounter;
            std::cout << iterationCounter << " [Debug] Max difference: " << GetMaxDifferenceFromArray(arrayWithMaxDifferences, numberOfProcesses) << "\n";
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

    delete[] topBorderPlane;
    delete[] bottomBorderPlane;
    delete[] arrayWithMaxDifferences;
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
