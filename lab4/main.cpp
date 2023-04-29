#include <iostream>
#include <cmath>
#include <cfloat>
#include <mpi.h>

#define EPSILON 1e-8
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 319,
    NUMBER_OF_POINTS_Y = 325,
    NUMBER_OF_POINTS_Z = 375
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

void SetInitialApproximationToPartOfGrid(double* partOfGrid, int numberOfPlanesXY) {
    int position = 0;
    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                position = Get3DGridPosition(i, j, k);
                partOfGrid[position] = INITIAL_APPROXIMATION;
            }
        }
    }
}

/* position formula: (i * N_y * N_x + j * N_x + k) */
void SetBoundaryConditionals(double* partOfGrid, int* countOfPlanes, int currentRank) {
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
                    double distanceCoordX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
                    double distanceCoordY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
                    double distanceCoordZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

                    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, k, distanceCoordX);
                    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, j, distanceCoordY);
                    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, i + handledBlocks, distanceCoordZ);
                    partOfGrid[position] = CalculateFunctionValue(localX, localY, localZ);
                }
            }
        }
    }
}

double* CreatePartOfGridWithPoints(int* countOfPlanes, int currentRank) {
    int numberOfPlanesXY = countOfPlanes[currentRank];
    double* partOfGrid = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y * numberOfPlanesXY];
    SetInitialApproximationToPartOfGrid(partOfGrid, numberOfPlanesXY);
    SetBoundaryConditionals(partOfGrid, countOfPlanes, currentRank);
    return partOfGrid;
}

double FindLeftMultiplier(double distanceCoordX, double distanceCoordY, double distanceCoordZ) {
    double partX = 2 / (distanceCoordX * distanceCoordX);
    double partY = 2 / (distanceCoordY * distanceCoordY);
    double partZ = 2 / (distanceCoordZ * distanceCoordZ);
    return 1 / (partX + partY + partZ + EQUATION_PARAMETER);
}

/* bottom boundary plane for PREVIOUS process; top boundary plane for NEXT process */
void MakeRequestsToNeighbors(double* partOfGrid, double* topBorderPlane, double* bottomBorderPlane,
                          int numberOfPlanesXY, int currentRank, int numberOfProcesses,
                          MPI_Request* topNeighborRequest, MPI_Request* bottomNeighborRequest) {
    if (currentRank != 0) {
        MPI_Isend(partOfGrid, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, currentRank - 1,
                  currentRank, MPI_COMM_WORLD, topNeighborRequest);

        MPI_Irecv(bottomBorderPlane, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, currentRank - 1, MPI_ANY_TAG,
                  MPI_COMM_WORLD, bottomNeighborRequest);
    }
    if (currentRank != numberOfProcesses - 1) {
        MPI_Isend(partOfGrid + (numberOfPlanesXY - 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y,
                  MPI_DOUBLE, currentRank + 1, currentRank, MPI_COMM_WORLD, bottomNeighborRequest);

        MPI_Irecv(topBorderPlane, NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y, MPI_DOUBLE, currentRank + 1, MPI_ANY_TAG,
                  MPI_COMM_WORLD, topNeighborRequest);
    }
}

void WaitDataFromNeighbors(int currentRank, int numberOfProcesses, MPI_Request* topNeighborRequest,
                           MPI_Request* bottomNeighborRequest) {
    if (currentRank != 0) {
        MPI_Wait(bottomNeighborRequest, MPI_STATUSES_IGNORE);
    }
    if (currentRank != numberOfProcesses - 1) {
        MPI_Wait(topNeighborRequest, MPI_STATUSES_IGNORE);
    }
}

double GetCheckValue(int indexI, int indexJ, int indexK, double distanceCoordX,
                     double distanceCoordY, double distanceCoordZ) {
    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceCoordX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceCoordY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI, distanceCoordZ);
    return CalculateFunctionValue(localX, localY, localZ);
}

void CompareResult(double* partOfGrid, int* countOfPlanes, int currentRank) {
    double distanceCoordX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceCoordY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceCoordZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double maxDifference = DBL_MIN;
    int numberOfPlanesXY = countOfPlanes[currentRank];

    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    for (int i = 0; i < numberOfPlanesXY; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int position = Get3DGridPosition(i, j, k);
                double checkValue = GetCheckValue(i + handledBlocks, j, k, distanceCoordX, distanceCoordY, distanceCoordZ);
                maxDifference = std::max(maxDifference, fabs(partOfGrid[position] - checkValue));
            }
        }
    }
    std::cout << "MAX DIFFERENCE: " << maxDifference << " FROM PROCESS: \"" << currentRank << "\"\n";
}

void CalculateInnerPart(double* partOfGrid, int numberOfPlanesXY, int handledBlocks,
                        double distanceCoordX, double distanceCoordY, double distanceCoordZ,
                        double leftMultiplier, double& maxDifference) {
    double sqrDistanceX = distanceCoordX * distanceCoordX;
    double sqrDistanceY = distanceCoordY * distanceCoordY;
    double sqrDistanceZ = distanceCoordZ * distanceCoordZ;

    for (int i = 1; i < numberOfPlanesXY - 1; ++i) {
        for (int j = 1; j < NUMBER_OF_POINTS_Y - 1; ++j) {
            for (int k = 1; k < NUMBER_OF_POINTS_X - 1; ++k) {
                double firstNeighborY = partOfGrid[Get3DGridPosition(i, j + 1, k)];
                double secondNeighborY = partOfGrid[Get3DGridPosition(i, j - 1, k)];

                double firstNeighborX = partOfGrid[Get3DGridPosition(i, j, k + 1)];
                double secondNeighborX = partOfGrid[Get3DGridPosition(i, j, k - 1)];

                double firstNeighborZ = partOfGrid[Get3DGridPosition(i - 1, j, k)];
                double secondNeighborZ = partOfGrid[Get3DGridPosition(i + 1, j, k)];

                double sumX = firstNeighborX + secondNeighborX;
                double sumY = firstNeighborY + secondNeighborY;
                double sumZ = firstNeighborZ + secondNeighborZ;

                double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, k, distanceCoordX);
                double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, j, distanceCoordY);
                double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, i + handledBlocks, distanceCoordZ);
                double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

                double previousValue = partOfGrid[Get3DGridPosition(i, j, k)];
                partOfGrid[Get3DGridPosition(i, j, k)] = leftMultiplier *
                        ((sumZ / sqrDistanceZ + sumY / sqrDistanceY + sumX / sqrDistanceX) - rightPartValue);
                double currentValue = partOfGrid[Get3DGridPosition(i, j, k)];
                maxDifference = std::max(maxDifference, fabs(currentValue - previousValue));
            }
        }
    }
}

void CalculateBoundaryPart(double* partOfGrid, int numberOfPlanesXY, int handledBlocks,
                            double distanceCoordX, double distanceCoordY, double distanceCoordZ,
                            double leftMultiplier, double* topBorderPlane, double* bottomBorderPlane,
                            double& maxDifference, int currentRank, int numberOfProcesses) {
    double sqrDistanceX = distanceCoordX * distanceCoordX;
    double sqrDistanceY = distanceCoordY * distanceCoordY;
    double sqrDistanceZ = distanceCoordZ * distanceCoordZ;

    int firstIndexZ = 0;
    int lastIndexZ = numberOfPlanesXY - 1;

    for (int j = 1; j < NUMBER_OF_POINTS_Y - 1; ++j) {
        for (int k = 1; k < NUMBER_OF_POINTS_X - 1; ++k) {
            if (currentRank != 0) {
                double firstNeighborY = partOfGrid[Get3DGridPosition(firstIndexZ, j + 1, k)];
                double secondNeighborY = partOfGrid[Get3DGridPosition(firstIndexZ, j - 1, k)];

                double firstNeighborX = partOfGrid[Get3DGridPosition(firstIndexZ, j, k + 1)];
                double secondNeighborX = partOfGrid[Get3DGridPosition(firstIndexZ, j, k - 1)];

                double firstNeighborZ = bottomBorderPlane[Get2DGridPosition(j, k)];
                double secondNeighborZ = partOfGrid[Get3DGridPosition(firstIndexZ + 1, j, k)];

                double sumX = firstNeighborX + secondNeighborX;
                double sumY = firstNeighborY + secondNeighborY;
                double sumZ = firstNeighborZ + secondNeighborZ;

                double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, k, distanceCoordX);
                double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, j, distanceCoordY);
                double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, handledBlocks, distanceCoordZ);
                double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

                double previousValue = partOfGrid[Get3DGridPosition(firstIndexZ, j, k)];
                partOfGrid[Get3DGridPosition(firstIndexZ, j, k)] = leftMultiplier *
                                                         ((sumZ / sqrDistanceZ + sumY / sqrDistanceY + sumX / sqrDistanceX) - rightPartValue);
                double currentValue = partOfGrid[Get3DGridPosition(firstIndexZ, j, k)];
                maxDifference = std::max(maxDifference, fabs(currentValue - previousValue));
            }

            if (currentRank != numberOfProcesses - 1) {
                double firstNeighborY = partOfGrid[Get3DGridPosition(lastIndexZ, j + 1, k)];
                double secondNeighborY = partOfGrid[Get3DGridPosition(lastIndexZ, j - 1, k)];

                double firstNeighborX = partOfGrid[Get3DGridPosition(lastIndexZ, j, k + 1)];
                double secondNeighborX = partOfGrid[Get3DGridPosition(lastIndexZ, j, k - 1)];

                double firstNeighborZ = partOfGrid[Get3DGridPosition(lastIndexZ - 1, j, k)];
                double secondNeighborZ = topBorderPlane[Get2DGridPosition(j, k)];

                double sumX = firstNeighborX + secondNeighborX;
                double sumY = firstNeighborY + secondNeighborY;
                double sumZ = firstNeighborZ + secondNeighborZ;

                double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, k, distanceCoordX);
                double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, j, distanceCoordY);
                double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, handledBlocks, distanceCoordZ);
                double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

                double previousValue = partOfGrid[Get3DGridPosition(lastIndexZ, j, k)];
                partOfGrid[Get3DGridPosition(lastIndexZ, j, k)] = leftMultiplier *
                                                                   ((sumZ / sqrDistanceZ + sumY / sqrDistanceY + sumX / sqrDistanceX) - rightPartValue);
                double currentValue = partOfGrid[Get3DGridPosition(lastIndexZ, j, k)];
                maxDifference = std::max(maxDifference, fabs(currentValue - previousValue));
            }
        }
    }
}

void IterativeProcessOfJacobiAlgorithm(double* partOfGrid, int* countOfPlanes, int currentRank, int numberOfProcesses) {
    double distanceCoordX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceCoordY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceCoordZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);
    double leftMultiplier = FindLeftMultiplier(distanceCoordX, distanceCoordY, distanceCoordZ);

    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    double* topBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];
    double* bottomBorderPlane = new double[NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y];
    MPI_Request topNeighborRequest;
    MPI_Request bottomNeighborRequest;

    int numberOfPlanesXY = countOfPlanes[currentRank];
    int iterationCounter = 0;
    while (true) {
        double maxDifference = DBL_MIN;
        MakeRequestsToNeighbors(partOfGrid, topBorderPlane, bottomBorderPlane, numberOfPlanesXY, currentRank,
                             numberOfProcesses, &topNeighborRequest, &bottomNeighborRequest);

        CalculateInnerPart(partOfGrid, numberOfPlanesXY, handledBlocks, distanceCoordX, distanceCoordY, distanceCoordZ,
                           leftMultiplier, maxDifference);

        WaitDataFromNeighbors(currentRank, numberOfProcesses, &topNeighborRequest, &bottomNeighborRequest);

        CalculateBoundaryPart(partOfGrid, numberOfPlanesXY, handledBlocks, distanceCoordX, distanceCoordY, distanceCoordZ,
                              leftMultiplier, topBorderPlane, bottomBorderPlane, maxDifference, currentRank, numberOfProcesses);

        double maxDiffFromAllProcesses = DBL_MIN;
        MPI_Allreduce(&maxDifference, &maxDiffFromAllProcesses, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (currentRank == 0) {
            ++iterationCounter;
            std::cout << iterationCounter << " [Debug] Max difference: " << maxDiffFromAllProcesses << "\n";
        }
        if (maxDiffFromAllProcesses < EPSILON) {
            break;
        }
    }
    CompareResult(partOfGrid, countOfPlanes, currentRank);

    delete[] topBorderPlane;
    delete[] bottomBorderPlane;
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

void CleanUp(double* partOfGrid, int* countOfPlanes) {
    delete[] partOfGrid;
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

    double* partOfGrid = CreatePartOfGridWithPoints(countOfPlanes, currentRank);
    double start = MPI_Wtime();
    IterativeProcessOfJacobiAlgorithm(partOfGrid, countOfPlanes, currentRank, numberOfProcesses);
    double end = MPI_Wtime();
    if (currentRank == 0) {
        std::cout << "Elapsed time: " << end - start << " [sec]\n";
    }

    CleanUp(partOfGrid, countOfPlanes);

    MPI_Finalize();

    return 0;
}
