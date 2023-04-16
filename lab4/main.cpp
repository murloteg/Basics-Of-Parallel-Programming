#include <iostream>
#include <cfloat>
#include <mpi.h>

#define CHECK_PARAMETER 10
#define EPSILON 0.00000001
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 150,
    NUMBER_OF_POINTS_Y = 200,
    NUMBER_OF_POINTS_Z = 20
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

void SetInitialApproximationToPartOfGrid(double* partOfGridWithPoints, int numberOfPlanesXY) {
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
void SetBoundaryConditionals(double* partOfGridWithPoints, int* countOfPlanes, int currentRank) {
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

double CalculateDifferenceBetweenPreviousValueAndNewValue(double previousValue, double newValue) {
    return std::abs(newValue - previousValue);
}

double FindLeftMultiplier(double distanceBetweenCoordinateX, double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ) {
    return 1 / (2 / (distanceBetweenCoordinateX * distanceBetweenCoordinateX) + 2 / (distanceBetweenCoordinateY * distanceBetweenCoordinateY)
                + 2 / (distanceBetweenCoordinateZ * distanceBetweenCoordinateZ) + EQUATION_PARAMETER);
}

bool IsThisPointTopNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
    if (IsBoundaryPositionInXYPlane(indexK, indexJ, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y)) {
        return false;
    }

    int position = indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK;
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int firstIndex = handledBlocks * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y;
    return (position >= firstIndex) && (position < firstIndex + NUMBER_OF_POINTS_Y);
}

bool IsThisPointBottomNeighbor(int indexI, int indexJ, int indexK, int* countOfPlanes, int currentRank) {
    if (IsBoundaryPositionInXYPlane(indexK, indexJ, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y)) {
        return false;
    }

    int position = indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK;
    int handledBlocks = 0;
    for (int count = 0; count < currentRank; ++count) {
        handledBlocks += countOfPlanes[count];
    }

    int lastIndex = (handledBlocks + countOfPlanes[currentRank]) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y - 1;
    return (position > lastIndex - NUMBER_OF_POINTS_Y) && (position <= lastIndex);
}

void SendBottomBoundaryPoint(double* sendValue, int currentRank, int numberOfProcesses, MPI_Request* request) {
    if (currentRank != numberOfProcesses - 1) {
        int tag = (currentRank + 1) * 100;
        MPI_Isend(sendValue, 1, MPI_DOUBLE, currentRank + 1, tag, MPI_COMM_WORLD, request);
    }
}

void SendTopBoundaryPoint(double* sendValue, int currentRank, MPI_Request* request) {
    if (currentRank != 0) {
        int tag = (currentRank - 1) * 100;
        MPI_Isend(sendValue, 1, MPI_DOUBLE, currentRank - 1, tag, MPI_COMM_WORLD, request);
    }
}

/* Receive bottom boundary point FROM previous process TO current process! */
void ReceiveBottomBoundaryPoint(double* receivedValue, int currentRank, MPI_Request* request) {
    if (currentRank != 0) {
        int tag = (currentRank - 1) * 100;
        MPI_Irecv(receivedValue, 1, MPI_DOUBLE, currentRank - 1, tag, MPI_COMM_WORLD, request);
    }
}

/* Receive top boundary point FROM next process TO current process! */
void ReceiveTopBoundaryPoint(double* receivedValue, int currentRank, int numberOfProcesses, MPI_Request* request) {
    if (currentRank != numberOfProcesses - 1) {
        int tag = (currentRank + 1) * 100;
        MPI_Irecv(receivedValue, 1, MPI_DOUBLE, currentRank + 1, tag, MPI_COMM_WORLD, request);
    }
}

double FindRightMultiplier(double* gridOfPoints, int indexI, int indexJ, int indexK, double distanceBetweenCoordinateX,
                           double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ,
                           int* countOfPlanes, int currentRank, int numberOfProcesses) {
    MPI_Request bottomNeighborRequest;
    MPI_Request topNeighborRequest;

    double* currentValue = new double[1];
    currentValue[0] = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];

    /* bottom --> for CURRENT process; top --> for CURRENT process! */
    double* bottomValue = new double[1];
    double* topValue = new double[1];
    if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank) &&
        IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        SendBottomBoundaryPoint(currentValue, currentRank, numberOfProcesses, &topNeighborRequest);
        SendTopBoundaryPoint(currentValue, currentRank, &bottomNeighborRequest);

        ReceiveBottomBoundaryPoint(topValue, currentRank, &topNeighborRequest);
        ReceiveTopBoundaryPoint(bottomValue, currentRank, numberOfProcesses, &bottomNeighborRequest);
    } else if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        SendBottomBoundaryPoint(currentValue, currentRank, numberOfProcesses, &topNeighborRequest);

        topValue[0] = gridOfPoints[(indexI - 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];
        ReceiveTopBoundaryPoint(bottomValue, currentRank, numberOfProcesses, &bottomNeighborRequest);
    } else if (IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        SendTopBoundaryPoint(currentValue, currentRank, &bottomNeighborRequest);

        bottomValue[0] = gridOfPoints[(indexI + 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];
        ReceiveBottomBoundaryPoint(topValue, currentRank, &topNeighborRequest);
    } else {
        bottomValue[0] = gridOfPoints[(indexI + 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];
        topValue[0] = gridOfPoints[(indexI - 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];
    }

    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI, distanceBetweenCoordinateZ);

    double firstNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ + 1) * NUMBER_OF_POINTS_X + indexK];
    double secondNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ - 1) * NUMBER_OF_POINTS_X + indexK];

    double firstNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK + 1];
    double secondNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK - 1];

    double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);

    if (IsThisPointBottomNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&topNeighborRequest, MPI_STATUS_IGNORE);
    }
    if (IsThisPointTopNeighbor(indexI, indexJ, indexK, countOfPlanes, currentRank)) {
        MPI_Wait(&bottomNeighborRequest, MPI_STATUS_IGNORE);
    }

    double firstNeighborZ = bottomValue[0];
    double secondNeighborZ = topValue[0];

    delete[] bottomValue;
    delete[] topValue;
    delete[] currentValue;

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

double GetMinDifferenceFromArray(double* arrayWithMaxDifferences, int numberOfProcesses) {
    double minDifference = DBL_MAX;
    for (int i = 0; i < numberOfProcesses; ++i) {
        minDifference = std::min(minDifference, arrayWithMaxDifferences[i]);
    }
    return minDifference;
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
                int position = i * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + j * NUMBER_OF_POINTS_X + k;
                double checkValue = GetCheckValue(i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                maxDifference = std::max(maxDifference, std::abs(partOfGridWithPoints[position] - checkValue));
            }
        }
    }

    std::cout << "MAX DIFFERENCE: " << maxDifference << " FROM PROCESS: " << currentRank << "\n";
    return (maxDifference > DBL_MIN && maxDifference < CHECK_PARAMETER);
}

void IterativeProcessOfJacobiAlgorithm(double* partOfGridWithPoints, int* countOfPlanes, int currentRank, int numberOfProcesses) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double* currentMaxDifference = new double[1];
    currentMaxDifference[0] = DBL_MAX;

    double* arrayWithMaxDifferences = new double[numberOfProcesses];
    for (int i = 0; i < numberOfProcesses; ++i) {
        arrayWithMaxDifferences[i] = DBL_MAX;
    }

    int numberOfPlanesXY = countOfPlanes[currentRank];
    double maxDifference = DBL_MAX;
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
                        double previousValue = partOfGridWithPoints[position];
                        partOfGridWithPoints[position] = FindLeftMultiplier(distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ) *
                        FindRightMultiplier(partOfGridWithPoints, i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ, countOfPlanes, currentRank, numberOfProcesses);
                        maxDifference = std::min(maxDifference, CalculateDifferenceBetweenPreviousValueAndNewValue(previousValue, partOfGridWithPoints[position]));
                        currentMaxDifference[0] = maxDifference;
                    }
                }
            }
        }
        MPI_Allgather(currentMaxDifference, 1, MPI_DOUBLE, arrayWithMaxDifferences, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if (CompareResult(partOfGridWithPoints, countOfPlanes, currentRank)) {
        std::cout << "CORRECT\n";
    } else {
        std::cout << "INCORRECT!\n";
    }

    delete[] currentMaxDifference;
    delete[] arrayWithMaxDifferences;
}

void DebugPrintOfGrid(double* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
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
        ++countOfPlanes[counter % numberOfProcesses];
        ++counter;
    }
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
    FillCountOfPlanesArray(countOfPlanes, numberOfProcesses);
    if (currentRank == 0) {
        for (int i = 0; i < numberOfProcesses; ++i) {
            std::cout << countOfPlanes[i] << " ";
        }
        std::cout << "\n";
    }

    double* partOfGridWithPoints = CreatePartOfGridWithPoints(countOfPlanes, currentRank);
//    DebugPrintOfGrid(partOfGridWithPoints, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, countOfPlanes[currentRank]); // DEBUG
    IterativeProcessOfJacobiAlgorithm(partOfGridWithPoints, countOfPlanes, currentRank, numberOfProcesses);
//    DebugPrintOfGrid(partOfGridWithPoints, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, countOfPlanes[currentRank]); // DEBUG
    CleanUp(partOfGridWithPoints, countOfPlanes);

    MPI_Finalize();

    return 0;
}
