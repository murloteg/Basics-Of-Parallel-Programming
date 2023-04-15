#include <iostream>
#include <cfloat>

#define CHECK_PARAMETER 0.01
#define EPSILON 0.00000001
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 7,
    NUMBER_OF_POINTS_Y = 4,
    NUMBER_OF_POINTS_Z = 3
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

bool isBoundaryPositionInXYPlane(int x, int y, int numberOfPointsX, int numberOfPointsY) {
    return (x == 0 || (x == numberOfPointsX - 1) || y == 0 || (y == numberOfPointsY - 1));
}

bool isBoundaryPositionInXZPlane(int x, int z, int numberOfPointsX, int numberOfPointsZ) {
    return (x == 0 || (x == numberOfPointsX - 1) || z == 0 || (z == numberOfPointsZ - 1));
}

bool isBoundaryPositionInYZPlane(int y, int z, int numberOfPointsY, int numberOfPointsZ) {
    return (y == 0 || (y == numberOfPointsY - 1) || z == 0 || (z == numberOfPointsZ - 1));
}

bool isBoundaryPosition(int x, int y, int z, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    bool isBoundaryPosition = isBoundaryPositionInXYPlane(x, y, numberOfPointsX, numberOfPointsY) ||
            isBoundaryPositionInXZPlane(x, z, numberOfPointsX, numberOfPointsZ) ||
            isBoundaryPositionInYZPlane(y, z, numberOfPointsY, numberOfPointsZ);
    return isBoundaryPosition;
}

void SetInitialApproximationToGrid(double* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    int position = 0;
    for (int i = 0; i < numberOfPointsZ; ++i) {
        for (int j = 0; j < numberOfPointsY; ++j) {
            for (int k = 0; k < numberOfPointsX; ++k) {
                position = i * numberOfPointsX * numberOfPointsY + j * numberOfPointsX + k;
                gridOfPoints[position] = INITIAL_APPROXIMATION;
            }
        }
    }
}

double GetCurrentCoordinateValue(int startPointCoordinate, int index, double distanceBetweenPoints) {
    return startPointCoordinate + index * distanceBetweenPoints;
}

/* position formula: (i * N_y * N_x + j * N_x + k) */
void SetBoundaryConditionals(double* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    int position = 0;
    for (int i = 0; i < numberOfPointsZ; ++i) {
        for (int j = 0; j < numberOfPointsY; ++j) {
            for (int k = 0; k < numberOfPointsX; ++k) {
                position = i * numberOfPointsX * numberOfPointsY + j * numberOfPointsX + k;
                if (isBoundaryPosition(k, j, i, numberOfPointsX, numberOfPointsY, numberOfPointsZ)) {
                    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
                    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
                    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

                    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, k, distanceBetweenCoordinateX);
                    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, j, distanceBetweenCoordinateY);
                    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, i, distanceBetweenCoordinateZ);
                    gridOfPoints[position] = CalculateFunctionValue(localX, localY, localZ);
                }
            }
        }
    }
}

double* CreateGridOfPoints(int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    double* gridOfPoints = new double[numberOfPointsX * numberOfPointsY * numberOfPointsZ];
    SetInitialApproximationToGrid(gridOfPoints, numberOfPointsX, numberOfPointsY, numberOfPointsZ);
    SetBoundaryConditionals(gridOfPoints, numberOfPointsX, numberOfPointsY, numberOfPointsZ);
    return gridOfPoints;
}

double CalculateDifferenceBetweenPreviousValueAndNewValue(double previousValue, double newValue) {
    return std::abs(newValue - previousValue);
}

double FindLeftMultiplier(double distanceBetweenCoordinateX, double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ) {
    return 1 / (2 / (distanceBetweenCoordinateX * distanceBetweenCoordinateX) + 2 / (distanceBetweenCoordinateY * distanceBetweenCoordinateY)
                + 2 / (distanceBetweenCoordinateZ * distanceBetweenCoordinateZ) + EQUATION_PARAMETER);
}

double FindRightMultiplier(double* gridOfPoints, int indexI, int indexJ, int indexK, double distanceBetweenCoordinateX,
                           double distanceBetweenCoordinateY, double distanceBetweenCoordinateZ) {
    double localX = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexK, distanceBetweenCoordinateX);
    double localY = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexJ, distanceBetweenCoordinateY);
    double localZ = GetCurrentCoordinateValue(AREA_START_COORDINATE, indexI, distanceBetweenCoordinateZ);

    double firstNeighborZ = gridOfPoints[(indexI + 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];
    double secondNeighborZ = gridOfPoints[(indexI - 1) * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK];

    double firstNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ + 1) * NUMBER_OF_POINTS_X + indexK];
    double secondNeighborY = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + (indexJ - 1) * NUMBER_OF_POINTS_X + indexK];

    double firstNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK + 1];
    double secondNeighborX = gridOfPoints[indexI * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + indexJ * NUMBER_OF_POINTS_X + indexK - 1];

    double rightPartValue = CalculateRightPartOfEquationValue(localX, localY, localZ);
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

bool CompareResult(double* gridOfPoints) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double maxDifference = DBL_MIN;
    for (int i = 0; i < NUMBER_OF_POINTS_Z; ++i) {
        for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
            for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                int position = i * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + j * NUMBER_OF_POINTS_X + k;
                double checkValue = GetCheckValue(i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                maxDifference = std::max(maxDifference, std::abs(gridOfPoints[position] - checkValue));
            }
        }
    }
    return (maxDifference > DBL_MIN && maxDifference < CHECK_PARAMETER);
}

void IterativeProcessOfJacobiAlgorithm(double* gridOfPoints) {
    double distanceBetweenCoordinateX = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_X);
    double distanceBetweenCoordinateY = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Y);
    double distanceBetweenCoordinateZ = CalculateDistanceBetweenPoints(NUMBER_OF_POINTS_Z);

    double maxDifference = DBL_MAX;
    while (maxDifference >= EPSILON) {
        for (int i = 0; i < NUMBER_OF_POINTS_Z; ++i) {
            for (int j = 0; j < NUMBER_OF_POINTS_Y; ++j) {
                for (int k = 0; k < NUMBER_OF_POINTS_X; ++k) {
                    int position = i * NUMBER_OF_POINTS_X * NUMBER_OF_POINTS_Y + j * NUMBER_OF_POINTS_X + k;
                    if (!isBoundaryPosition(k, j, i, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z)) {
                        double previousValue = gridOfPoints[position];
                        gridOfPoints[position] = FindLeftMultiplier(distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ)
                                * FindRightMultiplier(gridOfPoints, i, j, k, distanceBetweenCoordinateX, distanceBetweenCoordinateY, distanceBetweenCoordinateZ);
                        maxDifference = std::min(maxDifference, CalculateDifferenceBetweenPreviousValueAndNewValue(previousValue, gridOfPoints[position]));
                    }
                }
            }
        }
    }
    if (CompareResult(gridOfPoints)) {
        std::cout << "CORRECT\n";
    } else {
        std::cout << "INCORRECT!\n";
    }
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

void CleanUp(double* gridOfPoints) {
    delete[] gridOfPoints;
}

int main() {
    double* gridOfPoints = CreateGridOfPoints(NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z);
    IterativeProcessOfJacobiAlgorithm(gridOfPoints);
    CleanUp(gridOfPoints);

    return 0;
}
