#include <iostream>

#define EPSILON = 0.00000001
enum GeneralParameters {
    AREA_START_COORDINATE = -1,
    AREA_END_COORDINATE = 1,
    EQUATION_PARAMETER = 100000,
    INITIAL_APPROXIMATION = 0,
    BOUNDARY_CONDITIONALS = 2
};

enum GridParameters {
    NUMBER_OF_POINTS_X = 4,
    NUMBER_OF_POINTS_Y = 3,
    NUMBER_OF_POINTS_Z = 5,
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

double GetCurrentCoordinateValue(int startPointCoordinate, int index, double distanceBetweenPoints) {
    return startPointCoordinate + index * distanceBetweenPoints;
}

bool isBoundaryPositionInXYPlane(int x, int y, int numberOfPointsX, int numberOfPointsY) {
    if (x == 0 || (x == numberOfPointsX - 1) || y == 0 || (y == numberOfPointsY - 1)) {
        return true;
    }
    return false;
}

bool isBoundaryPositionInXZPlane(int x, int z, int numberOfPointsX, int numberOfPointsZ) {
    if (x == 0 || (x == numberOfPointsX - 1) || z == 0 || (z == numberOfPointsZ - 1)) {
        return true;
    }
    return false;
}

bool isBoundaryPositionInYZPlane(int y, int z, int numberOfPointsY, int numberOfPointsZ) {
    if (y == 0 || (y == numberOfPointsY - 1) || z == 0 || (z == numberOfPointsZ - 1)) {
        return true;
    }
    return false;
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

/* position formula: (i * N_y * N_x + j * N_x + k) */
void SetBoundaryConditionals(double* gridOfPoints, int numberOfPointsX, int numberOfPointsY, int numberOfPointsZ) {
    int position = 0;
    for (int i = 0; i < numberOfPointsZ; ++i) {
        for (int j = 0; j < numberOfPointsY; ++j) {
            for (int k = 0; k < numberOfPointsX; ++k) {
                position = i * numberOfPointsX * numberOfPointsY + j * numberOfPointsX + k;
                if (isBoundaryPosition(k, j, i, numberOfPointsX, numberOfPointsY, numberOfPointsZ)) {
                    gridOfPoints[position] = BOUNDARY_CONDITIONALS;
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
    DebugPrintOfGrid(gridOfPoints, NUMBER_OF_POINTS_X, NUMBER_OF_POINTS_Y, NUMBER_OF_POINTS_Z);
    CleanUp(gridOfPoints);

    return 0;
}
