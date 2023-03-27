#include <iostream>
#include <mpi.h>

enum SizeConsts {
    NUMBER_OF_LINES_A = 3,
    NUMBER_OF_COLUMNS_A = 6, /* must be the same with next parameter! */
    NUMBER_OF_LINES_B = 2, /* must be the same with previous parameter! */
    NUMBER_OF_COLUMNS_B = 3,
    HORIZONTAL_NUMBER_OF_COMPUTERS = 2, /* first parameter of grid: p1 */
    VERTICAL_NUMBER_OF_COMPUTERS = 2 /* second parameter of grid: p2 */
};

enum Axes {
    VERTICAL_AXIS = 0,
    HORIZONTAL_AXIS = 1
};

double* GenerateMatrixA() {
    double* matrixA = new double[NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_A];
    int currentPosition = 0;
    for (int i = 0; i < NUMBER_OF_LINES_A; ++i) {
        for (int j = 0; j < NUMBER_OF_COLUMNS_A; ++j) {
            matrixA[currentPosition] = currentPosition;
            ++currentPosition;
        }
    }
    return matrixA;
}

double* GenerateMatrixB() {
    double* matrixB = new double[NUMBER_OF_LINES_B * NUMBER_OF_COLUMNS_B];
    int currentPosition = 0;
    for (int i = 0; i < NUMBER_OF_LINES_B; ++i) {
        for (int j = 0; j < NUMBER_OF_COLUMNS_B; ++j) {
            matrixB[currentPosition] = currentPosition;
            ++currentPosition;
        }
    }
    return matrixB;
}

void DebugPrint(double* matrix, int lines, int columns) {
    int currentPosition = 0;
    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < columns; ++j) {
            std::cout << matrix[currentPosition] << " ";
            ++currentPosition;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void SendMatrixAFromRootToVerticalNeighbors(double* matrixA, int matrixSize, int rank, double* receiveBufferForMatrixA,
                                            MPI_Comm COMMUNICATOR, int commSize) {
//    MPI_Cart_coords(COMMUNICATOR, newRank, 2, coords);
    int sizeOfPartOfMatrixA = matrixSize / commSize;
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(COMMUNICATOR, coords, &rootRank);
    MPI_Scatter(matrixA, sizeOfPartOfMatrixA, MPI_DOUBLE, receiveBufferForMatrixA, sizeOfPartOfMatrixA, MPI_DOUBLE, rootRank, COMMUNICATOR);
}

void SendMatrixBFromRootToHorizontalNeighbors() {

}

int main(int argc, char** argv) {
    int numberOfProcesses;
    int currentRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm MPI_CUSTOM_2D_GRID;
    MPI_Dims_create(numberOfProcesses, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &MPI_CUSTOM_2D_GRID);

    if (currentRank == 0) {
        std::cout << "DIMS: " << dims[0] << " " << dims[1] << "\n";
    }

    double* matrixA = NULL;
    double* matrixB = NULL;
    double* matrixC = NULL; /* result-matrix */

    MPI_Comm ROOT_HORIZONTAL_NEIGHBORS;
    MPI_Comm ROOT_VERTICAL_NEIGHBORS;

    int horizontal_remain_dims[2] = {0, 1};
    int vertical_remain_dims[2] = {1, 0};
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, horizontal_remain_dims, &ROOT_HORIZONTAL_NEIGHBORS);
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, vertical_remain_dims, &ROOT_VERTICAL_NEIGHBORS);

    int currentRankCoords[2] = {0, 0};
    MPI_Cart_coords(MPI_CUSTOM_2D_GRID, currentRank, 2, currentRankCoords);
    if (currentRankCoords[0] == 0 && currentRankCoords[1] == 0) {
        matrixA = GenerateMatrixA();
        matrixB = GenerateMatrixB();
    }

    int sizeOfMatrixA = NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_A;
    double* receiveBufferForMatrixA = new double[sizeOfMatrixA / dims[VERTICAL_AXIS]];
    if (currentRank % dims[HORIZONTAL_AXIS] == 0) {
        SendMatrixAFromRootToVerticalNeighbors(matrixA, sizeOfMatrixA, currentRank, receiveBufferForMatrixA, ROOT_VERTICAL_NEIGHBORS, dims[VERTICAL_AXIS]);
    }

    std::cout << "CURRENT RANK: " << currentRank << "\n";
    DebugPrint(receiveBufferForMatrixA, NUMBER_OF_LINES_A / dims[VERTICAL_AXIS], NUMBER_OF_COLUMNS_A);

    double* receiveBufferForMatrixB;
    // TODO: create MPI_Type for matrix B before Scatter.

//    DebugPrint(matrixA, NUMBER_OF_LINES_A, NUMBER_OF_COLUMNS_A);
//    DebugPrint(matrixB, NUMBER_OF_LINES_B, NUMBER_OF_COLUMNS_B);

    MPI_Finalize();

    return 0;
}


/*
 * DEBUG GRIND INFO:
    if (currentRank == 0) {
        std::cout << dims[0] << " " << dims[1] << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << currentRank << "::::: " << coords[0] << " " << coords[1] << "\n";
 */
