#include <iostream>
#include <mpi.h>

enum SizeConsts {
    NUMBER_OF_LINES_A = 2,
    NUMBER_OF_COLUMNS_A = 4, /* must be the same with next parameter! */
    NUMBER_OF_LINES_B = 4, /* must be the same with previous parameter! */
    NUMBER_OF_COLUMNS_B = 8,
    HORIZONTAL_NUMBER_OF_NODES = 0, /* first parameter of grid: p1 */
    VERTICAL_NUMBER_OF_NODES = 0 /* second parameter of grid: p2 */
};

enum DimensionConsts {
    MAX_DIMENSION = 2
};

enum Axes {
    VERTICAL_AXIS = 0,
    HORIZONTAL_AXIS = 1
};

enum Errors {
    INCORRECT_MATRIX_SIZES = -1
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

void SendMatrixAFromRootToVerticalNeighbors(double* matrixA, int matrixSize, double* receiveBufferForMatrixA,
                                            MPI_Comm COMMUNICATOR, int commSize) {
    int sizeOfPartOfMatrixA = matrixSize / commSize;
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(COMMUNICATOR, coords, &rootRank);
    MPI_Scatter(matrixA, sizeOfPartOfMatrixA, MPI_DOUBLE, receiveBufferForMatrixA, sizeOfPartOfMatrixA, MPI_DOUBLE,
                rootRank, COMMUNICATOR);
}

void SendMatrixBFromRootToHorizontalNeighbors(double* matrixB, double* receiveBufferForMatrixB, const int* dims,
                                              MPI_Comm COMMUNICATOR, MPI_Datatype VECTOR_TYPE) {
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(COMMUNICATOR, coords, &rootRank);
    MPI_Scatter(matrixB, 1, VECTOR_TYPE, receiveBufferForMatrixB,
                NUMBER_OF_LINES_B * NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS], MPI_DOUBLE, rootRank, COMMUNICATOR);
}

void InvokeBcastForHorizontalCommunicator(double* partOfMatrixA, int sizeOfPartOfMatrixA, MPI_Comm HORIZONTAL_COMMUNICATOR) {
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(HORIZONTAL_COMMUNICATOR, coords, &rootRank);
    MPI_Bcast(partOfMatrixA, sizeOfPartOfMatrixA, MPI_DOUBLE, rootRank, HORIZONTAL_COMMUNICATOR);
}

void InvokeBcastForVerticalCommunicator(double* partOfMatrixB, int sizeOfPartOfMatrixB, MPI_Comm VERTICAL_COMMUNICATOR) {
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(VERTICAL_COMMUNICATOR, coords, &rootRank);
    MPI_Bcast(partOfMatrixB, sizeOfPartOfMatrixB, MPI_DOUBLE, rootRank, VERTICAL_COMMUNICATOR);
}

void MatrixMultiplication(double* firstMatrix, int firstMatrixLines, int firstMatrixColumns,
                          double* secondMatrix, int secondMatrixColumns, double* resultMatrix) {
    double partialSum = 0;
    for (int i = 0; i < firstMatrixLines; ++i) {
        for (int j = 0; j < secondMatrixColumns; ++j) {
            for (int k = 0; k < firstMatrixColumns; ++k) {
                partialSum += firstMatrix[i * firstMatrixColumns + k] * secondMatrix[k * secondMatrixColumns + j];
            }
            resultMatrix[i * secondMatrixColumns + j] = partialSum;
            partialSum = 0;
        }
    }
}

void CleanUp(double* matrixA, double* matrixB, double* matrixC, double* partOfMatrixA,
             double* partOfMatrixB, double* partOfMatrixC) {
    delete[] (matrixA);
    delete[] (matrixB);
    delete[] (matrixC);
    delete[] (partOfMatrixA);
    delete[] (partOfMatrixB);
    delete[] (partOfMatrixC);
}

int main(int argc, char** argv) {
    if (NUMBER_OF_COLUMNS_A != NUMBER_OF_LINES_B) {
        std::cout << "INCORRECT MATRIX SIZES!" << std::endl;
        return INCORRECT_MATRIX_SIZES;
    }

    int numberOfProcesses;
    int currentRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    /* Format of dims: (Y ; X) */
    int dims[MAX_DIMENSION] = {VERTICAL_NUMBER_OF_NODES, HORIZONTAL_NUMBER_OF_NODES};
    int periods[MAX_DIMENSION] = {0, 0};
    int reorder = 0;
    MPI_Comm MPI_CUSTOM_2D_GRID;
    MPI_Dims_create(numberOfProcesses, MAX_DIMENSION, dims);
    MPI_Cart_create(MPI_COMM_WORLD, MAX_DIMENSION, dims, periods, reorder, &MPI_CUSTOM_2D_GRID);

    if (currentRank == 0) {
        std::cout << "[DEBUG] DIMS: " << dims[0] << " " << dims[1] << "\n";
    }

    double* matrixA = NULL;
    double* matrixB = NULL;
    double* matrixC = NULL; /* result-matrix */

    MPI_Comm HORIZONTAL_NEIGHBORS;
    MPI_Comm VERTICAL_NEIGHBORS;

    int horizontal_remain_dims[MAX_DIMENSION] = {false, true};
    int vertical_remain_dims[MAX_DIMENSION] = {true, false};
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, horizontal_remain_dims, &HORIZONTAL_NEIGHBORS);
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, vertical_remain_dims, &VERTICAL_NEIGHBORS);

    int currentRankCoords[MAX_DIMENSION] = {0, 0};
    MPI_Cart_coords(MPI_CUSTOM_2D_GRID, currentRank, MAX_DIMENSION, currentRankCoords);
    if (currentRankCoords[0] == 0 && currentRankCoords[1] == 0) {
        matrixA = GenerateMatrixA();
        matrixB = GenerateMatrixB();
        DebugPrint(matrixB, NUMBER_OF_LINES_B, NUMBER_OF_COLUMNS_B);
    }

    int sizeOfMatrixA = NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_A;
    double* receiveBufferForMatrixA = new double[sizeOfMatrixA / dims[VERTICAL_AXIS]];
    if (currentRank % dims[HORIZONTAL_AXIS] == 0) {
        SendMatrixAFromRootToVerticalNeighbors(matrixA, sizeOfMatrixA, receiveBufferForMatrixA, VERTICAL_NEIGHBORS,
                                               dims[VERTICAL_AXIS]);
    }

    InvokeBcastForHorizontalCommunicator(receiveBufferForMatrixA, sizeOfMatrixA / dims[VERTICAL_AXIS], HORIZONTAL_NEIGHBORS);
    /* [DEBUG] Scatter and Bcast for matrixA [DEBUG]
        std::cout << "(Scatter A) CURRENT RANK: " << currentRank << "\n";
        DebugPrint(receiveBufferForMatrixA, NUMBER_OF_LINES_A / dims[VERTICAL_AXIS], NUMBER_OF_COLUMNS_A);
     */


    MPI_Datatype COLUMN_TYPE;
    /* 1 - distance between elements in column */
    MPI_Type_vector(NUMBER_OF_LINES_B, NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS], NUMBER_OF_COLUMNS_B, MPI_DOUBLE, &COLUMN_TYPE);
    MPI_Type_commit(&COLUMN_TYPE);

    MPI_Datatype RESIZED_COLUMN_TYPE;
    MPI_Type_create_resized(COLUMN_TYPE, 0, (NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]) * sizeof(double),
                            &RESIZED_COLUMN_TYPE);
    MPI_Type_commit(&RESIZED_COLUMN_TYPE);

    int sizeOfMatrixB = NUMBER_OF_LINES_B * NUMBER_OF_COLUMNS_B;
    double* receiveBufferForMatrixB = new double[sizeOfMatrixB / dims[HORIZONTAL_AXIS]];
    if (currentRank < dims[HORIZONTAL_AXIS]) {
        SendMatrixBFromRootToHorizontalNeighbors(matrixB, receiveBufferForMatrixB, dims, HORIZONTAL_NEIGHBORS,
                                                 RESIZED_COLUMN_TYPE);
    }

    InvokeBcastForVerticalCommunicator(receiveBufferForMatrixB, sizeOfMatrixB / dims[VERTICAL_AXIS], VERTICAL_NEIGHBORS);
    /* [DEBUG] Scatter matrixB [DEBUG]
        std::cout << "(Scatter B) CURRENT RANK: " << currentRank << "\n";
        DebugPrint(receiveBufferForMatrixB, NUMBER_OF_LINES_B, NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]);
     */

    double* partOfMatrixC = new double[NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_B / (dims[VERTICAL_AXIS] * dims[HORIZONTAL_AXIS])];

    /* Multiplication part of matrix A by part of matrix B in current node */
    MatrixMultiplication(receiveBufferForMatrixA, NUMBER_OF_LINES_A / dims[VERTICAL_AXIS], NUMBER_OF_COLUMNS_A,
                         receiveBufferForMatrixB, NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS], partOfMatrixC);

    MPI_Barrier(MPI_COMM_WORLD);
    if (currentRank == 0) {
        DebugPrint(partOfMatrixC, NUMBER_OF_LINES_A / dims[VERTICAL_AXIS], NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]);
    }

    CleanUp(matrixA, matrixB, matrixC, receiveBufferForMatrixA, receiveBufferForMatrixB, partOfMatrixC);

    MPI_Type_free(&COLUMN_TYPE);
    MPI_Type_free(&RESIZED_COLUMN_TYPE);

    MPI_Finalize();

    return 0;
}

/*
 * EXAMPLES:
 * A = 2x4, B = 4x8; NP = 4. RESULT OF NODE (0, 0): "112 118 124 130".
 */
