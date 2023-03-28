#include <iostream>
#include <mpi.h>

enum SizeConsts {
    NUMBER_OF_LINES_A = 3,
    NUMBER_OF_COLUMNS_A = 6, /* must be the same with next parameter! */
    NUMBER_OF_LINES_B = 6, /* must be the same with previous parameter! */
    NUMBER_OF_COLUMNS_B = 4,
    HORIZONTAL_NUMBER_OF_NODES = 2, /* first parameter of grid: p1 */
    VERTICAL_NUMBER_OF_NODES = 2 /* second parameter of grid: p2 */
};

enum DimensionConsts {
    MAX_DIMENSION = 2
};

enum Axes {
    VERTICAL_AXIS = 0,
    HORIZONTAL_AXIS = 1
};

double *GenerateMatrixA() {
    double *matrixA = new double[NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_A];
    int currentPosition = 0;
    for (int i = 0; i < NUMBER_OF_LINES_A; ++i) {
        for (int j = 0; j < NUMBER_OF_COLUMNS_A; ++j) {
            matrixA[currentPosition] = currentPosition;
            ++currentPosition;
        }
    }
    return matrixA;
}

double *GenerateMatrixB() {
    double *matrixB = new double[NUMBER_OF_LINES_B * NUMBER_OF_COLUMNS_B];
    int currentPosition = 0;
    for (int i = 0; i < NUMBER_OF_LINES_B; ++i) {
        for (int j = 0; j < NUMBER_OF_COLUMNS_B; ++j) {
            matrixB[currentPosition] = currentPosition;
            ++currentPosition;
        }
    }
    return matrixB;
}

void DebugPrint(double *matrix, int lines, int columns) {
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

void SendMatrixAFromRootToVerticalNeighbors(double *matrixA, int matrixSize, double *receiveBufferForMatrixA,
                                            MPI_Comm COMMUNICATOR, int commSize) {
    int sizeOfPartOfMatrixA = matrixSize / commSize;
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(COMMUNICATOR, coords, &rootRank);
    MPI_Scatter(matrixA, sizeOfPartOfMatrixA, MPI_DOUBLE, receiveBufferForMatrixA, sizeOfPartOfMatrixA, MPI_DOUBLE,
                rootRank, COMMUNICATOR);
}

void SendMatrixBFromRootToHorizontalNeighbors(double *matrixB, int numberOfColumnsInPartOfMatrixB,
                                              double *receiveBufferForMatrixB,
                                              MPI_Comm COMMUNICATOR, MPI_Datatype VECTOR_TYPE) {
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(COMMUNICATOR, coords, &rootRank);
    MPI_Scatter(matrixB, numberOfColumnsInPartOfMatrixB, VECTOR_TYPE, receiveBufferForMatrixB,
                numberOfColumnsInPartOfMatrixB * NUMBER_OF_LINES_B, MPI_DOUBLE, rootRank, COMMUNICATOR);
}

int main(int argc, char **argv) {
    int numberOfProcesses;
    int currentRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    int dims[MAX_DIMENSION] = {0, 0};
    int periods[MAX_DIMENSION] = {0, 0};
    int reorder = 0;
    MPI_Comm MPI_CUSTOM_2D_GRID;
    MPI_Dims_create(numberOfProcesses, MAX_DIMENSION, dims);
    MPI_Cart_create(MPI_COMM_WORLD, MAX_DIMENSION, dims, periods, reorder, &MPI_CUSTOM_2D_GRID);

    if (currentRank == 0) {
        std::cout << "DIMS: " << dims[0] << " " << dims[1] << "\n";
    }

    double *matrixA = NULL;
    double *matrixB = NULL;
    double *matrixC = NULL; /* result-matrix */

    MPI_Comm ROOT_HORIZONTAL_NEIGHBORS;
    MPI_Comm ROOT_VERTICAL_NEIGHBORS;

    int horizontal_remain_dims[MAX_DIMENSION] = {false, true};
    int vertical_remain_dims[MAX_DIMENSION] = {true, false};
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, horizontal_remain_dims, &ROOT_HORIZONTAL_NEIGHBORS);
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, vertical_remain_dims, &ROOT_VERTICAL_NEIGHBORS);

    int currentRankCoords[MAX_DIMENSION] = {0, 0};
    MPI_Cart_coords(MPI_CUSTOM_2D_GRID, currentRank, MAX_DIMENSION, currentRankCoords);
    if (currentRankCoords[0] == 0 && currentRankCoords[1] == 0) {
        matrixA = GenerateMatrixA();
        matrixB = GenerateMatrixB();
        DebugPrint(matrixB, NUMBER_OF_LINES_B, NUMBER_OF_COLUMNS_B);
    }

    int sizeOfMatrixA = NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_A;
    double *receiveBufferForMatrixA = new double[sizeOfMatrixA / dims[VERTICAL_AXIS]];
    if (currentRank % dims[HORIZONTAL_AXIS] == 0) {
        SendMatrixAFromRootToVerticalNeighbors(matrixA, sizeOfMatrixA, receiveBufferForMatrixA, ROOT_VERTICAL_NEIGHBORS,
                                               dims[VERTICAL_AXIS]);
    }

    std::cout << "(Scatter A) CURRENT RANK: " << currentRank << "\n";
    DebugPrint(receiveBufferForMatrixA, NUMBER_OF_LINES_A / dims[VERTICAL_AXIS], NUMBER_OF_COLUMNS_A);

    MPI_Datatype COLUMN_TYPE;
    /* 1 - distance between elements in column */
    MPI_Type_vector(NUMBER_OF_LINES_B, 1, NUMBER_OF_COLUMNS_B, MPI_DOUBLE, &COLUMN_TYPE);
    MPI_Type_commit(&COLUMN_TYPE);

    MPI_Datatype RESIZED_COLUMN_TYPE;
    MPI_Type_create_resized(COLUMN_TYPE, 0, (NUMBER_OF_COLUMNS_B / dims[VERTICAL_AXIS]) * sizeof(double),
                            &RESIZED_COLUMN_TYPE);
    MPI_Type_commit(&RESIZED_COLUMN_TYPE);

    MPI_Barrier(MPI_COMM_WORLD);
    int sizeOfMatrixB = NUMBER_OF_LINES_B * NUMBER_OF_COLUMNS_B;
    double *receiveBufferForMatrixB = new double[sizeOfMatrixB / dims[HORIZONTAL_AXIS]];
    if (currentRank < dims[HORIZONTAL_AXIS]) {
        SendMatrixBFromRootToHorizontalNeighbors(matrixB, NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS],
                                                 receiveBufferForMatrixB, ROOT_HORIZONTAL_NEIGHBORS,
                                                 RESIZED_COLUMN_TYPE);
    }

    if (currentRank == 0) {
        std::cout << "\n num * lines " << NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS] * NUMBER_OF_LINES_B << " num: "
                  << NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS] << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "(Scatter B) CURRENT RANK: " << currentRank << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
    DebugPrint(receiveBufferForMatrixB, NUMBER_OF_LINES_B / dims[HORIZONTAL_AXIS], NUMBER_OF_COLUMNS_B);
    MPI_Barrier(MPI_COMM_WORLD);

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

/*
 * COLUMN_TYPE_TEST
    if (currentRank == 0) {
        double matrix[6][3];
        double count = 10;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 3; ++j) {
                matrix[i][j] = count;
                ++count;
            }
        }
        MPI_Send(&matrix[0][0], 1, COLUMN_TYPE, 1, 0, MPI_COMM_WORLD);
    } else if (currentRank == 1) {
        double recvBuf[6];
        MPI_Recv(recvBuf, 6, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "TEST:\n";
        for (int i = 0; i < 6; ++i) {
            std::cout << recvBuf[i] << " ";
        }
    }
 */
