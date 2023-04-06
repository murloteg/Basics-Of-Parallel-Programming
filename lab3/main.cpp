#include <iostream>
#include <mpi.h>

enum SizeConsts {
    NUMBER_OF_LINES_A = 8,
    NUMBER_OF_COLUMNS_A = 8, /* must be the same with next parameter! */
    NUMBER_OF_LINES_B = 8, /* must be the same with previous parameter! */
    NUMBER_OF_COLUMNS_B = 12,
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
    int value = 2;
    for (int i = 0; i < NUMBER_OF_LINES_B; ++i) {
        for (int j = 0; j < NUMBER_OF_COLUMNS_B; ++j) {
            if (i == j) {
                matrixB[currentPosition] = value;
            }
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

void InvokeBcastForCommunicator(double* partOfMatrixA, int sizeOfPartOfMatrixA, MPI_Comm COMMUNICATOR) {
    int coords[1] = {0};
    int rootRank = 0;
    MPI_Cart_rank(COMMUNICATOR, coords, &rootRank);
    MPI_Bcast(partOfMatrixA, sizeOfPartOfMatrixA, MPI_DOUBLE, rootRank, COMMUNICATOR);
}

void MatrixMultiplication(double* firstMatrix, int firstMatrixLines, int firstMatrixColumns,
                          double* secondMatrix, int secondMatrixColumns, double* resultMatrix) {
    for (int i = 0; i < firstMatrixLines; ++i) {
        for (int k = 0; k < firstMatrixColumns; ++k) {
            for (int j = 0; j < secondMatrixColumns; ++j) {
                resultMatrix[i * secondMatrixColumns + j] += firstMatrix[i * firstMatrixColumns + k] * secondMatrix[k * secondMatrixColumns + j];
            }
        }
    }
}

void PrepareGathervArguments(int* receiveCounts, int* offsets, int* dims) {
    for (int i = 0; i < dims[VERTICAL_AXIS]; ++i) {
        for (int j = 0; j < dims[HORIZONTAL_AXIS]; ++j) {
            offsets[i * dims[VERTICAL_AXIS] + j] =
                    ((i * NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_B / dims[VERTICAL_AXIS]) +
                     j * NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]) / (NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]);
            std::cout << "pos: (" << j << " : " << i << "), offset: " << offsets[i * dims[VERTICAL_AXIS] + j] << "\n";
            receiveCounts[i * dims[VERTICAL_AXIS] + j] = 1;
        }
    }
}

void MergeAllPartialResultsIntoResultMatrix(double* resultMatrix, double* partOfMatrixC, int numberOfLinesPartMatrixC,
                                            int numberOfColumnsPartMatrixC, int numberOfProcesses,
                                            int* dims, MPI_Comm MPI_CUSTOM_2D_GRID) {
    int coords[MAX_DIMENSION] = {0, 0};
    int rootRank = 0;
    MPI_Cart_rank(MPI_CUSTOM_2D_GRID, coords, &rootRank);

    MPI_Datatype BLOCK_OF_MATRIX_C;
    MPI_Type_vector(numberOfLinesPartMatrixC, numberOfColumnsPartMatrixC, NUMBER_OF_COLUMNS_B, MPI_DOUBLE, &BLOCK_OF_MATRIX_C);
    MPI_Type_commit(&BLOCK_OF_MATRIX_C);

    MPI_Datatype RESIZED_BLOCK_OF_MATRIX_C;
    MPI_Type_create_resized(BLOCK_OF_MATRIX_C, 0, numberOfColumnsPartMatrixC * sizeof(double), &RESIZED_BLOCK_OF_MATRIX_C);
    MPI_Type_commit(&RESIZED_BLOCK_OF_MATRIX_C);
    // OK!

    int receiveCounts[numberOfProcesses];
    int offsets[numberOfProcesses];

    int currentRank;
    MPI_Comm_rank(MPI_CUSTOM_2D_GRID, &currentRank);
    int currentRankCoords[MAX_DIMENSION] = {0, 0};
    MPI_Cart_coords(MPI_CUSTOM_2D_GRID, currentRank, MAX_DIMENSION, currentRankCoords);

    if (currentRankCoords[VERTICAL_AXIS] == 0 && currentRankCoords[HORIZONTAL_AXIS] == 0) {
        PrepareGathervArguments(receiveCounts, offsets, dims);
    }
    MPI_Gatherv(partOfMatrixC, numberOfLinesPartMatrixC * numberOfColumnsPartMatrixC, MPI_DOUBLE, resultMatrix, receiveCounts, offsets, RESIZED_BLOCK_OF_MATRIX_C, rootRank, MPI_CUSTOM_2D_GRID);

    MPI_Type_free(&BLOCK_OF_MATRIX_C);
    MPI_Type_free(&RESIZED_BLOCK_OF_MATRIX_C);
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
        std::cerr << "INCORRECT MATRIX SIZES!" << std::endl;
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
        std::cout << "[DEBUG] DIMS: " << dims[HORIZONTAL_AXIS] << " " << dims[VERTICAL_AXIS] << "\n";
    }

    double* matrixA = NULL;
    double* matrixB = NULL;
    double* matrixC = NULL; /* result-matrix */

    MPI_Comm HORIZONTAL_NEIGHBORS;
    MPI_Comm VERTICAL_NEIGHBORS;

    int horizontalRemainDims[MAX_DIMENSION] = {false, true};
    int verticalRemainDims[MAX_DIMENSION] = {true, false};
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, horizontalRemainDims, &HORIZONTAL_NEIGHBORS);
    MPI_Cart_sub(MPI_CUSTOM_2D_GRID, verticalRemainDims, &VERTICAL_NEIGHBORS);

    int currentRankCoords[MAX_DIMENSION] = {0, 0};
    MPI_Cart_coords(MPI_CUSTOM_2D_GRID, currentRank, MAX_DIMENSION, currentRankCoords);

    double start = MPI_Wtime();

    /* "Root" node in the 2D-grid */
    if (currentRankCoords[VERTICAL_AXIS] == 0 && currentRankCoords[HORIZONTAL_AXIS] == 0) {
        matrixA = GenerateMatrixA();
        matrixB = GenerateMatrixB();
        matrixC = new double[NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_B];
    }

    int sizeOfMatrixA = NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_A;
    double* receiveBufferForMatrixA = new double[sizeOfMatrixA / dims[VERTICAL_AXIS]];
    if (currentRankCoords[HORIZONTAL_AXIS] == 0) {
        SendMatrixAFromRootToVerticalNeighbors(matrixA, sizeOfMatrixA, receiveBufferForMatrixA, VERTICAL_NEIGHBORS,
                                               dims[VERTICAL_AXIS]);
    }

    InvokeBcastForCommunicator(receiveBufferForMatrixA, sizeOfMatrixA / dims[VERTICAL_AXIS], HORIZONTAL_NEIGHBORS);
    /* [DEBUG] Scatter and Bcast for matrixA [DEBUG]
        std::cout << "(After Scatter A) CURRENT RANK: " << currentRank << "\n";
        DebugPrint(receiveBufferForMatrixA, NUMBER_OF_LINES_A / dims[VERTICAL_AXIS], NUMBER_OF_COLUMNS_A);
     */


    MPI_Datatype COLUMN_TYPE;
    MPI_Type_vector(NUMBER_OF_LINES_B, NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS], NUMBER_OF_COLUMNS_B, MPI_DOUBLE, &COLUMN_TYPE);
    MPI_Type_commit(&COLUMN_TYPE);

    MPI_Datatype RESIZED_COLUMN_TYPE;
    MPI_Type_create_resized(COLUMN_TYPE, 0, (NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]) * sizeof(double),
                            &RESIZED_COLUMN_TYPE);
    MPI_Type_commit(&RESIZED_COLUMN_TYPE);

    int sizeOfMatrixB = NUMBER_OF_LINES_B * NUMBER_OF_COLUMNS_B;
    double* receiveBufferForMatrixB = new double[sizeOfMatrixB / dims[HORIZONTAL_AXIS]];
    if (currentRankCoords[VERTICAL_AXIS] == 0) {
        SendMatrixBFromRootToHorizontalNeighbors(matrixB, receiveBufferForMatrixB, dims, HORIZONTAL_NEIGHBORS,
                                                 RESIZED_COLUMN_TYPE);
    }

    InvokeBcastForCommunicator(receiveBufferForMatrixB, sizeOfMatrixB / dims[HORIZONTAL_AXIS], VERTICAL_NEIGHBORS);
    /* [DEBUG] Scatter and Bcast for matrixB [DEBUG]
        std::cout << "(After Scatter B) CURRENT RANK: " << currentRank << "\n";
        DebugPrint(receiveBufferForMatrixB, NUMBER_OF_LINES_B, NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]);
     */

    /* Multiplication part of matrix A by part of matrix B in current node */
    MPI_Barrier(MPI_CUSTOM_2D_GRID);
    double* partOfMatrixC = new double[NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_B / (dims[VERTICAL_AXIS] * dims[HORIZONTAL_AXIS])];
    MatrixMultiplication(receiveBufferForMatrixA, NUMBER_OF_LINES_A / dims[VERTICAL_AXIS], NUMBER_OF_COLUMNS_A,
                         receiveBufferForMatrixB, NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS], partOfMatrixC);


    int numberOfLinesPartMatrixC = NUMBER_OF_LINES_A / dims[VERTICAL_AXIS]; // OK!
    int numberOfColumnsPartMatrixC = NUMBER_OF_COLUMNS_B / dims[HORIZONTAL_AXIS]; // OK!
    if (currentRank == 1) {
        std::cout << "LINES: " << numberOfLinesPartMatrixC << " COLUMNS: " << numberOfColumnsPartMatrixC << "\n";
        DebugPrint(partOfMatrixC, numberOfLinesPartMatrixC, numberOfColumnsPartMatrixC);
    }

    MPI_Barrier(MPI_CUSTOM_2D_GRID);
    MergeAllPartialResultsIntoResultMatrix(matrixC, partOfMatrixC, numberOfLinesPartMatrixC, numberOfColumnsPartMatrixC, numberOfProcesses, dims, MPI_CUSTOM_2D_GRID);
    double end = MPI_Wtime();

    /* "Check" block */
    if (currentRankCoords[HORIZONTAL_AXIS] == 0 && currentRankCoords[VERTICAL_AXIS] == 0) {
        double* checkMatrix = new double[NUMBER_OF_LINES_A * NUMBER_OF_COLUMNS_B];
        MatrixMultiplication(matrixA, NUMBER_OF_LINES_A, NUMBER_OF_COLUMNS_A, matrixB, NUMBER_OF_COLUMNS_B, checkMatrix);
        DebugPrint(checkMatrix, NUMBER_OF_LINES_A, NUMBER_OF_COLUMNS_B);
        DebugPrint(matrixC, NUMBER_OF_LINES_A, NUMBER_OF_COLUMNS_B);
        delete[] checkMatrix;
    }

    CleanUp(matrixA, matrixB, matrixC, receiveBufferForMatrixA, receiveBufferForMatrixB, partOfMatrixC);

    if (currentRank == 0) {
        std::cout << "Elapsed time [sec]: " << end - start << "\n";
    }

    MPI_Type_free(&COLUMN_TYPE);
    MPI_Type_free(&RESIZED_COLUMN_TYPE);

    MPI_Finalize();

    return 0;
}
