#include <iostream>
#include <random>
#include <mpi.h>
#include <vector>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <atomic>

typedef int task_complexity_t;

enum Consts {
    MAX_TASK_COMPLEXITY = 2,
    ACCELERATION_PARAMETER = 1,
    NUMBER_OF_TASKS = 40
};

enum MPIConsts {
    REQUEST_TAG = 111,
    RESPONSE_TAG = 222
};

enum TaskStatuses {
    EMPTY_TASK = -1
};

std::mutex mutex;

std::atomic<bool> isExecutorInterrupted(false);
std::atomic<bool> isSenderInterrupted(false);
std::atomic<bool> isRequesterInterrupted(false);

std::atomic<bool> gotSignalFromAnotherProcess(false);
std::atomic<bool> sentSignalToAnotherProcess(false);

int GenerateRandomValue() {
    std::random_device device;
    std::mt19937 range(device());
    std::uniform_int_distribution<std::mt19937::result_type> distribution(1, MAX_TASK_COMPLEXITY);
    return (int) distribution(range);
}

/* task describes by integer value (time to sleep) */
task_complexity_t CreateTask() {
    task_complexity_t taskComplexity = GenerateRandomValue();
    return taskComplexity;
}

void CreateTaskList(std::vector<task_complexity_t>& taskList) {
    for (int i = 0; i < NUMBER_OF_TASKS; ++i) {
        taskList.push_back(CreateTask());
    }
}

void ExecuteTask(std::vector<task_complexity_t>& taskList, int rank) {
    while (!isExecutorInterrupted) {
        if (!gotSignalFromAnotherProcess && !sentSignalToAnotherProcess) {
            mutex.lock();
            if (!taskList.empty()) {
                task_complexity_t taskComplexity = taskList.back();
                taskList.pop_back();
                std::cout << "From rank: " << rank << "; task complexity: " << taskComplexity << "\n";
                sleep(taskComplexity);
            }
            mutex.unlock();
        }
        if (isRequesterInterrupted) {
            isExecutorInterrupted = true;
        }
    }
}

void SendTask(std::vector<task_complexity_t>& taskList, int rank) {
    while (!isSenderInterrupted) {
        int requesterRank;
        MPI_Recv(&requesterRank, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        gotSignalFromAnotherProcess = true;
        std::cout << "Rank: " << rank << " got signal from process: " << requesterRank << "\n";
        task_complexity_t task = EMPTY_TASK;
        if (!sentSignalToAnotherProcess) {
            mutex.lock();
            if (!taskList.empty()) {
                task = taskList.back();
                taskList.pop_back();
            }
            mutex.unlock();
        }
        MPI_Send(&task, 1, MPI_INT, requesterRank, RESPONSE_TAG, MPI_COMM_WORLD);
        gotSignalFromAnotherProcess = false;
        std::cout << "Rank: " << rank << " trying to response to process: " << requesterRank << "\n";
        if (isRequesterInterrupted) {
            isSenderInterrupted = true;
        }
    }
}

void RequestTask(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    while (!isRequesterInterrupted) {
        mutex.lock();
        if (taskList.empty()) {
            std::cout << "Rank: " << rank << " trying to request some task...\n";
            int failedResponsesCounter = 0;
            for (int i = 0; i < numberOfProcesses; ++i) {
                if (rank != i) {
                    MPI_Send(&rank, 1, MPI_INT, i, REQUEST_TAG, MPI_COMM_WORLD);
                    sentSignalToAnotherProcess = true;
                    task_complexity_t responseTask = EMPTY_TASK;
                    MPI_Recv(&responseTask, 1, MPI_INT, i, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                    std::cout << "Rank: " << rank << " received task with complexity: " << responseTask << " from process: " << i << "\n";
                    if (responseTask == EMPTY_TASK) {
                        ++failedResponsesCounter;
                    } else {
                        taskList.push_back(responseTask);
                        sentSignalToAnotherProcess = false;
                        break;
                    }
                }
            }

            if (failedResponsesCounter == numberOfProcesses - 1) {
                std::cout << "All requests from rank: " << rank << " were ignored\n";
                // TODO: check other processes and exit if needed.
                isRequesterInterrupted = true;
            }
        }
        mutex.unlock();
    }
}

int* CreateNumberOfTasksArray(int numberOfProcesses) {
    int* numberOfTasksArray = new int[numberOfProcesses];
    for (int i = 0; i < numberOfProcesses; ++i) {
        numberOfTasksArray[i] = NUMBER_OF_TASKS / numberOfProcesses;
        if (i < NUMBER_OF_TASKS % numberOfProcesses) {
            ++numberOfTasksArray[i];
        }
    }
    return numberOfTasksArray;
}

void DebugPrintVector(std::vector<task_complexity_t>& taskList, int rank) {
    std::cout << "Rank: " << rank << "\n";
    for (int i = 0; i < taskList.size(); ++i) {
        std::cout << "Task <" << i << ">: " << taskList.at(i) << "\n";
    }
}

std::vector<task_complexity_t> DistributeTasks(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    int* numberOfTasksArray = CreateNumberOfTasksArray(numberOfProcesses);

    /* [DEBUG] Distribution of tasks [DEBUG] */
    if (rank == 0) {
        for (int i = 0; i < numberOfProcesses; ++i) {
            std::cout << "Rank <" << i << ">: " << numberOfTasksArray[i] << "\n";
        }
        std::cout << "BEFORE DISTRIBUTING:\n";
        DebugPrintVector(taskList, rank);
    }

    std::vector<task_complexity_t> temporaryList;
    task_complexity_t* tasks = new task_complexity_t[numberOfTasksArray[rank]];
    if (rank == 0) {
        for (int i = numberOfProcesses - 1; i >= 0; --i) {
            int requiredTasks = numberOfTasksArray[i];
            for (int j = 0; j < requiredTasks; ++j) {
                tasks[j] = taskList.back();
                tasks[j] += ACCELERATION_PARAMETER * i;
                taskList.pop_back();
            }
            MPI_Send(tasks, requiredTasks, MPI_INT, i, i, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(tasks, numberOfTasksArray[rank], MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }

    /* Create a vector from array */
    std::vector<task_complexity_t> distributedTaskList(tasks, tasks + numberOfTasksArray[rank]);

    delete[] tasks;
    delete[] numberOfTasksArray;
    return distributedTaskList;
}

void CleanUp(std::vector<task_complexity_t>& taskList) {
    taskList.clear();
}

int main(int argc, char** argv) {
    int rank;
    int numberOfProcesses;
    int status;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &status);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    std::vector<task_complexity_t> taskList;
    taskList.resize(0);
    if (rank == 0) {
        CreateTaskList(taskList);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    taskList = DistributeTasks(taskList, rank, numberOfProcesses);
    /* [DEBUG] Check distribution of tasks [DEBUG]
    if (rank == numberOfProcesses - 1) {
        DebugPrintVector(taskList, rank);
    }
    */

    std::thread executorThread(ExecuteTask, std::ref(taskList), rank);
    std::thread senderThread(SendTask, std::ref(taskList), rank);
    std::thread requesterThread(RequestTask, std::ref(taskList), rank, numberOfProcesses);

    executorThread.join();
    senderThread.join();
    requesterThread.join();

    double end = MPI_Wtime();
    if (rank == numberOfProcesses - 1) {
        std::cout << "Elapsed time: " << end - start << "\n";
    }

    CleanUp(taskList);

    MPI_Finalize();

    return 0;
}
