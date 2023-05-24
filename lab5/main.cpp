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
    MIN_TASK_COMPLEXITY = 1,
    MAX_TASK_COMPLEXITY = 5,
    NUMBER_OF_TASKS = 30
};

enum MPIConsts {
    REQUEST_TAG = 111,
    RESPONSE_TAG = 222,
    FINISH_PROCESS = -1
};

enum TaskStatuses {
    EMPTY_TASK = -1
};

std::mutex mutex;
std::condition_variable executorCondVar;

std::atomic<bool> isExecutorInterrupted(false);
std::atomic<bool> isRequesterInterrupted(false);

/* task describes by integer value (time to sleep) */
int GenerateRandomTask() {
    std::random_device device;
    std::mt19937 range(device());
    std::uniform_int_distribution<std::mt19937::result_type> distribution(1, MAX_TASK_COMPLEXITY);
    return (int) distribution(range);
}

void CreateRandomTaskList(std::vector<task_complexity_t>& taskList) {
    for (int i = 0; i < NUMBER_OF_TASKS; ++i) {
        taskList.push_back(GenerateRandomTask());
    }
}

void CreateDeterminedTaskList(std::vector<task_complexity_t>& taskList, int numberOfProcesses) {
    int taskComplexity = MIN_TASK_COMPLEXITY;
    int currentRank = 0;
    int requiredTasksForProcess = NUMBER_OF_TASKS / numberOfProcesses;
    while (taskList.size() < NUMBER_OF_TASKS) {
        for (int i = 0; i < requiredTasksForProcess; ++i) {
            taskList.push_back(2 * taskComplexity);
        }
        if (currentRank < NUMBER_OF_TASKS % numberOfProcesses) {
            taskList.push_back(2 * taskComplexity);
        }
        ++taskComplexity;
        ++currentRank;
    }
}

void ExecuteTask(std::vector<task_complexity_t>& taskList, int rank, int& tasksTimeTakenByProcess) {
    while (!isExecutorInterrupted) {
        std::unique_lock<std::mutex> uniqueLock(mutex);
        executorCondVar.wait(uniqueLock);

        if (!taskList.empty()) {
            task_complexity_t taskComplexity = taskList.back();
            taskList.pop_back();
//            std::cout << "Rank: " << rank << " executing task with complexity: " << taskComplexity << "\n";
            sleep(taskComplexity);
            tasksTimeTakenByProcess += taskComplexity;
        }
        uniqueLock.unlock();

        if (isRequesterInterrupted) {
            isExecutorInterrupted = true;
        }
    }
}

void SendTask(std::vector<task_complexity_t>& taskList, int rank) {
    while (true) {
        int requesterRank;
        MPI_Recv(&requesterRank, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        if (requesterRank == FINISH_PROCESS) {
            break;
        }

//        std::cout << "Rank: " << rank << " got signal from process: " << requesterRank << "\n";
        task_complexity_t task = EMPTY_TASK;

        mutex.lock();
        if (!taskList.empty()) {
            task = taskList.back();
            taskList.pop_back();
        }
        mutex.unlock();
        MPI_Send(&task, 1, MPI_INT, requesterRank, RESPONSE_TAG, MPI_COMM_WORLD);
//        std::cout << "Rank: " << rank << " trying to response to process: " << requesterRank << "\n";
    }
}

void RequestTask(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    while (!isRequesterInterrupted) {
        bool isTaskListEmpty = false;
        mutex.lock();
        if (taskList.empty()) {
            isTaskListEmpty = true;
//            std::cout << "Rank: " << rank << " trying to request some task...\n";
        }
        mutex.unlock();

        if (isTaskListEmpty) {
            int failedResponsesCounter = 0;
            for (int i = 0; i < numberOfProcesses; ++i) {
                if (rank != i) {
                    MPI_Send(&rank, 1, MPI_INT, i, REQUEST_TAG, MPI_COMM_WORLD);
                    task_complexity_t responseTask;
                    MPI_Recv(&responseTask, 1, MPI_INT, i, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
//                    std::cout << "Rank: " << rank << " received task with complexity: " << responseTask << " from process: " << i << "\n";
                    if (responseTask == EMPTY_TASK) {
                        ++failedResponsesCounter;
                    } else {
                        mutex.lock();
                        taskList.push_back(responseTask);
                        mutex.unlock();
                        break;
                    }
                }
            }

            if (failedResponsesCounter == numberOfProcesses - 1) {
                MPI_Barrier(MPI_COMM_WORLD);
//                std::cout << "Rank: " << rank << " finishing all threads...\n";
                isRequesterInterrupted = true;
                executorCondVar.notify_all();
                int status = FINISH_PROCESS;
                MPI_Send(&status, 1, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);
            }
        } else {
            executorCondVar.notify_all();
        }
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

int CalculateTotalTasksComplexity(std::vector<task_complexity_t>& taskList) {
    int totalComplexity = 0;
    for (int i : taskList) {
        totalComplexity += i;
    }
    return totalComplexity;
}

std::vector<task_complexity_t> DistributeTasks(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    int* numberOfTasksArray = CreateNumberOfTasksArray(numberOfProcesses);
    /* [DEBUG] Distribution of tasks [DEBUG] */
//    if (rank == 0) {
//        for (int i = 0; i < numberOfProcesses; ++i) {
//            std::cout << "Rank <" << i << ">: " << numberOfTasksArray[i] << "\n";
//        }
//        std::cout << "BEFORE DISTRIBUTING:\n";
//        DebugPrintVector(taskList, rank);
//    }

    std::vector<task_complexity_t> temporaryList;
    task_complexity_t* tasks = new task_complexity_t[numberOfTasksArray[rank]];
    if (rank == 0) {
        for (int i = numberOfProcesses - 1; i >= 0; --i) {
            int requiredTasks = numberOfTasksArray[i];
            for (int j = 0; j < requiredTasks; ++j) {
                tasks[j] = taskList.back();
                taskList.pop_back();
            }
            MPI_Send(tasks, requiredTasks, MPI_INT, i, i, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(tasks, numberOfTasksArray[rank], MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }

    /* Create a vector from array */
    std::vector<task_complexity_t> distributedTaskList(tasks, tasks + numberOfTasksArray[rank]);
    std::cout << "Total complexity for rank: <" << rank << "> is: " << CalculateTotalTasksComplexity(distributedTaskList) << " sec.\n";

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
//        CreateRandomTaskList(taskList);
        CreateDeterminedTaskList(taskList, numberOfProcesses);
        std::cout << "Summary complexity: " << CalculateTotalTasksComplexity(taskList) << " sec.\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    taskList = DistributeTasks(taskList, rank, numberOfProcesses);
    /* [DEBUG] Check distribution of tasks [DEBUG]
    if (rank == numberOfProcesses - 1) {
        DebugPrintVector(taskList, rank);
    }
    */

    int tasksTimeTakenByProcess = 0;
    std::thread executorThread(ExecuteTask, std::ref(taskList), rank, std::ref(tasksTimeTakenByProcess));
    std::thread senderThread(SendTask, std::ref(taskList), rank);
    std::thread requesterThread(RequestTask, std::ref(taskList), rank, numberOfProcesses);

    executorThread.join();
    senderThread.join();
    requesterThread.join();

    std::cout << "Process with rank: <" << rank << "> slept summary " << tasksTimeTakenByProcess << " sec.\n";
    double end = MPI_Wtime();
    if (rank == numberOfProcesses - 1) {
        std::cout << "Elapsed time: " << end - start << "\n";
    }

    int allProcessesTime = 0;
    MPI_Allreduce(&tasksTimeTakenByProcess, &allProcessesTime, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == numberOfProcesses - 1) {
        std::cout << "[Total tasks time taken by all processes " << allProcessesTime << " sec.]\n";
    }

    CleanUp(taskList);

    MPI_Finalize();

    return 0;
}
