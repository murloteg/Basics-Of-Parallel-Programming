#include <iostream>
#include <random>
#include <mpi.h>
#include <vector>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

/* Task describes by integer value (time to sleep) */
typedef int task_complexity_t;

enum Consts {
    MIN_TASK_COMPLEXITY = 1,
    MAX_TASK_COMPLEXITY = 5,
    NUMBER_OF_TASKS = 125
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

void ExecuteTask(std::vector<task_complexity_t>& taskList, int rank, int& timeTakenByProcess) {
    while (!isExecutorInterrupted) {
        std::unique_lock<std::mutex> uniqueLock(mutex);
        executorCondVar.wait(uniqueLock);
        if (!taskList.empty()) {
            task_complexity_t taskComplexity = taskList.back();
            taskList.pop_back();
//            std::cout << "Rank: " << rank << " executing task with complexity: " << taskComplexity << "\n";
            sleep(taskComplexity);
            timeTakenByProcess += taskComplexity;
        }
        uniqueLock.unlock();

        if (isRequesterInterrupted) {
            isExecutorInterrupted = true;
        }
    }
}

void SendTask(std::vector<task_complexity_t>& taskList) {
    while (true) {
        int requesterRank;
        MPI_Recv(&requesterRank, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        if (requesterRank == FINISH_PROCESS) {
            break;
        }
        
        task_complexity_t task = EMPTY_TASK;
        mutex.lock();
        if (!taskList.empty()) {
            task = taskList.back();
            taskList.pop_back();
        }
        mutex.unlock();
        MPI_Send(&task, 1, MPI_INT, requesterRank, RESPONSE_TAG, MPI_COMM_WORLD);
    }
}

void RequestTask(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    while (!isRequesterInterrupted) {
        bool isTaskListEmpty = false;
        mutex.lock();
        if (taskList.empty()) {
            isTaskListEmpty = true;
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

int GenerateRandomTask() {
    std::random_device device;
    std::mt19937 range(device());
    std::uniform_int_distribution<std::mt19937::result_type> distribution(1, MAX_TASK_COMPLEXITY);
    return (int) distribution(range);
}

/* The first mode in the first scenario: absolutely random task list */
void CreateRandomTaskList(std::vector<task_complexity_t>& taskList) {
    for (int i = 0; i < NUMBER_OF_TASKS; ++i) {
        taskList.push_back(GenerateRandomTask());
    }
}

/* The second mode in the first scenario: task list depends on the number of processes */
void CreateIncreasingTaskList(std::vector<task_complexity_t>& taskList, int numberOfProcesses) {
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

int CalculateOptimalNumberOfComplexities() {
    int maxComplexity = MAX_TASK_COMPLEXITY;
    while (NUMBER_OF_TASKS % maxComplexity != 0) {
        --maxComplexity;
    }
    return maxComplexity;
}

/* The third mode in the first scenario: task list depends only on the number of tasks */
void CreateIndependentTaskList(std::vector<task_complexity_t>& taskList) {
    int taskComplexity = MIN_TASK_COMPLEXITY;
    int numberOfComplexities = CalculateOptimalNumberOfComplexities();
    int requiredTasks = NUMBER_OF_TASKS / numberOfComplexities;
    while (taskList.size() < NUMBER_OF_TASKS) {
        for (int i = 0; i < requiredTasks; ++i) {
            taskList.push_back(taskComplexity);
        }
        taskComplexity += 1;
    }
}

bool IsEvenNumberOfProcesses(int numberOfProcesses) {
    return (numberOfProcesses % 2 == 0);
}

int CalculateOffsetFromMiddle(int rank, int middleRank) {
    int offset = std::abs(middleRank - rank);
    if (rank < middleRank) {
        offset *= (-1);
    }
    return offset;
}

/* The first mode in the second scenario: every process creates task list */
void CreateProcessTaskList(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    int requiredTasks = NUMBER_OF_TASKS / numberOfProcesses;
    if (rank < NUMBER_OF_TASKS % numberOfProcesses) {
        ++requiredTasks;
    }
    int middle = (int) round(numberOfProcesses / 2);
    int offset = CalculateOffsetFromMiddle(rank, middle);
    if (IsEvenNumberOfProcesses(numberOfProcesses)) {
        if (rank == middle) {
            offset += 1;
        }
    }
    for (int i = 0; i < requiredTasks + offset; ++i) {
        taskList.push_back((int) round(GenerateRandomTask() / (rank + 1)));
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

void PrepareFirstScenario(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    if (rank == 0) {
//        CreateRandomTaskList(taskList);
//        CreateIncreasingTaskList(taskList, numberOfProcesses);
        CreateIndependentTaskList(taskList);
        std::cout << "[SCENARIO 1] Summary complexity: " << CalculateTotalTasksComplexity(taskList) << " sec.\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    taskList = DistributeTasks(taskList, rank, numberOfProcesses);
}

void PrepareSecondScenario(std::vector<task_complexity_t>& taskList, int rank, int numberOfProcesses) {
    CreateProcessTaskList(taskList, rank, numberOfProcesses);
    int processTasksTime = CalculateTotalTasksComplexity(taskList);
    std::cout << "[SCENARIO 2] Summary complexity for rank: <" << rank << "> is " << processTasksTime << " sec.\n";

    int totalTasksTime = 0;
    MPI_Allreduce(&processTasksTime, &totalTasksTime, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "[SCENARIO 2] Summary complexity: " << totalTasksTime << " sec.\n";
    }
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

//    PrepareFirstScenario(taskList, rank, numberOfProcesses);
    PrepareSecondScenario(taskList, rank, numberOfProcesses);

    double start = MPI_Wtime();
    int timeTakenByProcess = 0;
    std::thread executorThread(ExecuteTask, std::ref(taskList), rank, std::ref(timeTakenByProcess));
    std::thread senderThread(SendTask, std::ref(taskList));
    std::thread requesterThread(RequestTask, std::ref(taskList), rank, numberOfProcesses);

    executorThread.join();
    senderThread.join();
    requesterThread.join();

    double end = MPI_Wtime();
    std::cout << "Process with rank: <" << rank << "> slept summary " << timeTakenByProcess << " sec.\n";
    if (rank == 0) {
        std::cout << "Elapsed time: " << end - start << " sec.\n";
    }

    int timeTakenByAllProcesses = 0;
    MPI_Allreduce(&timeTakenByProcess, &timeTakenByAllProcesses, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "[Total tasks time taken by all processes: " << timeTakenByAllProcesses << " sec.]\n";
    }

    CleanUp(taskList);

    MPI_Finalize();

    return 0;
}
