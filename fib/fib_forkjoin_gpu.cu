// GPU上でfork-joinを実行する

#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_TASKS       16384   // 生成されるタスクの上限
#define MAX_TASKS_PER_BLOCK 512 // ブロック内でのタスクの最大数
#define QUEUE_SIZE      8192   // タスクキューのサイズ
#define MAX_CHILD_TASKS 32     // 子タスクの最大数
#define STACK_SIZE      4      // 1タスクあたりのスタック領域のサイズ
#define NUM_BLOCKS      128    // ブロック数
#define THREADS_PER_BLK 128    // ブロック内のスレッド数
#define DATA_LENGTH     1000   // ブロック内で共有するデータのサイズ，重い操作のためのデータ

// タスクのステージを表す
enum State : int {
    STATE_START = 0,
    STATE_AFTER_CHILD1 = 1,
};

// タスク構造体
struct Task {
    int   parentId;                        // 親タスク ID
    int   child_ids[MAX_CHILD_TASKS];      // 子タスクのIDたち
    /* TODO: MAX_CHILD_TASKSを撤廃して動的配列にできない？Task[]を宣言しているから現状厳しい */
    int   joinCount;                       // 子タスク残数
    int   n;                               // フィボナッチ数列のn
    /* TODO: nを渡すような形ではなく，関数を渡すような形にできる？一般のTask構造体が何を持ってるか調べないといけない */
    State state;                           // どのステージにいるか，たとえば同期1後のタスクはSTATE_AFTER_CHILD1にいる
    /* TODO: 以下2つは未実装 */
    // int   stack[STACK_SIZE];               // スタック領域，join後に使う変数を保存する
    // int   sp;                              // stack[] の中でのトップオフセット
};

// グローバル領域
__device__ Task  d_tasks[MAX_TASKS];    // タスクの配列
__device__ int   d_results[MAX_TASKS];  // タスクの返り値の配列
/* NOTE: 返り値が一定の型であることを仮定している */
__device__ int   d_queue[QUEUE_SIZE];   // タスクキュー，実行可能なタスクのid（配列d_tasks上の位置）を保存する
/* TODO: 優先度を考慮したキューにするのもありかもしれない */
__device__ int   d_head, d_tail, d_pendingTasks, d_taskId; // キューのhead，tail，未処理タスク数，タスクID

// キューにタスクIDをenqueueし，pendingTasksを増やす
__device__ void enqueue(int taskId) {
    int pos = atomicAdd(&d_tail, 1) % QUEUE_SIZE;
    d_queue[pos] = taskId;
    atomicAdd(&d_pendingTasks, 1);
}

// 親タスクを再開させる用：pendingTasksは増やさない
__device__ void enqueueResumableTask(int taskId) {
    int pos = atomicAdd(&d_tail, 1) % QUEUE_SIZE;
    d_queue[pos] = taskId;
    // d_pendingTasksは増やさない
}

// キューからタスクIDをdequeueできればtrueを返す
__device__ bool dequeue(int &taskId) {
    int old_head, tail;
    do {
        old_head = d_head;
        tail = d_tail;
        if (old_head >= tail) return false; // キューが空なのでdequeueできない
    } while (atomicCAS(&d_head, old_head, old_head + 1) != old_head);
    taskId = d_queue[old_head % QUEUE_SIZE];
    return true;
}

// 親タスクのjoinCountをデクリメントし，0なら再enqueueする
__device__ void notifyParent(int parentId) {
    int rem = atomicSub(&d_tasks[parentId].joinCount, 1);
    /* NOTE: atomicSubは引く前の値を返すことに注意 */
    if (rem == 1) {
        // すべての子が完了したら親をタスクキューへ
        enqueueResumableTask(parentId);
    }
}

/* TODO: 以下は未実装 */
// // スタックフレームとして退避するローカル変数の例
// struct FibFrame {
//     int n;
// };
// // スタック上にフレームをプッシュ
// __device__ FibFrame* pushFrame(Task &t) {
//   // アライメントを考慮しつつスペースを確保
//   const int frameSize = sizeof(FibFrame);
//   int new_sp = t.sp + frameSize;
//   if (new_sp > STACK_SIZE) {
//     // オーバーフロー時はエラー（実装上は十分大きく取るか、
//     // プールから外部割り当てにフォールバック）
//     return nullptr;
//   }
//   FibFrame* frame = reinterpret_cast<FibFrame*>(t.stack + t.sp);
//   t.sp = new_sp;
//   return frame;
// }

// // スタック上のフレームをポップ
// __device__ FibFrame* popFrame(Task &t) {
//   const int frameSize = sizeof(FibFrame);
//   int old_sp = t.sp - frameSize;
//   if (old_sp < 0) return nullptr;
//   t.sp = old_sp;
//   return reinterpret_cast<FibFrame*>(t.stack + old_sp);
// }

// ブロック内の全スレッドで協調して重い操作を行う例
__device__ void heavy_operation() {
    __shared__ int sdata[DATA_LENGTH]; // ここはshared memoryも使える
    for (int i = threadIdx.x; i < DATA_LENGTH; i += blockDim.x) {
        sdata[i] = i;
    }
}

// タスクを実行する関数
__device__ void TaskFunction(Task t, int shared_tid) {
    if (t.n < 2) {
        heavy_operation();
        d_results[shared_tid] = t.n;  // 結果をd_resultsに保存
        if (threadIdx.x == 0) {
            notifyParent(t.parentId);
            atomicSub(&d_pendingTasks, 1);
        }
    } else {
        switch (t.state) {
            case STATE_START: {
                if (threadIdx.x == 0) {
                    int child1_id = atomicAdd(&d_taskId, 2);
                    // 子タスク1のIDチェック
                    if (child1_id >= MAX_TASKS) {
                        printf("FATAL ERROR: Task limit exceeded after child1 creation (ID: %d, MAX: %d)\n", 
                               child1_id, MAX_TASKS);
                        __trap();
                    }
                    Task child1;
                    child1.parentId = shared_tid;
                    child1.state = STATE_START;
                    child1.joinCount = 0;
                    child1.n = t.n - 1;
                    Task child2;
                    child2.parentId = shared_tid;
                    child2.state = STATE_START;
                    child2.joinCount = 0;
                    child2.n = t.n - 2;
                    d_tasks[child1_id] = child1;
                    d_tasks[child1_id + 1] = child2;

                    // 親タスクの情報を更新
                    // これを子タスクをキューに追加する前にやらないと，子タスクが未更新の親タスクを実行してしまう可能性がある
                    d_tasks[shared_tid].joinCount = 2;
                    d_tasks[shared_tid].state = STATE_AFTER_CHILD1;
                    d_tasks[shared_tid].child_ids[0] = child1_id;
                    d_tasks[shared_tid].child_ids[1] = child1_id + 1;

                    // 子タスクをキューに追加
                    enqueue(child1_id);
                    enqueue(child1_id + 1);
                }
                break;
            }
            case STATE_AFTER_CHILD1: {
                // ここでjoinする
                // 子タスクの結果を取得
                if (threadIdx.x == 0) {
                    int child1_result = d_results[t.child_ids[0]];
                    int child2_result = d_results[t.child_ids[1]];
                    d_results[shared_tid] = child1_result + child2_result;
                }
                heavy_operation();
                if (threadIdx.x == 0) {
                    notifyParent(t.parentId);
                    atomicSub(&d_pendingTasks, 1);
                }
                break;
            }
            default: {
                break;
            }
        }
    }
}

// __device__ bool steal(int &tid_tmp) {
//     // ここでstealする
//     return false;
// }

__global__ void persistentKernel() {
    __shared__ int shared_tid; // ブロック内で共有するタスクID

    while (atomicAdd(&d_pendingTasks, 0) > 0) {
        // 代表スレッドがタスクIDを取得
        if (threadIdx.x == 0) {
            int tid_tmp;
            if (!dequeue(tid_tmp)) {
                shared_tid = -1; // タスクがなければ-1
            } else {
                shared_tid = tid_tmp;
            }
        }

        __syncthreads();
        if (shared_tid == -1) continue;

        Task t = d_tasks[shared_tid];
        TaskFunction(t, shared_tid);
        // __syncthreads();
    }
}

// 最初のタスクをキューに追加するためのカーネル
__global__ void enqueueKernel() {
    enqueue(0);  // タスクID 0 をキューに追加
}

// メイン関数
int main() {
    // CUDAデバイスの初期化
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // 初期タスクの設定
    Task initialTask;
    initialTask.parentId = 0;
    initialTask.joinCount = 0;
    initialTask.n = 18;
    initialTask.state = STATE_START;

    // デバイスメモリに初期タスクをコピー
    cudaStatus = cudaMemcpyToSymbol(d_tasks, &initialTask, sizeof(Task), 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed!");
        return 1;
    }

    // グローバル変数の初期化
    int initialValues[] = {0, 0, 0, 1};  // d_head, d_tail, d_pendingTasks, d_taskId
    cudaStatus = cudaMemcpyToSymbol(d_head, &initialValues[0], sizeof(int), 0);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "Global variable initialization failed! (d_head)\n"); return 1; }
    cudaStatus = cudaMemcpyToSymbol(d_tail, &initialValues[1], sizeof(int), 0);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "Global variable initialization failed! (d_tail)\n"); return 1; }
    cudaStatus = cudaMemcpyToSymbol(d_pendingTasks, &initialValues[2], sizeof(int), 0);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "Global variable initialization failed! (d_pendingTasks)\n"); return 1; }
    cudaStatus = cudaMemcpyToSymbol(d_taskId, &initialValues[3], sizeof(int), 0);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "Global variable initialization failed! (d_taskId)\n"); return 1; }

    // 最初のタスクをキューに追加
    enqueueKernel<<<1, 1>>>();
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Initial enqueue failed!");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    persistentKernel<<<NUM_BLOCKS, THREADS_PER_BLK>>>();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // カーネルの完了を待つ
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed!");
        return 1;
    }

    // 結果を取得
    int result;
    cudaStatus = cudaMemcpyFromSymbol(&result, d_results, sizeof(int), 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Result retrieval failed!");
        return 1;
    }

    printf("Fibonacci(%d) = %d\n", initialTask.n, result);
    printf("Kernel execution time: %.3f ms (%.6f s)\n", milliseconds, milliseconds/1000.0);

    // クリーンアップ
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
