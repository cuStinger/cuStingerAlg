#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


namespace cuStingerAlgs{



// Typedef to vertex frontier expansion
typedef void (*cusSubKernelVertex)(cuStinger* custing,vertexId_t src,void* metadata);
// Typedef to vertex frontier expansion
typedef void (*cusSubKernelEdge)(cuStinger* custing,vertexId_t src,vertexId_t dst,void* metadata);



class StaticAlgorithm{
public:
	virtual void Init(cuStinger& custing) = 0;
	virtual void Reset() = 0;
	virtual void Run(cuStinger& custing) = 0;
	virtual void Release() = 0;

	virtual void SyncHostWithDevice() = 0;
	virtual void SyncDeviceWithHost() = 0;
};

template <class T> 
class Queue{
public:
	Queue(){
		alreadyInit=false;
		queueCurr=queueEnd=0;
	}

	Queue(length_t length){
		maxLen=length;
		queue = (T*) allocDeviceArray(maxLen, sizeof(T));
		alreadyInit=true;
		queueCurr=queueEnd=0;
		createDevQueue();
	}

	~Queue(){
		if (alreadyInit){
			freeDeviceArray(queue);
			freeDevQueue();
		}
	}

	void Init(length_t length){
		if(alreadyInit==true){
			if(length==maxLen)
				return;
			freeDeviceArray(queue);
		}
		alreadyInit=true;
		maxLen=length;
		queue = (T*) allocDeviceArray(maxLen, sizeof(T));
		queueCurr=queueEnd=0;
		createDevQueue();
	}

	__device__ void enqueue(T t){
		length_t prevPos = atomicAdd(&this->queueEnd,1);
		queue[prevPos] = t;		
	}

	void enqueueFromHost(T t){
		copyArrayHostToDevice(&t,queue+queueEnd,1,sizeof(t));
		queueEnd++;
	}

	T* getQueue(){return queue;}
	__host__ __device__	length_t getQueueEnd(){return queueEnd;}
	__host__ __device__ length_t getQueueCurr(){return queueCurr;}

	void setQueueCurr(length_t curr) { queueCurr = curr; }

	void setQueueEnd(length_t end) { queueEnd = end; }

	void resetQueue(){queueCurr=queueEnd=0;}
	length_t getActiveQueueSize(){return queueEnd-queueCurr;}

	T* getQueueAtCurr(){return queue+queueCurr;}
	T* getQueueAtEnd(){return queue+queueEnd;}

	static void swapQueues(Queue* q1, Queue* q2){
		Queue temp;
		temp.queue=q1->queue; temp.queueCurr=q1->queueCurr; temp.queueEnd=q1->queueEnd;
		q1->queue=q2->queue; q1->queueCurr=q2->queueCurr; q1->queueEnd=q2->queueEnd;
		q2->queue=temp.queue; q2->queueCurr=temp.queueCurr; q2->queueEnd=temp.queueEnd;
	}

	Queue<T>* devPtr(){return devQueue;}

	void SyncHostWithDevice(){
		copyArrayDeviceToHost(devQueue,this,1, sizeof(Queue<T>));
	}
	void SyncDeviceWithHost(){
		copyArrayHostToDevice(this,devQueue,1, sizeof(Queue<T>));
	}


private:
	T* queue;
	length_t queueCurr;
	length_t queueEnd;

	length_t maxLen;

	bool alreadyInit=false;

	Queue<T>* devQueue;
	void createDevQueue(){
		devQueue = (Queue<T>*)allocDeviceArray(1, sizeof(Queue<T>));
		SyncDeviceWithHost();

	}
	void freeDevQueue(){
		freeDeviceArray(devQueue);
	}


};


typedef Queue<vertexId_t> vertexQueue;

typedef struct{
	vertexId_t* queue;
	length_t queueCurr;
	length_t queueEnd;
	private:
	length_t maxLen;
} queue;

}
