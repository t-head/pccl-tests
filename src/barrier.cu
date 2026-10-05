/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "socket.h"
#include "barrier.h"
#include <pthread.h>
#include <assert.h>

#define BARRIER_PORT 54321
static bool barInit = false;
static struct pcclSocket localSocket;
static struct socketMsg localMsg;
static struct pcclSocket *barrierSockets = nullptr;
static pthread_t barrierThread;
static int average = 1;
static struct threadArgs {
  struct pcclSocket *sock;
  int worldSize;
  volatile int stop;
} Args;

static int getBarrierPort() {
  char *env = getenv("NCCL_BARRIER_PORT");
  return env ? atoi(env) : BARRIER_PORT;
}

static pcclResult_t pcclBarrierBcast(struct pollfd *pollfds, int worldSize, struct socketMsg *msg) {
  // send back for each rank
  for (int s = 0; s < worldSize; s++) {
    msg->rank = s;
    PCCLCHECK(pcclSocketSend(pollfds[s].fd, msg, sizeof(struct socketMsg)));
  }

  return pcclSuccess;
}

static void* pcclBarrierService(void* args) {
  struct threadArgs *_args = (struct threadArgs*) args;
  struct pcclSocket *listenSock = _args->sock;
  const int worldSize = _args->worldSize;

  // get listen fd
  pcclSocketListen(listenSock);

  // Prepare poll descriptor
  struct pollfd *pollfds = (struct pollfd *)malloc((worldSize + 1) * sizeof(struct pollfd));
  struct socketMsg recvMsg;
  struct socketMsg sendMsg;
  for (int s = 0; s < worldSize; s++) {
    pollfds[s].fd = -1;
    pollfds[s].events = POLLHUP | POLLIN;
  }
  pollfds[worldSize].fd = listenSock->fd;
  pollfds[worldSize].events = POLLIN;

  int barrier = 0;
  int *barrierIds = (int *)malloc(worldSize * sizeof(int));
  for (int i = 0; i < worldSize; i++) {
    barrierIds[i] = -1;
  }

  int valueCount = 0;
  double value = 0.0;

  while (!_args->stop) {
    // polling connection
    int error = 0;
    do {
      error = poll(pollfds, worldSize+1, 100/*ms*/);
    } while (error < 0 && errno == EINTR);
    if (error  < 0) {
      ERROR("[Barrier Thread] Poll failed with error %d\n", error);
      return NULL;
    }

    // comming connection
    if (pollfds[worldSize].revents) {
      int s = 0;
      while (s < worldSize && pollfds[s].fd != -1) s++;
      if (pcclBarrierAccept(pollfds[worldSize].fd, &pollfds[s].fd) != pcclSuccess) {
        pollfds[s].fd = -1;
      }
      //WARN("conn at %d\n", s);
    }

    // polling barrier msg
    for (int s = 0; s < worldSize; s++) {
      // remote disconnect
      if (pollfds[s].revents & POLLHUP) {
        close(pollfds[s].fd);
        pollfds[s].fd = -1;
      }

      // barrier mesg
      if (pollfds[s].revents & POLLIN) {
        int closed = 0;
        pcclResult_t ret = pcclSocketTryRecv(pollfds[s].fd, &recvMsg, sizeof(struct socketMsg), &closed);
        if (ret != pcclSuccess || closed == 1) {
          close(pollfds[s].fd);
          pollfds[s].fd = -1;
        } else {
          if (recvMsg.msgId == pcclBarrier) {
            if (barrierIds[recvMsg.rank] == -1) {
              barrier++;
              barrierIds[recvMsg.rank] = pollfds[s].fd;
            } else {
              ERROR("invalid barrier msg from rank: %d\n", recvMsg.rank);
              assert(0); // unlikely here, if so abort process.
            }
          } else if (recvMsg.msgId == pcclAllreduce) {
            switch(average) {
              case /*avg*/1: value += recvMsg.msgContent; break;
              case /*min*/2: value = value < 1e-2 ? recvMsg.msgContent :
                                     value < recvMsg.msgContent ? value : recvMsg.msgContent; break;
              case /*max*/3: value = value > recvMsg.msgContent ? value : recvMsg.msgContent; break;
              case /*sum*/4: value += recvMsg.msgContent; break;
            }
            valueCount++;
          } else {
            ERROR("invalid msgId: %d from rank: %d\n", recvMsg.msgId, recvMsg.rank);
            assert(0);
          }
        }
      }
    }

    if (barrier == worldSize) {
      sendMsg.msgId = pcclBarrier;
      sendMsg.msgContent = worldSize;
      pcclBarrierBcast(pollfds, worldSize, &sendMsg);
      for (int i = 0; i < worldSize; i++) {
        barrierIds[i] = -1;
      }

      barrier = 0;
    } else if (valueCount == worldSize) {
      sendMsg.msgId = pcclAllreduce;
      sendMsg.msgContent = value;
      pcclBarrierBcast(pollfds, worldSize, &sendMsg);

      valueCount = 0;
      value = 0.0;
    }
  }

  // loop end
  for (int s = 0; s < worldSize; s++) {
    if (pollfds[s].fd != -1)
      close(pollfds[s].fd);
  }

  if (pollfds) free(pollfds);
  if (barrierIds) free(barrierIds);
  close(listenSock->fd);
  return NULL;
}

int barrierInit(int rank, int worldSize) {
  // skip single node
  if (worldSize < 2) return pcclSuccess;

  // obtain COMM_ID
  char* commId = getenv("NCCL_COMM_ID");
  if (!commId) {
    ERROR("Missing NCCL_COMM_ID\n");
    return pcclInvalidUsage;
  }

  union pcclSocketAddress idAddr;
  if (pcclGetSocketAddrFromString(&idAddr, commId, getBarrierPort()) != pcclSuccess) {
    ERROR("pcclGetSocketAddrFromString Failed\n");
    return pcclInvalidArgument;
  }

  if (rank == 0) {
    // root rank, need listen
    if (!barrierSockets) {
      barrierSockets = (struct pcclSocket *)malloc((worldSize + 1) * sizeof(struct pcclSocket));
      if (!barrierSockets) {
        ERROR("No Memory for barrierSockets\n");
        return pcclSystemError;
      }
    }
    memcpy(&barrierSockets[worldSize].addr, &idAddr, sizeof(union pcclSocketAddress));
    Args.sock = &barrierSockets[worldSize];
    Args.worldSize = worldSize;
    Args.stop = 0;
    pthread_create(&barrierThread, NULL, pcclBarrierService, &Args);
  }

  // connect to barrier thread
  localSocket.asyncFlag = 0;
  memcpy(&localSocket.addr, &idAddr, sizeof(union pcclSocketAddress));
  PCCLCHECK(pcclSocketConnect(&localSocket, 1));
  localMsg.rank = rank;

  barInit = true;
  return pcclSuccess;
}

int barrier(int rank, int worldSize) {
  if (!barInit) return pcclSuccess;
  localMsg.msgId = pcclBarrier;
  localMsg.msgContent = (double)rank;
  PCCLCHECK(pcclSocketSend(localSocket.fd, &localMsg, sizeof(struct socketMsg)));
  PCCLCHECK(pcclSocketRecv(localSocket.fd, &localMsg, sizeof(struct socketMsg)));
  return pcclSuccess;
}

int allreduceFromSocket(double *value, int rank, int avg) {
  if (!barInit) return pcclSuccess;
  average = avg;
  localMsg.msgId = pcclAllreduce;
  localMsg.msgContent = *value;
  PCCLCHECK(pcclSocketSend(localSocket.fd, &localMsg, sizeof(struct socketMsg)));
  PCCLCHECK(pcclSocketRecv(localSocket.fd, &localMsg, sizeof(struct socketMsg)));
  *value = localMsg.msgContent;
  return pcclSuccess;
}

void barrierDeinit(int rank, int worldSize) {
  if (!barInit) return;
  if (rank == 0) {
    Args.stop = 1;
    pthread_join(barrierThread, NULL);
    if (barrierSockets) free(barrierSockets);
  }
  close(localSocket.fd);
}
