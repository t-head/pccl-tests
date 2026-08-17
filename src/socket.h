/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PCCL_SOCKET_H_
#define PCCL_SOCKET_H_

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "pccl.h"

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT            1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES   2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES    3 // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)

/* Common socket address storage structure for IPv4/IPv6 */
union pcclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum pcclSocketState {
  pcclSocketConnecting = 0,
  pcclSocketConnected = 1,
  pcclSocketError = 2,
  pcclSocketStateNum = 3
};

enum pcclMsgType {
  pcclBarrier = 0,
  pcclAllreduce = 1
};

struct pcclSocket {
  int fd;
  union pcclSocketAddress addr;
  volatile uint32_t* abortFlag;
  int asyncFlag;
  enum pcclSocketState state;
};

struct socketMsg {
  pcclMsgType msgId;
  int rank;
  double msgContent;
};

struct netIf {
  char prefix[64];
  int port;
};

#define TRACE(FLAGS, ...)
#define INFO(FLAGS, ...) printf(__VA_ARGS__)
#define WARN(...) printf(__VA_ARGS__)
#define ERROR(...) printf(__VA_ARGS__)
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    ERROR("Call to " name " failed : %s\n", strerror(errno)); \
    return pcclSystemError; \
  } \
} while (false)

#define SYSCHECKGOTO(call, name, res, label) do { \
  int retval; \
  SYSCHECKVALGOTO(call, name, retval, res, label); \
} while (false)

#define SYSCHECKVALGOTO(call, name, retval, res, label) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    ERROR("Call to " name " failed : %s\n", strerror(errno)); \
    res = pcclSystemError; \
    goto label; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(PCCL_ALL,"Call to " name " returned %s, retrying\n", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define PCCLCHECK(call) do { \
  pcclResult_t res = call; \
  if (res != pcclSuccess) { \
    /* Print the back trace*/ \
    ERROR("%s:%d -> %d\n", __FILENAME__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#define PCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != pcclSuccess) { \
    /* Print the back trace*/ \
    ERROR("%s:%d -> %d\n", __FILENAME__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    INFO(PCCL_ALL,"%s:%d -> %d\n", __FILE__, __LINE__, pcclSystemError);    \
    return pcclSystemError;     \
  }                             \
} while (0);

#define NEQCHECKGOTO(statement, value, res, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    res = pcclSystemError;    \
    INFO(PCCL_ALL,"%s:%d -> %d\n", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    INFO(PCCL_ALL,"%s:%d -> %d\n", __FILE__, __LINE__, pcclSystemError);    \
    return pcclSystemError;     \
  }                             \
} while (0);

#define EQCHECKGOTO(statement, value, res, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    res = pcclSystemError;    \
    INFO(PCCL_ALL,"%s:%d -> %d\n", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

static int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

/* Format a string representation of a (struct sockaddr *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline const char *pcclSocketToString(struct sockaddr *saddr, char *buf) {
  if (buf == NULL || saddr == NULL) return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void) getnameinfo(saddr, sizeof(union pcclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

static inline uint16_t socketToPort(struct sockaddr *saddr) {
  return ntohs(saddr->sa_family == AF_INET ? ((struct sockaddr_in*)saddr)->sin_port : ((struct sockaddr_in6*)saddr)->sin6_port);
}

static uint16_t socketToPort(union pcclSocketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  char* env = getenv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;

  INFO(PCCL_ENV, "NCCL_SOCKET_FAMILY set by environment to %s\n", env);

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static pcclResult_t getFdState(int fd, enum pcclSocketState* state) {
    struct pollfd pfd;
    int timeout = 1, ret;
    socklen_t rlen = sizeof(int);

    memset(&pfd, 0, sizeof(struct pollfd));
    pfd.fd = fd;
    pfd.events = POLLOUT;
    SYSCHECK(ret = poll(&pfd, 1, timeout), "poll");
    if (ret == 0) {
      ret = EINPROGRESS;
    } else {
      /* check socket status */
      EQCHECK(ret == 1 && (pfd.revents & POLLOUT), 0);
      SYSCHECK(getsockopt(fd, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen), "getsockopt");
    }

    if (ret == EINPROGRESS)
      *state = pcclSocketConnecting;
    else if (ret == 0)
      *state = pcclSocketConnected;
    else
      *state = pcclSocketError;
    return pcclSuccess;
}

static pcclResult_t pcclSocketListen(struct pcclSocket* sock) {
  /* IPv4/IPv6 support */
  int family = sock->addr.sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  int flags;

  /* Create socket and bind it to a port */
  int fd = socket(family, SOCK_STREAM, 0);
  if (fd == -1) {
    WARN("Net : Socket creation failed : %s\n", strerror(errno));
    return pcclSystemError;
  }

  if (socketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#else
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  /* make all new sockets non-blocking */
  EQCHECK(flags = fcntl(fd, F_GETFL), -1);
  SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");

  // addr port should be 0 (Any port)
  SYSCHECK(bind(fd, &sock->addr.sa, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  SYSCHECK(getsockname(fd, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(PCCL_INIT|PCCL_NET,"Listening on socket %s", pcclSocketToString(&sock->addr.sa, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(fd, 16384), "listen");
  sock->fd = fd;
  return pcclSuccess;
}

static pcclResult_t pcclSocketConnect(struct pcclSocket* sock, int blocking=0) {
  char line[SOCKET_NAME_MAXLEN+1];
  /* IPv4/IPv6 support */
  int family = sock->addr.sa.sa_family;
  if (family != AF_INET && family != AF_INET6) {
    WARN("Net : connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)\n",
         pcclSocketToString(&sock->addr.sa, line), family, AF_INET, AF_INET6);
    return pcclInternalError;
  }
  int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  int flags;

  /* Connect to a hostname / port */
  int fd = socket(family, SOCK_STREAM, 0);
  if (fd == -1) {
    WARN("Net : Socket creation failed : %s\n", strerror(errno));
    return pcclSystemError;
  }

  const int one = 1;
  SYSCHECK(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");

  if (!blocking) {
    /* support non-blocking socket; by default, the socket is non-blocking */
    EQCHECK(flags = fcntl(fd, F_GETFL), -1);
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

  TRACE(PCCL_INIT|PCCL_NET,"Connecting to socket %s", pcclSocketToString(&sock->addr.sa, line));

  int ret;
  int timedout_retries = 0;
  int refused_retries = 0;
retry:
  /* async connect; abort when error happens and abortFlag is present. */
  ret = connect(fd, &sock->addr.sa, salen);

  if (errno == EAGAIN || (errno == ECONNREFUSED && ++refused_retries < RETRY_REFUSED_TIMES) ||
    (errno == ETIMEDOUT && ++timedout_retries < RETRY_TIMEDOUT_TIMES)) {
    if (refused_retries % 1000 == 0) INFO(PCCL_ALL, "Call to connect returned %s, retrying\n", strerror(errno));
    usleep(SLEEP_INT);
    goto retry;
  } else if (errno == EINPROGRESS && !sock->asyncFlag) {
    enum pcclSocketState state;
    do {
      if (sock->abortFlag) NEQCHECK(*sock->abortFlag, 0);
      PCCLCHECK(getFdState(fd, &state));
    } while (state == pcclSocketConnecting);
    EQCHECK(state, pcclSocketError);
    ret = 0;
  } else if (errno == EISCONN) {
    // client could get error "Transport endpoint is already connected", treate it as success.
    ret = 0;
  }

  if (ret == 0 || (errno == EINPROGRESS && sock->asyncFlag)) {
    sock->fd = fd;
    return pcclSuccess;
  }

  WARN("Net : Connect to %s failed : %s\n", pcclSocketToString(&sock->addr.sa, line), strerror(errno));
  return pcclSystemError;
}

static pcclResult_t pcclGetSocketAddrFromString(union pcclSocketAddress* ua, const char* ip_port_pair, int barrier_port=-1) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    ERROR("Net : string is null\n");
    return pcclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      ERROR("Net : No valid <IPv4_or_hostname>:<port> pair found\n");
      return pcclInvalidArgument;
    }

    if (barrier_port != -1) ni.port = barrier_port;

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      ERROR("Net : error encountered when getting address info : %s\n", gai_strerror(rv));
      return pcclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family\n");
      freeaddrinfo(p); // all done with this structure
      return pcclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      ERROR("Net : No valid [IPv6]:port pair found\n");
      return pcclInvalidArgument;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);
    strncpy(port_str, ip_port_pair+i+2, len-i-1);
    int port = barrier_port == -1 ? atoi(port_str) : barrier_port;
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return pcclSuccess;
}

static pcclResult_t createListenSocket(int *fd, union pcclSocketAddress *localAddr) {
  pcclResult_t res = pcclSuccess;
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
  socklen_t size = salen;

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
  if (sockfd == -1) {
    ERROR("Net : Socket creation failed : %s\n", strerror(errno));
    return pcclSystemError;
  }

  if (socketToPort(&localAddr->sa)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    SYSCHECKGOTO(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt", res, fail);
#else
    SYSCHECKGOTO(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt", res, fail);
#endif
  }

  // localAddr port should be 0 (Any port)
  SYSCHECKGOTO(bind(sockfd, &localAddr->sa, salen), "bind", res, fail);

  /* Get the assigned Port */
  SYSCHECKGOTO(getsockname(sockfd, &localAddr->sa, &size), "getsockname", res, fail);

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(PCCL_INIT|PCCL_NET,"Listening on socket %s", pcclSocketToString(&localAddr->sa, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  SYSCHECKGOTO(listen(sockfd, 16384), "listen", res, fail);
  *fd = sockfd;
exit:
  return res;
fail:
  if (sockfd >= 0) close(sockfd);
  *fd = -1;
  goto exit;
}

static pcclResult_t connectAddress(int* fd, union pcclSocketAddress* remoteAddr) {
  /* IPv4/IPv6 support */
  int family = remoteAddr->sa.sa_family;
  if (family != AF_INET && family != AF_INET6) {
    ERROR("Error : connecting to address with family %d is neither AF_INET(%d) nor AF_INET6(%d)\n", family, AF_INET, AF_INET6);
    return pcclInternalError;
  }
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Connect to a hostname / port */
  *fd = socket(family, SOCK_STREAM, 0);
  if (*fd == -1) {
    ERROR("Net : Socket creation failed : %s\n", strerror(errno));
    return pcclSystemError;
  }

  const int one = 1;
  SYSCHECK(setsockopt(*fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(PCCL_INIT|PCCL_NET,"Connecting to socket %s", pcclSocketToString(&remoteAddr->sa, line));

  int ret;
  int timedout_retries = 0;
  int refused_retries = 0;
retry:
  SYSCHECKSYNC(connect(*fd, &remoteAddr->sa, salen), "connect", ret);
  if (ret == 0) return pcclSuccess;
  if ((errno == ECONNREFUSED || errno == ETIMEDOUT)) {
    if ((errno == ECONNREFUSED && ++refused_retries < RETRY_REFUSED_TIMES) ||
        (errno == ETIMEDOUT && ++timedout_retries < RETRY_TIMEDOUT_TIMES)) {
      if (refused_retries % 1000 == 0) INFO(PCCL_ALL,"Call to connect returned %s, retrying\n", strerror(errno));
      usleep(SLEEP_INT);
      goto retry;
    }
  }
  ERROR("Connect to %s failed : %s\n", pcclSocketToString(&remoteAddr->sa, line), strerror(errno));
  return pcclSystemError;
}

static pcclResult_t pcclSocketAccept(struct pcclSocket* sock, struct pcclSocket* listenSocket) {
  int family = listenSocket->addr.sa.sa_family;
  socklen_t socklen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
  int tmpFd = sock->fd = -1;

  do {
    if (listenSocket->abortFlag) NEQCHECK(*listenSocket->abortFlag, 0);
    tmpFd = accept(listenSocket->fd, &sock->addr.sa, &socklen);
  } while ((errno == EAGAIN || errno == EWOULDBLOCK) && tmpFd == -1 && !listenSocket->asyncFlag);

  if (!listenSocket->asyncFlag) {
    EQCHECK(tmpFd, -1);
  } else if (tmpFd == -1 && errno != EAGAIN && errno != EWOULDBLOCK) {
    return pcclSystemError;
  }

  sock->fd = tmpFd;
  return pcclSuccess;
}

#define PCCL_SOCKET_SEND 0
#define PCCL_SOCKET_RECV 1
static pcclResult_t pcclSocketProgressOpt(int op, int fd, void* ptr, int size, int* offset, int block, int *closed) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;
  do {
    if (op == PCCL_SOCKET_RECV) bytes = recv(fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    if (op == PCCL_SOCKET_SEND) bytes = send(fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    if (op == PCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return pcclSuccess;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        ERROR("Call to recv failed : %s\n", strerror(errno));
        return pcclSystemError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
  } while (bytes > 0 && (*offset) < size);
  return pcclSuccess;
}

static pcclResult_t pcclSocketProgress(int op, int fd, void* ptr, int size, int* offset, int block=0) {
  int closed;
  PCCLCHECK(pcclSocketProgressOpt(op, fd, ptr, size, offset, block, &closed));
  if (closed) {
    WARN("Net : Connection closed by remote peer");
    return pcclSystemError;
  }
  return pcclSuccess;
}

static pcclResult_t pcclSocketWait(int op, int fd, void* ptr, int size, int* offset) {
  while (*offset < size)
    PCCLCHECK(pcclSocketProgress(op, fd, ptr, size, offset, 1));
  return pcclSuccess;
}

static pcclResult_t pcclSocketSend(int fd, void* ptr, int size) {
  int offset = 0;
  PCCLCHECK(pcclSocketWait(PCCL_SOCKET_SEND, fd, ptr, size, &offset));
  return pcclSuccess;
}

static pcclResult_t pcclSocketRecv(int fd, void* ptr, int size) {
  int offset = 0;
  PCCLCHECK(pcclSocketWait(PCCL_SOCKET_RECV, fd, ptr, size, &offset));
  return pcclSuccess;
}

// Receive or detect connection closed
static pcclResult_t pcclSocketTryRecv(int fd, void* ptr, int size, int* closed) {
  int offset = 0;
  *closed = 0;
  while (offset < size) {
    PCCLCHECK(pcclSocketProgressOpt(PCCL_SOCKET_RECV, fd, ptr, size, &offset, 0, closed));
    if (*closed) return pcclSuccess;
  }
  return pcclSuccess;
}

static pcclResult_t pcclBarrierAccept(int listenFd, int* recvFd) {
  struct sockaddr_in pcclSockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(listenFd, (struct sockaddr*)&pcclSockaddr, &socklen), "accept", *recvFd);
  return pcclSuccess;
}

#endif
