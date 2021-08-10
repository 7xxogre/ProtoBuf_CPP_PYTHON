#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <Windows.h>
#include <string>
#include <sstream>
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <direct.h>
#include <queue>
#include <thread>
#include <chrono>
#include <mutex>

#include "client.h"

#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024

using namespace std;


Client::Client(const char* host, short port_) {
    serv_addr = host;
    port = port_;

    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    inet_pton(AF_INET, serv_addr, &target.sin_addr);

    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
}

int Client::Connect() {
    if (sock == INVALID_SOCKET)
    {
        return -1;
    }

    if (connect(sock, (SOCKADDR*)&target, sizeof(SOCKADDR_IN)) == SOCKET_ERROR) {
        return -1;
    }
    return 0;
}

int Client::send_msg(string input) {
    int n;
    n = send(sock, input.data(), input.size(), 0);
    return n;
}

string Client::recv_msg() {
    char buffer[BUFSIZE];
    memset(&buffer, 0, BUFSIZE);
    recv(sock, buffer, BUFSIZE, 0);
    return buffer;
}

void Client::slice_msg(string input)
{
    char str_buff[200];
    strcpy_s(str_buff, input.c_str());
    char* context = NULL;
    char* ptr = strtok_s(str_buff, "?",&context);
    int cnt = 0;
    while (ptr)
    {   if(cnt == 0)
        {   
            diagnosis_result = ptr;
        }
        else 
        {
            result_path = ptr;
        }
        ptr = strtok_s(NULL, "?",&context);
        cnt++;
    }
}

//void Client::slice_str(string input)
//{
//    char str_buff[128];
//    strcpy_s(str_buff, input.c_str());
//    char* context = NULL; 
//    char* ptr = strtok_s(str_buff, "?", &context);
//    header = ptr;
//}

//return reulst_path(str)  path of result image made by server 
string Client::get_path()
{
    return result_path;
}

//return Client diagnosis_result(str)  Normal,Abnormal
string Client::get_result()
{
    return diagnosis_result;
}

void Client::close() {
    send_msg("finish");
    closesocket(sock);
}

