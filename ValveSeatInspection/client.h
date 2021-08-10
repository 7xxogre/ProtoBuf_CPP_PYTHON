#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <Windows.h>
#include <string>
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <direct.h>
#include <queue>
#include <thread>
#include <chrono>
#include <mutex>

#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024

using namespace std;

class Client {

private:
    string diagnosis_result;
    string result_path;
    

public:

    const char* serv_addr;
    short port;
    SOCKADDR_IN target;
    SOCKET sock;
    char* header;
    /*string change_param(int p1, int p2);*/

    Client(const char* host = "127.0.0.1", short port_ = 3070);

    int Connect();

    int send_msg(string input);

    //string send_path(string input);
    
    void slice_msg(string);

    /*void slice_str(string input);*/
    string recv_msg();

    string get_path();

    string get_result();

    void close();
};

