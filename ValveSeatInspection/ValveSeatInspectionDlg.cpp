// ValveSeatInspectionDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "ValveSeatInspection.h"
#include "ValveSeatInspectionDlg.h"
#include "afxdialogex.h"
#include <afxmt.h>
//hci add
#include <thread>
#include <mutex>
#include <string>
#include <queue>
#include "client.h"

//hci's yumin add (for read file in data directory and send to python server (Sleep))
#include <windows.h>

/*
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
*/
using namespace cv;

#define PI CV_PI
#define DEG2RAD PI/180
#define RAD2DEG 180/PI


//hci add
//추가 변수와 함수
mutex mu;
queue<string> qu;


//hci add
//string change_param(string p,string option_param)
//{
//	string str1 = "chg_param?";
//	string option = "";
//	//stringstream param;
//
//	//param << p;
//
//	str1 = str1 + option_param;
//	str1 = str1 + "?" + p;
//	//str1 = str1 + "?" + param.str();
//
//	return str1;
//}


//hci add
string send_path(string input)
{
	char str1[] = "path?";
	strcat_s(str1, input.c_str());
	string input_path(str1);
	//send_msg(input_path);
	return input_path;
}

// CValveSeatInspectionDlg 대화 상자
CValveSeatInspectionDlg* pFormView = NULL;

//hci add

UINT cthread(LPVOID pParam) {
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		cout << "[client]WSAStartup error" << endl;
		return 0;
	}

	Client C = Client();
	string confirm_msg;

	C.Connect();
	confirm_msg = C.recv_msg();
	cout << "[client]from server : " << confirm_msg << endl;

	string path;
	string result;
	Mat result_img;
	bool start = false;
	while (true) {
		mu.lock();
		//g_mutex.Lock();
		if (qu.empty()) {
			mu.unlock();
			continue;
		}
		else {
			path = qu.front();
			qu.pop();
			//mu.unlock();
			mu.unlock();
			if (!start)
			{
				start = true;
				C.send_msg(String("start1"));	//신호
			}
			
			CFileFind finder;
			CString CstrPathName(path.c_str());

			BOOL bWorking = finder.FindFile(CstrPathName + "\\*.jpg");

			while (bWorking)
			{
				bWorking = finder.FindNextFile();
				if (finder.IsDirectory() || finder.IsDots())
					continue;
				CString _fileName = finder.GetFileName();

				CT2CA pszConvertedAnsiString(_fileName);
				std::string fileName(pszConvertedAnsiString);
				C.send_msg(String(path + "\\" + fileName + "?"));			// ?를 구분자로 사용
				Sleep(200);
			}

			bWorking = finder.FindFile(CstrPathName + "\\*.bmp");

			while (bWorking)
			{
				bWorking = finder.FindNextFile();
				if (finder.IsDirectory() || finder.IsDots())
					continue;
				CString _fileName = finder.GetFileName();

				CT2CA pszConvertedAnsiString(_fileName);
				std::string fileName(pszConvertedAnsiString);
				C.send_msg(String(path + "\\" + fileName + "?"));			// ?를 구분자로 사용
				Sleep(200);
			}


			// C.send_msg(path);
			//C.send_msg(path);	//경로 보내는거
			


			//result = C.recv_msg();	//결과 받는거
			//if(result =="pramchange")
			//{
			//	continue;
			//}
			//C.slice_msg(result);
			//string paths = C.get_path();
			//result = C.get_result();
			//result_img = imread(paths);
			


			//resize(result_img, result_img, Size(720, 860));
			//pFormView->DisplayImage(result_img, 1);
			
		}
		//g_mutex.Unlock();
	}
	WSACleanup();
	return 0;
}

CValveSeatInspectionDlg::CValveSeatInspectionDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_VALVESEATINSPECTION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	//hci add : new thread
	CWinThread* cth;
	cth = AfxBeginThread(cthread, this);
	cth->m_bAutoDelete = TRUE;
}

void CValveSeatInspectionDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_BTN_CAPTURE, m_BTN_CAPTURE);
	DDX_Control(pDX, IDC_ORG_PICTURE, img_box1);
	DDX_Control(pDX, IDC_BTN_INIT, 파라미터변경);
}

BEGIN_MESSAGE_MAP(CValveSeatInspectionDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_CAMOPEN, &CValveSeatInspectionDlg::OnBnClickedBtnCamopen)
	ON_BN_CLICKED(IDC_BTN_CAPTURE, &CValveSeatInspectionDlg::OnBnClickedBtnCapture)
	ON_BN_CLICKED(IDC_BTN_SavePath, &CValveSeatInspectionDlg::OnBnClickedBtnSavepath)
	ON_BN_CLICKED(IDC_BTN_APPLY, &CValveSeatInspectionDlg::OnBnClickedBtnApply)
	ON_BN_CLICKED(IDCBTN_TEST, &CValveSeatInspectionDlg::OnBnClickedTest)
	ON_BN_CLICKED(IDC_BTN_INIT, &CValveSeatInspectionDlg::OnBnClickedBtnInit)
	ON_BN_CLICKED(IDCBTN_FIND_CENTER, &CValveSeatInspectionDlg::OnBnClickedFindCenter)
	ON_BN_CLICKED(IDC_BUTTON1, &CValveSeatInspectionDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON3, &CValveSeatInspectionDlg::OnBnClickedButton3)
	ON_STN_CLICKED(IDC_MY_PICTURE, &CValveSeatInspectionDlg::OnStnClickedMyPicture)
	ON_STN_CLICKED(IDC_ORG_PICTURE, &CValveSeatInspectionDlg::OnStnClickedOrgPicture)
	ON_STN_CLICKED(IDC_RESULT_IMG, &CValveSeatInspectionDlg::OnStnClickedResultImg)
	ON_BN_CLICKED(IDC_BUTTON4, &CValveSeatInspectionDlg::OnBnClickedButton4)
END_MESSAGE_MAP()


// CValveSeatInspectionDlg 메시지 처리기

BOOL CValveSeatInspectionDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.



	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	pFormView = this;
	m_iRLengh = 100;
	m_BTN_CAPTURE.EnableWindow(false);
	GetDlgItem(IDC_ED_SAVEPATH)->SetWindowTextW(_T("D:"));


	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CValveSeatInspectionDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

void CValveSeatInspectionDlg::DisplayImage(cv::Mat& _targetMat, int viewer_index)
{
	CDC* pDC;

	if (viewer_index == 1)
	{
		pDC = GetDlgItem(IDC_ORG_PICTURE)->GetDC();
	}
	else
	{
		pDC = GetDlgItem(IDC_RESULT_IMG)->GetDC();
	}
	cv::Mat tempImage;  // 화면용
	//resize_width = (int)(_targetMat.cols * 1/*m_dResizeRatio*/);
	//if (resize_width % 4) {
	//	resize_width -= resize_width % 4; resize_width += 4;
	//}
	flip(_targetMat, _targetMat, 0);
	BITMAPINFO bitmapInfo;
	bitmapInfo.bmiHeader.biYPelsPerMeter = 0;
	bitmapInfo.bmiHeader.biBitCount = 24;
	bitmapInfo.bmiHeader.biWidth = _targetMat.cols;
	bitmapInfo.bmiHeader.biHeight = _targetMat.rows;
	bitmapInfo.bmiHeader.biPlanes = 1;
	bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfo.bmiHeader.biCompression = BI_RGB;
	bitmapInfo.bmiHeader.biClrImportant = 0;
	bitmapInfo.bmiHeader.biClrUsed = 0;
	bitmapInfo.bmiHeader.biSizeImage = 200;
	bitmapInfo.bmiHeader.biXPelsPerMeter = 0;

	if (_targetMat.channels() != 3)
	{
		tempImage = _targetMat.clone();
	}

	// 영상 비율 계산 및 반영
	CRect rect;
	if (viewer_index == 1) {
		GetDlgItem(IDC_ORG_PICTURE)->GetClientRect(&rect);
		//GetDlgItem(IDC_ORG_PICTURE)->MoveWindow(10, 10, 1008, 720);
	}
	else
	{
		GetDlgItem(IDC_RESULT_IMG)->GetClientRect(&rect);
		//GetDlgItem(IDC_RESULT_IMG)->MoveWindow(10, 10, 840, 720);
	}
	
	

	pDC->SetStretchBltMode(COLORONCOLOR);
	StretchDIBits(pDC->GetSafeHdc(), rect.left, rect.top, rect.right, rect.bottom,
		0, 0, _targetMat.cols, _targetMat.rows, _targetMat.data,
		&bitmapInfo, DIB_RGB_COLORS, SRCCOPY);

	_targetMat.release();
	ReleaseDC(pDC);
}



// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CValveSeatInspectionDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

DWORD lpThreadId_1 = 1;
HANDLE hThreadCamera_1 = NULL;

DWORD WINAPI showImgSeq_Thread(LPVOID param)
{
	pFormView->ShowCamera_1();
	return 0;
}

void CValveSeatInspectionDlg::OnBnClickedBtnCamopen()
{
	m_bVideoRunning = !m_bVideoRunning; //버튼 클릭시 마다 변경
	if (m_bVideoRunning)
	{
		GetDlgItem(IDC_BTN_CAMOPEN)->SetWindowTextW(_T("Disconnect"));
		//m_Vcap_1.open(0);
		// OR advance usage: select any API backend
		int deviceID = 0;             // 0 = open default camera
		int apiID = cv::CAP_ANY;      // 0 = autodetect default API
									  // open selected camera using selected API
		m_Vcap_1.open(deviceID);
		// check if we succeeded
		if (!m_Vcap_1.isOpened()) {
			std::cerr << "ERROR! Unable to open camera\n";
		}
		m_Vcap_1.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
		m_Vcap_1.set(cv::CAP_PROP_FRAME_HEIGHT, 720); 
		m_BTN_CAPTURE.EnableWindow(true);

		hThreadCamera_1 = CreateThread(NULL, 0, showImgSeq_Thread, this, 0, &lpThreadId_1);
	}
	else
	{
		m_BTN_CAPTURE.EnableWindow(false);
		GetDlgItem(IDC_BTN_CAMOPEN)->SetWindowTextW(_T("Connect"));
	}
}

void CValveSeatInspectionDlg::ShowCamera_1()
{
	for (;;)
	{
		m_Vcap_1.read(m_Mframe_1);
		Mat temp = m_Mframe_1.clone();


		Point center = (Point)temp.size() / 2;
		circle(temp, center, m_iRLengh, Scalar(0, 0, 255));

		Point CrossX_1, CrossX_2;
		Point CrossY_1, CrossY_2;
		CrossX_1 = Point(center.x - 10, center.y);
		CrossX_2 = Point(center.x + 10, center.y);

		CrossY_1 = Point(center.x , center.y-10);
		CrossY_2 = Point(center.x, center.y + 10);

		line(temp, CrossX_1, CrossX_2, Scalar(0, 0, 255));
		line(temp, CrossY_1, CrossY_2, Scalar(0, 0, 255));

		DisplayImage(temp, 1);
		// check if we succeeded
		if (m_Mframe_1.empty()) {
			std::cerr << "ERROR! blank frame grabbed\n";
		}
		// show live and wait for a key with timeout long enough to show images
	//		imshow("Live", frame);
		if (m_bVideoRunning == false)
		{
			return;
		}
	}
}


//변경사항 존재
void CValveSeatInspectionDlg::OnBnClickedBtnCapture()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	SYSTEMTIME tm;
	GetLocalTime(&tm);
	
	int nYear = 32;
	CString strTemp;
	GetDlgItem(IDC_ED_SAVEPATH)->GetWindowTextW(strTemp);
	strTemp.Format(_T("%s\\%02d%02d_%02d%02d%02d.bmp"), strTemp,tm.wMonth ,tm.wDay,tm.wHour, tm.wMinute, tm.wSecond);
	m_Vcap_1.read(m_Mframe_1);

	//CStirng -> String
	CT2CA pszConvertedAnsiString(strTemp);
	string str(pszConvertedAnsiString);

	cv::imwrite(str, m_Mframe_1);

	//hci add
	//queue에 파일경로를 추가
	mu.lock();
	//g_mutex.Lock();
	str = send_path(str);
	qu.push(str);
	//g_mutex.Unlock();
	mu.unlock();
}

//hci
void CValveSeatInspectionDlg::OnBnClickedBtnSavepath()
{
	BROWSEINFO BrInfo;
	TCHAR szBuffer[512];                                      // 경로저장 버퍼 

	::ZeroMemory(&BrInfo, sizeof(BROWSEINFO));
	::ZeroMemory(szBuffer, 512);

	BrInfo.hwndOwner = GetSafeHwnd();
	BrInfo.lpszTitle = _T("파일이 저장될 폴더를 선택하세요");
	BrInfo.ulFlags = BIF_NEWDIALOGSTYLE | BIF_EDITBOX | BIF_RETURNONLYFSDIRS;
	LPITEMIDLIST pItemIdList = ::SHBrowseForFolder(&BrInfo);
	::SHGetPathFromIDList(pItemIdList, szBuffer);				// 파일경로 읽어오기
	if (pItemIdList != NULL)
	{
		CString str;
		str.Format(_T("%s"), szBuffer);
		GetDlgItem(IDC_ED_SAVEPATH)->SetWindowText(str);
	}
}


BOOL CValveSeatInspectionDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
	if (pMsg->message == WM_KEYDOWN)
	{
		if (pMsg->wParam == VK_RETURN)
		{
			CString round;
			GetDlgItem(IDC_ED_Round)->GetWindowText(round);
			m_iRLengh = _ttoi(round);
			return TRUE;
		}
	}

	return CDialogEx::PreTranslateMessage(pMsg);
}


void CValveSeatInspectionDlg::OnBnClickedBtnApply()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString round;
	GetDlgItem(IDC_ED_Round)->GetWindowText(round);
	m_iRLengh = _ttoi(round);
}

//hci add
//이미지경로 넘겨주는 이벤트
void CValveSeatInspectionDlg::OnBnClickedTest()
{
	Mat m_matImageOrigin;
	Mat m_matImage;
	CString path;

	CFileDialog fileDlg(TRUE, NULL, NULL, OFN_READONLY, _T("image file(*.jpg;*.bmp;*.png;)|*.jpg;*.bmp;*.png;|All Files(*.*)|*.*||"));
	if (fileDlg.DoModal() == IDOK)
	{
		path = fileDlg.GetPathName();
		std::cout << "check\n";
		CT2CA pszString(path);
		std::string strPath(pszString);
		mu.lock();
		//g_mutex.Lock();
		strPath = send_path(strPath);
		qu.push(strPath);
		//g_mutex.Unlock();
		mu.unlock();

		
	}
	
}


void CValveSeatInspectionDlg::OnBnClickedBtnInit()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	Mat m_matImageOrigin;
	Mat m_matImage;
	Mat src, gray;
	Scalar blue(255, 0, 0), green(0, 255, 0), red(0, 0, 255);

	//	Point prCenterPos = Point(1280 / 2 - 7 + 20, 720 / 2 + 10), prEndPos;
	int fBeginDegree = 0;
	int fEndDegree = 360;
	float x, y;
	float fBeginRadius = 350, fEndRadius = 1;   // 밖에서 안으로
	float step;
	float fFinalRadius = 0;
	vector<Vec3f> circles;
	//path = "D:\\valve_seat\\21.01.18\\정상2\\0118_144902.bmp";
	//CT2CA pszString(path);
	//std::string strPath(pszString);
	//m_matImageOrigin = imread(strPath, IMREAD_UNCHANGED);

	String path("D:\\식산\\정상\\*.bmp"); // *.jpg						// jpg 확장자 파일만 읽음
	vector<String> str;
	// 이미지 저장을 위한 변수
	int index = 0;
	char buf[256];
	glob(path, str, false); // 파일 목록을 가져오는 glob 함수

	for (int cnt = 0; cnt < str.size(); cnt++)
	{
		// 칼라
		src = imread(str[cnt], IMREAD_UNCHANGED);
		m_matImageOrigin = src.clone();
		cvtColor(src, gray, COLOR_BGR2GRAY);


		// Reduce the noise so we avoid false circle detection
		GaussianBlur(gray, gray, Size(15, 15), 2, 2);
		//		namedWindow("GaussianBlur", WINDOW_AUTOSIZE);
		//		imshow("GaussianBlur", gray);
		//imshow("GaussianBlur", gray); //waitKey(0);

		int th = 0;
		//th = threshold(gray, gray, 80, 255, THRESH_OTSU);  // ADAPTIVE_THRESH_MEAN_C 
		adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 2);   // ADAPTIVE_THRESH_GAUSSIAN_C
		//imshow("threshold", gray);// waitKey(0);
		//imwrite("d:\\threshold.bmp", gray);

		//Canny(gray, gray, 50, 150);
		//imshow("Canny", gray);


		cv::resize(gray, gray, Size(gray.cols / 2, gray.rows / 2));

		// Apply the Hough Transform to find the circles
//		HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 30, 200, 50, 50, 300);
//		HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 140, 50, 50, 80, 150);
		HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 150, 200, 50, 80, 150);


		src = gray;
		cvtColor(src, src, COLOR_GRAY2BGR);
		// Draw the circles detected
		prCenterPos = Point(1280 / 2 - 10, 720 / 2 - 10);
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);

			if (center.y > 720 / 2 * 0.9 && center.y < 720 / 2 * 1.1 && center.x > 1280 / 2 * 0.9 && center.x < 1280 / 2 * 1.1) {
				circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);// circle center     
				circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);// circle outline
				cout << "center : " << center << "\nradius : " << radius << endl;

				fFinalRadius = radius;
				prCenterPos = center;

				CString str;
				str.Format(_T("찾은 원의 radius = %d \n"), radius);
				OutputDebugString(str);
			}
		}

		// Show your results
		namedWindow("Hough Circle Transform Demo", WINDOW_AUTOSIZE);
		imshow("Hough Circle Transform Demo", src);

		//return;
		if (circles.size() != 1 && fFinalRadius == 0) continue;     // 원 중심이 어렷이거나 원 중심을 못 찾은 경우

		if (prCenterPos.y < fBeginRadius) {
			fBeginRadius = prCenterPos.y - 1;   // 밖에서 안으로
		}
		if (m_matImageOrigin.rows - prCenterPos.y < fBeginRadius) {
			fBeginRadius = m_matImageOrigin.rows - prCenterPos.y - 1;   // 밖에서 안으로
		}

		// 이미지를 펼침
		if (m_matImageOrigin.channels() == 3)
			cvtColor(m_matImageOrigin, m_matImageOrigin, COLOR_BGR2GRAY);
		Mat output = Mat::zeros(fBeginRadius - fEndRadius + 1, 3600, CV_8UC1);

		CString str;
		step = 0.1;
		int count = 0;
		float r;
		for (r = fBeginRadius; r >= fEndRadius; r--) {
			count = 0;
			for (float angle = fBeginDegree; angle <= fEndDegree; angle = angle + step) {
				x = cos(angle * DEG2RAD) * r;
				y = sin(angle * DEG2RAD) * r;
				prEndPos.x = prCenterPos.x + x;
				prEndPos.y = prCenterPos.y + y;


				output.at<char>(r - fEndRadius, count) = m_matImageOrigin.at<char>(prEndPos.y, prEndPos.x);
				count++;

			}
			//waitKey(0);
		}
		//line(m_matImageOrigin, prCenterPos, prEndPos, Scalar(r / 210 * 255, 0, 0), 1);


		Mat resize;
		//	cv::resize(output, resize, Size(1920, output.rows));
		cv::resize(output, resize, Size(1920, output.rows));



		imshow("resize", resize);
		imwrite("d:\\resize.bmp", resize);
		imwrite("d:\\src.bmp", src);
		//	imshow("m_matImageOrigin", m_matImageOrigin);
			//imwrite("d:\\m_matImageOrigin.bmp", m_matImageOrigin);

		waitKey(0);
		//Sleep(10);
	}
}


double Distance(const Point& p1, const Point& p2) {

	double distance;

	// 피타고라스의 정리
	// pow(x,2) x의 2승,  sqrt() 제곱근
	distance = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));

	return distance;
}


void CValveSeatInspectionDlg::OnBnClickedFindCenter()
{

	Mat m_matImageOrigin;
	Mat m_matImage;
	Mat src, gray;
	Scalar blue(255, 0, 0), green(0, 255, 0), red(0, 0, 255);
	
//	Point prCenterPos = Point(1280 / 2 - 7 + 20, 720 / 2 + 10), prEndPos;
	int fBeginDegree = 0;
	int fEndDegree = 360;
	float x, y;
	float step;
	float fFinalRadius = 0;
	//path = "D:\\valve_seat\\21.01.18\\정상2\\0118_144902.bmp";
	//CT2CA pszString(path);
	//std::string strPath(pszString);
	//m_matImageOrigin = imread(strPath, IMREAD_UNCHANGED);
	
	String path("D:\\식산\\정상\\*.bmp"); // *.jpg						// jpg 확장자 파일만 읽음
	vector<String> str;
	// 이미지 저장을 위한 변수
	int index = 0;
	char buf[256];
	glob(path, str, false); // 파일 목록을 가져오는 glob 함수
					
	for (int cnt = 0; cnt < str.size(); cnt++)
	{
		float fBeginRadius = 350, fEndRadius = 1;   // 밖에서 안으로

		vector<Vec3f> circles;
		fFinalRadius = 0;
		prCenterPos = Point(0, 0);//Point(1280/2 / 2, 720/2 / 2);
			
		// 칼라
		src = imread(str[cnt], IMREAD_UNCHANGED);
		m_matImageOrigin = src.clone();
		cvtColor(src, gray, COLOR_BGR2GRAY);
	
	
		blur(gray, gray, Size(5, 5));
		//Mat blur_ = gray.clone();
		//cvtColor(blur_, blur_, COLOR_GRAY2BGR);
		//imshow("blur", gray);

		/// <summary>
		/// 케니엣지
		/// </summary>
		Mat canny_output;
		int thr = 30;
		Canny(gray, canny_output, thr, thr * 1.5);
		dilate(canny_output, canny_output, Mat());
		//dilate(canny_output, canny_output, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 2);
		//imshow("canny_output", canny_output);
	
	
		cv::resize(canny_output, gray, Size(gray.cols/2, gray.rows/2));
	
		// Apply the Hough Transform to find the circles
//		HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 30, 200, 50, 50, 300);
//		HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 140, 50, 50, 80, 150);
		HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 10, 100, 30, 90/2, 110/2);
	
		// 칼라 원본 확인시 주석 달것
		cv::resize(gray, gray, Size(gray.cols*2, gray.rows*2)); src = gray;  cvtColor(src, src, COLOR_GRAY2BGR);
			
		// Draw the circles detected
	
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
	
			double ret = Distance(center*2, Point(1280 / 2, 720 / 2));

			//if (center.y > 720/2 / 2 * 0.9 && center.y < 720/2 / 2 * 1.1 && center.x > 1280/2 / 2 * 0.9 && center.x < 1280/2 / 2 * 1.1) {
			if (ret < 20 && radius*2 < 350) {
				circle(src, center*2, 3, Scalar(0, 255, 0), -1, 8, 0);// circle center     
				circle(src, center * 2, radius * 2, Scalar(0, 0, 255), 1, 8, 0);// circle outline
				circle(m_matImageOrigin, center * 2, radius * 2, Scalar(0, 0, 255), 1, 8, 0);// circle outline
				cout << "center : " << center << "\nradius : " << radius << endl;
	
				fFinalRadius = radius*2;
				prCenterPos = center*2;
				index++;
				/*CString str;
				str.Format(_T("radius = %d, center(%d,%d) \n"), fFinalRadius, prCenterPos.x, prCenterPos.y);
				OutputDebugString(str);*/

				CString str2;
				str2.Format(_T("[%d/%d | %f%%] radius = %f, center(%d,%d), dist=%d  \n"), cnt, str.size(), 100.0*index/cnt, fFinalRadius, prCenterPos.x, prCenterPos.y, (int)ret);
				OutputDebugString(str2);
			}
		}
	
		// Show your results
		namedWindow("Hough Circle Transform Demo", WINDOW_AUTOSIZE);
		imshow("Hough Circle Transform Demo", src);
	
		//return;
		imshow("m_matImageOrigin", m_matImageOrigin);

		if (circles.size() != 1 && fFinalRadius == 0) {
			waitKey(10);
			continue;     // 원 중심이 어렷이거나 원 중심을 못 찾은 경우
		}

		if (prCenterPos.y < fBeginRadius) {
			fBeginRadius = prCenterPos.y - 1;   // 밖에서 안으로
		}
		if (m_matImageOrigin.rows - prCenterPos.y < fBeginRadius) {
			fBeginRadius = m_matImageOrigin.rows - prCenterPos.y - 1;   // 밖에서 안으로
		}
	
		// 이미지를 펼침
		if (m_matImageOrigin.channels() == 3)
			cvtColor(m_matImageOrigin, m_matImageOrigin, COLOR_BGR2GRAY);
		Mat output;
		if (fFinalRadius != 0) {
			output  = Mat::zeros(fBeginRadius - fEndRadius + 1, 3600, CV_8UC1);
		}
		CString str;
		step = 0.1;
		int count = 0;
		float r;
		if (fFinalRadius != 0) {

			for (r = fBeginRadius; r >= fEndRadius; r--) {
				count = 0;
				for (float angle = fBeginDegree; angle <= fEndDegree; angle = angle + step) {
					x = cos(angle * DEG2RAD) * r;
					y = sin(angle * DEG2RAD) * r;
					prEndPos.x = prCenterPos.x + x;
					prEndPos.y = prCenterPos.y + y;


					output.at<char>(r - fEndRadius, count) = m_matImageOrigin.at<char>(prEndPos.y, prEndPos.x);
					count++;

				}
				//waitKey(0);
			}
			//line(m_matImageOrigin, prCenterPos, prEndPos, Scalar(r / 210 * 255, 0, 0), 1);


			Mat resize;
			//	cv::resize(output, resize, Size(1920, output.rows));
			cv::resize(output, resize, Size(1920, output.rows));



			imshow("resize", resize);
		}
		//imwrite("d:\\resize.bmp", resize);
		//imwrite("d:\\src.bmp", src);
		//	imshow("m_matImageOrigin", m_matImageOrigin);
			//imwrite("d:\\m_matImageOrigin.bmp", m_matImageOrigin);
	
		waitKey(10);
		//Sleep(10);
	}

	waitKey(0);


}

void CValveSeatInspectionDlg::OnBnClickedButton1()
{
	Mat m_matImageOrigin;
	Mat m_matImage;
	Mat src, gray;
	Scalar blue(255, 0, 0), green(0, 255, 0), red(0, 0, 255);

	//	Point prCenterPos = Point(1280 / 2 - 7 + 20, 720 / 2 + 10), prEndPos;
	int fBeginDegree = 0;
	int fEndDegree = 360;
	float x, y;
	float step;
	float fFinalRadius ;
	//path = "D:\\valve_seat\\21.01.18\\정상2\\0118_144902.bmp";
	//CT2CA pszString(path);
	//std::string strPath(pszString);
	//m_matImageOrigin = imread(strPath, IMREAD_UNCHANGED);

	String path("D:\\식산\\정상\\*.bmp"); // *.jpg						// jpg 확장자 파일만 읽음
	vector<String> str;
	// 이미지 저장을 위한 변수
	int index = 0;
	char buf[256];
	glob(path, str, false); // 파일 목록을 가져오는 glob 함수

	for (int cnt = 0; cnt < str.size(); cnt++)
	{
		float fBeginRadius = 350, fEndRadius = 1;   // 밖에서 안으로

		vector<Vec3f> circles;
		fFinalRadius = 0;
		prCenterPos = Point(0, 0);//Point(1280/2 / 2, 720/2 / 2);

		int thresh = 30;
		RNG rng(12345);

		// 칼라
		src = imread(str[cnt], IMREAD_UNCHANGED);
		m_matImageOrigin = src.clone();
		cvtColor(src, gray, COLOR_BGR2GRAY);

		blur(gray, gray, Size(5, 5));
		Mat blur_ = gray.clone();
		cvtColor(blur_, blur_, COLOR_GRAY2BGR);
		//imshow("blur", gray);

		/// <summary>
		/// 케니엣지
		/// </summary>
		Mat canny_output;
		Canny(gray, canny_output, thresh, thresh * 2);
		dilate(canny_output, canny_output, Mat());
		//dilate(canny_output, canny_output, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 2);
		imshow("canny_output", canny_output);

		/// <summary>
		/// 소벨엣지
		/// </summary>
		//Mat grad_x, grad_y;
		//Mat abs_grad_x, abs_grad_y;
		//Mat grad;
		//int ddepth = CV_16S;
		//Sobel(gray, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		//Sobel(gray, grad_y, ddepth, 0, 1, 3, 1, 0, BORDER_DEFAULT);
		//convertScaleAbs(grad_x, abs_grad_x);
		//convertScaleAbs(grad_y, abs_grad_y);
		//addWeighted(abs_grad_x, 0.8, abs_grad_y, 0.8, 0, grad);
		//canny_output = grad + 100;
		//imshow("sobel", grad);
		//threshold(canny_output, canny_output, 80, 255, THRESH_OTSU);  // ADAPTIVE_THRESH_MEAN_C 
		//imshow("adaptiveThreshold", canny_output);
		///// /////////////////////////////////////////////////////////////////////

		vector<vector<Point> > contours;
		findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f>centers(contours.size());
		vector<float>radius(contours.size());
		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(contours[i], contours_poly[i], 3, true);
			boundRect[i] = boundingRect(contours_poly[i]);
			minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
			//fitEllipse
		}
		Mat drawing = blur_; Mat::zeros(canny_output.size(), CV_8UC3);
		double old_ret = 100;
		for (size_t i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			drawContours(drawing, contours_poly, (int)i, color);
			//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 1);
		//	if ((int)radius[i] < 105 && (int)radius[i] > 95) {

			double ret = Distance(centers[i], Point(1280 / 2, 720 / 2));
			//double ret = Distance(boundRect[i].tlcenters[i], Point(1280 / 2, 720 / 2));

			if (ret < 10 && radius[i] < 200) {
				circle(drawing, centers[i], (int)radius[i], Scalar(0, 0, 255), 2);
				prCenterPos = centers[i];

				if (old_ret > ret) fFinalRadius = (float)radius[i];

				old_ret = ret;
				CString str2;
				str2.Format(_T("[%d/%d] radius = %f, center(%d,%d), dist=%d  \n"), cnt, str.size(), fFinalRadius, prCenterPos.x, prCenterPos.y, (int)ret);
				OutputDebugString(str2);
			}
			//	}
		}
		imshow("Contours", drawing);

		//waitKey(0); continue;


		////////////////////////////////////////////////////////////////////////////////
		//if (fFinalRadius == 0) continue;     // 원 중심이 어렷이거나 원 중심을 못 찾은 경우

		if (prCenterPos.y < fBeginRadius) {
			fBeginRadius = prCenterPos.y - 1;   // 밖에서 안으로
		}
		if (m_matImageOrigin.rows - prCenterPos.y < fBeginRadius) {
			fBeginRadius = m_matImageOrigin.rows - prCenterPos.y - 1;   // 밖에서 안으로
		}

		// 이미지를 펼침
		if (m_matImageOrigin.channels() == 3)
			cvtColor(m_matImageOrigin, m_matImageOrigin, COLOR_BGR2GRAY);
		Mat output;
		if (fFinalRadius != 0) {
			output = Mat::zeros(fBeginRadius - fEndRadius + 1, 3600, CV_8UC1);
		}
		CString str;
		step = 0.1;
		int count = 0;
		float r;

		if (fFinalRadius != 0) {
			for (r = fBeginRadius; r >= fEndRadius; r--) {
				count = 0;
				for (float angle = fBeginDegree; angle <= fEndDegree; angle = angle + step) {
					x = cos(angle * DEG2RAD) * r;
					y = sin(angle * DEG2RAD) * r;
					prEndPos.x = prCenterPos.x + x;
					prEndPos.y = prCenterPos.y + y;

					output.at<char>(r - fEndRadius, count) = m_matImageOrigin.at<char>(prEndPos.y, prEndPos.x);
					count++;
				}
				//waitKey(0);
			}
			//line(m_matImageOrigin, prCenterPos, prEndPos, Scalar(r / 210 * 255, 0, 0), 1);

			Mat resize;
			//	cv::resize(output, resize, Size(1920, output.rows));
			cv::resize(output, resize, Size(1920, output.rows));

			imshow("resize", resize);
			imwrite("d:\\resize.bmp", resize);
			//imwrite("d:\\src.bmp", src);
			//	imshow("m_matImageOrigin", m_matImageOrigin);
				//imwrite("d:\\m_matImageOrigin.bmp", m_matImageOrigin);
		}
		waitKey(0);
		//Sleep(10);
	}
}


//hci add
HBITMAP mat2bmp(cv::Mat* image)
{
	// 현재 응용프로그램의 스크린과 호환되는 memory dc를 생성한다.
	HDC         hDC = ::CreateCompatibleDC(0);
	BYTE        tmp[sizeof(BITMAPINFO) + 255 * sizeof(RGBQUAD)];
	BITMAPINFO* bmi = (BITMAPINFO*)tmp;
	HBITMAP     hBmp;
	int i;
	int w = image->cols, h = image->rows;
	int bpp = image->channels() * 8;

	memset(bmi, 0, sizeof(BITMAPINFO));
	bmi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi->bmiHeader.biWidth = w;
	bmi->bmiHeader.biHeight = h;
	bmi->bmiHeader.biPlanes = 1;
	bmi->bmiHeader.biBitCount = bpp;
	bmi->bmiHeader.biCompression = BI_RGB;
	bmi->bmiHeader.biSizeImage = w * h * 1;
	bmi->bmiHeader.biClrImportant = 0;

	switch (bpp)
	{
	case 8:
		for (i = 0; i < 256; i++)
		{
			bmi->bmiColors[i].rgbBlue = i;
			bmi->bmiColors[i].rgbGreen = i;
			bmi->bmiColors[i].rgbRed = i;
		}
		break;
	case 32:
	case 24:
		((DWORD*)bmi->bmiColors)[0] = 0x00FF0000; /* red mask  */
		((DWORD*)bmi->bmiColors)[1] = 0x0000FF00; /* green mask */
		((DWORD*)bmi->bmiColors)[2] = 0x000000FF; /* blue mask  */
		break;
	}

	hBmp = ::CreateDIBSection(hDC, bmi, DIB_RGB_COLORS, NULL, 0, 0);
	::SetBitmapBits(hBmp, image->total() * image->channels(), image->data);
	::DeleteDC(hDC);

	return hBmp;
}

//hci add
void CValveSeatInspectionDlg::CreateBitmapInfo(int w, int h, int bpp)
{
	

	if (m_pBitmapInfo != NULL)
	{
		delete m_pBitmapInfo;
		m_pBitmapInfo = NULL;
	}

	if (bpp == 8)
		m_pBitmapInfo = (BITMAPINFO*) new BYTE[sizeof(BITMAPINFO) + 255 * sizeof(RGBQUAD)];
	else // 24 or 32bit
		m_pBitmapInfo = (BITMAPINFO*) new BYTE[sizeof(BITMAPINFO)];

	m_pBitmapInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	m_pBitmapInfo->bmiHeader.biPlanes = 1;
	m_pBitmapInfo->bmiHeader.biBitCount = bpp;
	m_pBitmapInfo->bmiHeader.biCompression = BI_RGB;
	m_pBitmapInfo->bmiHeader.biSizeImage = 0;
	m_pBitmapInfo->bmiHeader.biXPelsPerMeter = 0;
	m_pBitmapInfo->bmiHeader.biYPelsPerMeter = 0;
	m_pBitmapInfo->bmiHeader.biClrUsed = 0;
	m_pBitmapInfo->bmiHeader.biClrImportant = 0;

	if (bpp == 8)
	{
		for (int i = 0; i < 256; i++)
		{
			m_pBitmapInfo->bmiColors[i].rgbBlue = (BYTE)i;
			m_pBitmapInfo->bmiColors[i].rgbGreen = (BYTE)i;
			m_pBitmapInfo->bmiColors[i].rgbRed = (BYTE)i;
			m_pBitmapInfo->bmiColors[i].rgbReserved = 0;
		}
	}

	m_pBitmapInfo->bmiHeader.biWidth = w;
	m_pBitmapInfo->bmiHeader.biHeight = -h;
}
//hci add
void CValveSeatInspectionDlg::DrawImage()
{
	
	CClientDC dc(GetDlgItem(IDC_RESULT_IMG));
	CRect rect;
	GetDlgItem(IDC_RESULT_IMG)->GetClientRect(&rect);
	SetStretchBltMode(dc.GetSafeHdc(), COLORONCOLOR);

	StretchDIBits(dc.GetSafeHdc(), 0, 0, rect.Width(), rect.Height(), 0, 0, m_matImage.cols,
		m_matImage.rows, m_matImage.data, m_pBitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

//hci add
void CValveSeatInspectionDlg::OnBnClickedButton3()
{
	Mat m_matImageOrigin;
	Mat m_matImage;
	CImage image_data;

	CString path;
	
	
	/*
	CFileDialog fileDlg(TRUE, NULL, NULL, OFN_READONLY, _T("image file(*.jpg;*.bmp;*.png;)|*.jpg;*.bmp;*.png;|All Files(*.*)|*.*||"));
	if (fileDlg.DoModal() == IDOK)
	{
		CString strPathName = fileDlg.GetPathName();
		CT2CA pszConvertedAnsiString(strPathName);
		// 파일 경로를 가져와 사용할 경우, Edit Control에 값 저장
		//SetDlgItemText(IDC_EDIT1, strPathName);
		std::string imagepath(pszConvertedAnsiString);

		//이미지 불러오기 완료
		//if (!m_matImageOrigin.data) MessageBox(_T("Image open error"));

		//server queue 에 넣기
		mu.lock();
		qu.push(imagepath);
		mu.unlock();

		// 계속 꺼지는데 m_matImage, resize, DisplayImage를 주석으로 만들면 꺼지지 않습니다
		//image_data.Load(strPathName);
		m_matImage = imread(imagepath, IMREAD_UNCHANGED);
		//Mat m_matImage2 = m_matImage(Range(720, 0), Range(280, 1000));
		//imshow("test",m_matImage);
		resize(m_matImage, m_matImage, Size(720, 1280));
		DisplayImage(m_matImage, 0);
	}
	*/
	

	// 폴더로 선택
	CFolderPickerDialog Picker(NULL, OFN_READONLY, NULL, 0);
	if (Picker.DoModal() == IDOK)
	{
		CString strPathName = Picker.GetPathName();
		CT2CA pszConvertedAnsiString(strPathName);
		std::string strFolderPath(pszConvertedAnsiString);

		mu.lock();
		qu.push(strFolderPath);
		mu.unlock();
	}

}


void CValveSeatInspectionDlg::OnStnClickedMyPicture()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CValveSeatInspectionDlg::OnStnClickedOrgPicture()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CValveSeatInspectionDlg::OnStnClickedResultImg()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}

//hci add
// server 계산 파라미터 수정
void CValveSeatInspectionDlg::OnBnClickedButton4()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	printf("일단 여기는 호출된다");
	CString thres_abnormal;
	CString thres_bright;
	CString thres_area;
	CString weight_area1;
	CString weight_area2;

	GetDlgItemText(IDC_EDIT3, thres_abnormal); //픽셀별 밝기
	GetDlgItemText(IDC_EDIT4, thres_bright);
	GetDlgItemText(IDC_EDIT5, thres_area);
	GetDlgItemText(IDC_EDIT6, weight_area1);
	GetDlgItemText(IDC_EDIT7, weight_area2);

	
	/*
	GetDlgItemText(IDC_EDIT3, thres_abnormal); //픽셀별 밝기
	string thres_bright = GetDlgItemText(IDC_EDIT4); //영역별 평균 밝기
	double thres_area = GetDlgItemInt(IDC_EDIT5); //영역의 픽셀 수
	double weight_area1 = GetDlgItemInt(IDC_EDIT6); // 1구간 가중치
	double weight_area2 = GetDlgItemInt(IDC_EDIT7); // 2구간 가중치
	*/

	//String thres_abnormal_str = change_param(String(CT2CA(thres_abnormal)), "thres_abnormal");
	/*String thres_bright_str = change_param(String(CT2CA(thres_bright)), "thres_bright");
	String thres_area_str = change_param(String(CT2CA(thres_area)), "thres_area");
	String weight_area1_str = change_param(String(CT2CA(weight_area1)), "weight_area1");
	String weight_area2_str = change_param(String(CT2CA(weight_area2)), "weight_area2");*/
	
	mu.lock();
	//qu.push(thres_abnormal_str);
	/*qu.push(thres_bright_str);
	qu.push(thres_area_str);
	qu.push(weight_area1_str);
	qu.push(weight_area2_str);*/
	mu.unlock();
	
}


