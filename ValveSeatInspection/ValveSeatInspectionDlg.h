
// ValveSeatInspectionDlg.h: 헤더 파일
//

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/opencv.hpp"
//#pragma comment (lib, "opencv_world440")

using namespace cv;
using namespace std;

// CValveSeatInspectionDlg 대화 상자
class CValveSeatInspectionDlg : public CDialogEx
{
// 생성입니다.
public:
	CValveSeatInspectionDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_VALVESEATINSPECTION_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.
		

// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
	//BITMAPINFO *m_pBitmapInfo;
	CImage result_img; //result image

public:
	afx_msg void OnBnClickedBtnCamopen();
	
	int resize_width; // bmp 폭을 4배수로 변환

	Mat m_matImage;  //mat image 임시저장
	BITMAPINFO* m_pBitmapInfo;
	Point prCenterPos, prEndPos;

	//hci add  function
	void CreateBitmapInfo(int w, int h, int bpp);
	void DrawImage();
	void DisplayImage(Mat& _targetMat, int viewer_index);


	Mat m_Mframe_1;                  //0번 카레라 정보
//--- INITIALIZE VIDEOCAPTURE
	VideoCapture m_Vcap_1;
	bool m_bVideoRunning;


	// 영상 쓰레드 
	void ShowCamera_1();
	afx_msg void OnBnClickedBtnCapture();
	afx_msg void OnBnClickedBtnSavepath();
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	afx_msg void OnBnClickedBtnApply();

	int m_iRLengh;
	CButton m_BTN_CAPTURE;
	afx_msg void OnBnClickedTest();
	afx_msg void OnBnClickedBtnInit();
	afx_msg void OnBnClickedFindCenter();
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnStnClickedMyPicture();
	afx_msg void OnStnClickedOrgPicture();
	afx_msg void OnStnClickedResultImg();
	CStatic img_box1;
	CButton 파라미터변경;
	afx_msg void OnBnClickedButton4();
};


extern CValveSeatInspectionDlg* pFormView;