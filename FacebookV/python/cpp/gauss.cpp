#include <math.h>

#ifdef WIN
	#include <windows.h>
	BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID)
	{
    		return TRUE;
	}
	#define CALLSPEC __declspec (dllexport)
#else
	#define CALLSPEC
#endif

extern "C" CALLSPEC double gauss(
		    double mu1, double mu2,//means
		    double s1, double s12, double s2,//covariances
		    double x1, double x2)//coordinates
{
	double detS = s1*s2-s12*s12;
	if (detS > 0) {
		return exp(-0.5*(mu2*mu2*s1-2.0*mu1*mu2*s12+mu1*mu1*s2+2.0*mu2*s12*x1-2.0*mu1*s2*x1+s2*x1*x1-2.0*mu2*s1*x2+2.0*mu1*s12*x2-2.0*s12*x1*x2+s1*x2*x2)/detS)/(2.0*M_PI*sqrt(detS));
	} else {
		return 0;
	}
};
