#include "mex.h"
#include "math.h"

void GJ(double *A, int rcnt, int ccnt, double tol)
{
    int r = 0;      // row
    int c = 0;      // col
    int k;
    int l;
    int dstofs;
    int srcofs;    
    int ofs = 0;
    int pofs = 0;
    double b;
    
    // gj
    ofs = 0;
    pofs = 0;
    while (r < rcnt && c < ccnt) {
        
        // find pivot
        double apivot = 0;
        double pivot = 0;
        int pivot_r = -1;
        
        pofs = ofs;
        for (k = r; k < rcnt; k++) {
            
            // pivot selection criteria here !
            if (fabs(*(A+pofs)) > apivot) {
                
                pivot = *(A+pofs);
                apivot = fabs(pivot);
                pivot_r = k;
            }
            pofs += ccnt;
        }
        
        if (apivot < tol) {
            
            // empty col - shift to next col (or jump)
            c++;
            ofs++;
            
        } else {
            
            // process rows

            // exchange pivot and selected rows
            // + divide row
            if (pivot_r == r) {

                srcofs = ofs;
                for (l = c; l < ccnt; l++) {

                    *(A+srcofs) = *(A+srcofs)/pivot;
                    srcofs++;
                }
                
            } else {
            
                srcofs = ofs;
                dstofs = ccnt*pivot_r+c;
                for (l = c; l < ccnt; l++) {

                    b = *(A+srcofs);
                    *(A+srcofs) = *(A+dstofs)/pivot;
                    *(A+dstofs) = b;

                    srcofs++;
                    dstofs++;
                }
            }            
            
            // zero bottom
            pofs = ofs + ccnt;
            for (k = r + 1; k < rcnt; k++) {
                
                    
                    // nonzero row
                    b = *(A+pofs);
                    dstofs = pofs + 1;
                    srcofs = ofs + 1;
                    for (l = c + 1; l < ccnt; l++) {
                        
                        *(A+dstofs) = (*(A+dstofs) - *(A+srcofs) * b);
                        dstofs++;
                        srcofs++;
                    }
                    *(A+pofs) = 0;

                pofs += ccnt;
            }
            
            // zero top
            pofs = c;
            for (k = 0; k < r; k++) {
                
                    
                    // nonzero row
                    b = *(A+pofs);
                    dstofs = pofs + 1;
                    srcofs = ofs + 1;
                    for (l = c + 1; l < ccnt; l++) {
                        
                        *(A+dstofs) = (*(A+dstofs) - *(A+srcofs) * b);
                        dstofs++;
                        srcofs++;
                    }
                    *(A+pofs) = 0;

                pofs += ccnt;
            }            

            r++;
            c++;
            ofs += ccnt + 1;
        }
    }
}

void CopyTranspose(double *src, double *tg, int r, int c)
{
	int ri, ci;
	for (ci = 0; ci < c; ci++) {
		
		double *iter = tg;
		for (ri = 0; ri < r; ri++) {

			*iter = *src;
			src ++;
			iter += c;
		}
		tg++;
	}
}

void mexFunction(int nlhs , mxArray *plhs[] , int nrhs , const mxArray *prhs[]){

	double *A, *Areduced, *tmp, p, tol;
	int m, n;

	if (nrhs < 2) tol = 1e-15;
	else tol = *mxGetPr(prhs[1]);
	
	A = mxGetPr(prhs[0]);
	m = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);

	plhs[0]	= mxCreateDoubleMatrix(m, n, mxREAL);
	Areduced = mxGetPr(plhs[0]);

	tmp = (double*)malloc(m*n*sizeof(double));

	CopyTranspose(A, tmp, m, n);
	GJ(tmp, m, n, tol);
	CopyTranspose(tmp, Areduced, n, m);

	free(tmp);
}
