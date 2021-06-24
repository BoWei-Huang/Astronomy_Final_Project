#include<iostream>
#include<fftw3.h>
#include<cmath>
#include<omp.h>
using namespace std;

#define REAL 0
#define IMAG 1

#define IsoBC  //if not defined, use periodicBC




//some global variables
int NThread=8;
const int size=32; // box size
const int padding=(2*size)*(2*size)*(2*size);
const int Rseed =1234;
double scale=1.0/(pow(static_cast<double>(size),3));
double paddingscale=1.0/(pow(static_cast<double>(2*size),3));
double *phians;  // analytic solution to center delta mass //
double phi[size][size][size];
double phians3D[size][size][size];
double bias; //tune the farest phi to 0//
double k1=M_PI/4.0;  // sine source k factor
double k2=M_PI/16.0; // cosine source k factor
void analytic(); // give reference solution for phi (1)delta source (2) sin,cos source  
// ------------------------------------------ //
#ifndef IsoBC
double *phi1D;    //   declare 3-d phi matrix to return //
double *rho; //   declare 3-d rho matrix //
fftw_complex *rhok; //    declare 3-D rhok of fftw_cpmplex object  //

void initforperiodic();
void deleteforperiodic();
void print(double [size][size][size]);
void FFT();
void IRFFT();
void D1_to_3D(double [size][size][size], double []);
#endif
// ---------------------------------------------  //
#ifdef IsoBC
double *isophi;
double *isorho;
double *R;
fftw_complex *isorhok;
fftw_complex *Rk; 

void initforisolated();
void deleteforisolated();
void isoFFT();
void isoIRFFT();
void iso_D1_to_3D(double [size][size][size], double[]); // just return size*size*size
void isoprint(double[size][size][size]); //print only size*size*size
#endif
//  ----------------------------------------------  //

// give reference solution to phi
void analytic()
{
    // 1. delta source //
/*    for(int x=0;x<size*size*size;x++)
    {  
        int k=x%(size);
        int j=(x/(size))%(size);
        int i=(x/(size))/(size);
        double dis=sqrt(static_cast<double>((k-size/2)*(k-size/2)+(j-size/2)*(j-size/2)+(i-size/2)*(i-size/2)));
        phians3D[i][j][k]=-100/dis-bias;
    }
    phians3D[size/2][size/2][size/2]=10000000000;
  */
    // 2. sine function source //
#pragma omp parallel
{
#pragma omp for
    for(int x=0;x<size*size*size;x++)
    {  
        int k=x%(size);
        int j=(x/(size))%(size);
        int i=(x/(size))/(size);
        phians3D[i][j][k]=5.0*(sin(k1*k)+sin(k1*j)+sin(k1*i))+10.0*(sin(k2*k)+sin(k2*j)+sin(k2*i));
        //cout<<phians3D[i][j][k]<<endl;
    }
}// end parallel

}

#ifndef IsoBC
// new matrix and give initial value //
void initforperiodic()
{
    phi1D=new double [size*size*size];
    
    phians= new double[size*size*size];
    
    rho=new double [size*size*size];
    
    //  -------- give initial value for rho(ex(1)) --------------//
    //rho[size/2+size*(size/2+size*(size/2))]=100;
    //bias=-100/sqrt(static_cast<double>(3*(size/2)*(size/2)));

    // ------------give initial value for rho ex(2) ------------ //
#pragma omp parallel
{
# pragma omp for
    for(int x=0;x<size*size*size;x++)
    {
        int k=x%(size);
        int j=(x/(size))%(size);
        int i=(x/(size))/(size);
        rho[x]=(1/(4*M_PI))*(-5.0*k1*k1*(sin(k1*k)+sin(k1*j)+sin(k1*i))-10.0*k2*k2*(sin(k2*k)+sin(k2*j)+sin(k2*i)));

    }
    
}//end parallel
    rhok=(fftw_complex*) fftw_malloc(size*size*(size/2+1) *sizeof(fftw_complex));

}


void deleteforperiodic()
{
    fftw_free(rhok);
    fftw_cleanup();

    //   delete pointer   //
    // 1. rho

    delete [] rho;

    // 2. phi
    delete [] phi1D;

    delete [] phians;
    
}
//   execute FFT //
void FFT() 
{
    fftw_plan plan= fftw_plan_dft_r2c_3d(size,size,size,rho,rhok,FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
#pragma omp parallel
{
#pragma omp for
    for(int x=0;x<size*size*(size/2+1);x++)
    {
                int zsize=size/2+1;
                int k=x%zsize;
                int j=(x/zsize)%size;
                int i=(x/zsize)/size;

                double kx = (i<=size/2) ? 2.0*M_PI*i/(size) : 2.0*M_PI*(i-size)/(size);
                double ky = (j<=size/2) ? 2.0*M_PI*j/(size) : 2.0*M_PI*(j-size)/(size);
                double kz = 2.0*M_PI*k/(size);
                               
                double scales=(-kx*kx-ky*ky-kz*kz-0.00000001);
                rhok[k+zsize*(j+size*i)][REAL]/=scales;
                rhok[k+zsize*(j+size*i)][IMAG]/=scales;
    }
}//end parallel 
    rhok[0][REAL]=0.0;
  
}

//inverse FFT with normalization//
void IRFFT()
{
    fftw_plan plan2=fftw_plan_dft_c2r_3d(size,size,size,rhok,phi1D,FFTW_ESTIMATE);
    fftw_execute(plan2);
    fftw_destroy_plan(plan2);
    //   normalize, show the result   //
    bias= phi1D[0]*scale*4*M_PI;
    //cout<<"bias"<<bias;
#pragma omp parallel
{
# pragma omp for
    for(int x=0;x<(size*size*size);x++)
    {
        phi1D[x]=(phi1D[x])*scale*4*M_PI;
        //cout<<phi[x]<<endl;
    }

}//end parallel
}
//  return phi to 3D //
void D1_to_3D(double mat[size][size][size], double mattt[])
{
#pragma omp parallel
{
# pragma omp for
    for(int x=0;x<size*size*size;x++)
    {
        int k=x%(size);
        int j=(x/(size))%(size);
        int i=(x/(size))/(size);

        mat[i][j][k]=mattt[x];
    }
}//end parallel
}
// show the result(3D-matrix) //
void print(double mat[size][size][size])
{
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            for(int k=0;k<size;k++)
            {
                //if(k>size/2-10 && k<size/2+10 && j>size/2-10 && j<size/2+10 && i>size/2-10 && i<size/2+10 )
                {
                    cout<<mat[i][j][k]/phians3D[i][j][k]<<" "; 
                    //cout<<mat[i][j][k]<<" ";
                }
                
            }
            cout<<endl;
            cout<<endl;
        }
        cout<<endl;
        cout<<endl;   
        cout<<endl;
    }
}
#endif
int main()
{
    omp_set_num_threads(NThread);
    double ctime1=omp_get_wtime();
#ifndef IsoBC
    analytic();
    initforperiodic();
    FFT();
    IRFFT();
    D1_to_3D(phi,phi1D);
    //print(phi);
    //print(phians3D);
    deleteforperiodic();

#endif
#ifdef IsoBC
    analytic();
    initforisolated();
    isoFFT();
    isoIRFFT();
    iso_D1_to_3D(phi,isophi);
    //isoprint(phi);
    //isoprint(phians3D);
    deleteforisolated();
#endif
    double ctime2=omp_get_wtime();
    cout<<"execution time"<<" "<<ctime2-ctime1;
    return 0;
}

#ifdef IsoBC
void initforisolated()
{
//    declare 3-d phi matrix to return      //
    isophi= new double[padding];
//               declare 3-d rho matrix          //
    isorho= new double[padding];
    // ------------- give initial value ex(1):delta mass -------------- //
    //isorho[size/2+size*(size/2+size*(size/2))]=100;

    // -------------- ex(2) sin,cos ------------ //
#pragma omp parallel
{
#pragma omp for
    for(int x=0;x<padding;x++)
    {
        int k=x%(2*size);
        int j=(x/(2*size))%(2*size);
        int i=(x/(2*size))/(2*size);
        if(k<size && j<size && i<size)
        {
            isorho[x]=(1.0/(4.0*M_PI))*(-5.0*k1*k1*(sin(k1*k)+sin(k1*j)+sin(k1*i))-10.0*k2*k2*(sin(k2*k)+sin(k2*j)+sin(k2*i)));
        }
        else
        {
            isorho[x]=0.0;
        }
    }
}
//          declare R-matrix(distance matrix)  //
    R= new double [padding];
# pragma omp parallel
{
# pragma omp for
    for(int x=0;x<padding;x++)
    {
        int k=x%(2*size);
        int j=(x/(2*size))%(2*size);
        int i=(x/(2*size))/(2*size);
        int ri = (i<size) ? i: 2*size-i;
        int rj = (j<size) ? j : 2*size-j;
        int rk=  (k<size) ? k : 2*size-k;
              
        R[x]=(-1/sqrt(pow(static_cast<double>(rk),2)+pow(static_cast<double>(rj),2)+pow(static_cast<double>(ri),2)));
        
           
    }
}//end parallel
    R[0]=0.0;

//    initial kspace r-matrix and rhokmatrix  //
    isorhok=(fftw_complex*) fftw_malloc((2*size)*(2*size)*(size+1) *sizeof(fftw_complex));
    Rk=(fftw_complex*) fftw_malloc((2*size)*(2*size)*(size+1) *sizeof(fftw_complex));

}

void deleteforisolated()
{
    fftw_free(isorhok);
    fftw_free(Rk);
    fftw_cleanup();
//      delete pointer       //
    // 1. rho

    delete [] isorho;

    // 2. phi
    delete [] isophi;
    // 3. R
    delete [] R;

    

}

void isoFFT()
{
    fftw_plan plan= fftw_plan_dft_r2c_3d(2*size,2*size,2*size,isorho,isorhok,FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    fftw_plan plan2= fftw_plan_dft_r2c_3d(2*size,2*size,2*size,R,Rk,FFTW_ESTIMATE);
    fftw_execute(plan2);
    fftw_destroy_plan(plan2);
    
#pragma omp parallel
{
#pragma omp for
    //  **** FFT(R)*FFT(rho)**** //
    for(int x=0 ; x<(2*size)*(2*size)*(size+1);x++ )
    {
        double Re=isorhok[x][REAL];
        double Im=isorhok[x][IMAG];
        isorhok[x][REAL]=(Re*Rk[x][REAL]-Im*Rk[x][IMAG]);
        isorhok[x][IMAG]=(Re*Rk[x][IMAG]+Im*Rk[x][REAL]);
        //cout<<"yo"<<isorhok[x][REAL]<<"　"<<isorhok[x][IMAG]<<"i"<<endl;
    }  
}//end parallel
    isorhok[0][REAL]=0.0; //assure DC term=0(no DC-bias) 

}

void isoIRFFT()
{
    fftw_plan plan3 = fftw_plan_dft_c2r_3d(2*size,2*size,2*size,isorhok,isophi,FFTW_ESTIMATE);
    fftw_execute(plan3);
    fftw_destroy_plan(plan3);

    //   normalize    //
# pragma omp parallel
{
# pragma omp for
    for(int x=0;x<(2*size*2*size*2*size);x++)
    {
        isophi[x]=(isophi[x])*paddingscale;
    }

}
}

// ----------- return 3D phi with only size*size*size  ------- //
void iso_D1_to_3D(double mat[size][size][size], double mattt[])
{
# pragma omp parallel
{
# pragma omp for
    for(int x=0;x<padding;x++)
    {
        int k=x%(2*size);
        int j=(x/(2*size))%(2*size);
        int i=(x/(2*size))/(2*size);
        if(i<size && j<size && k<size)
        {
            mat[i][j][k]=mattt[x];
        }
    }
}//end parallel
}

void isoprint(double mat[size][size][size])
{
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            for(int k=0;k<size;k++)
            {
                //if(k>size/2-5 && k<size/2+5 && j>size/2-5 && j<size/2+5 && i>size/2-5 && i<size/2+5 )
                {
                    cout<<mat[i][j][k]<<" "; 
                }               
            }
            cout<<endl;
            cout<<endl;
        }
        cout<<endl;
        cout<<endl;   
        cout<<endl;
    }

}
#endif
 




















