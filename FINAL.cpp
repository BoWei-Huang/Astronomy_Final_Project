#include<iostream>
#include<fftw3.h>
#include<cmath>
#include<omp.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include<fstream>
#include<string>
using namespace std;

#define REAL 0
#define IMAG 1

#define IsoBC  //if not defined, use periodicBC

#define KD //if not defined, use DKD to update position


//FFT some global variables
int NThread= 4;
const int N=128; // box N
const int padding=(2*N)*(2*N)*(2*N);
const int Rseed =1234;
float scale=1.0/(pow(static_cast<float>(N),3));
float paddingscale=1.0/(pow(static_cast<float>(2*N),3));
float *phians;  // analytic solution to center delta mass //
float phi[N][N][N];
float phians3D[N][N][N];
float bias; //tune the farest phi to 0//
float k1=M_PI/4.0;  // sine source k factor
float k2=M_PI/16.0; // cosine s for (float t=0; t<T; t=t+dt)
    
void analytic(); // give reference solution for phi (1)delta source (2) sin,cos source  
// ------------------------------------------ //
#ifndef IsoBC
float *phi1D;    //   declare 3-d phi matrix to return //
float *rho; //   declare 3-d rho matrix //
fftwf_complex *rhok; //    declare 3-D rhok of fftwf_cpmplex object  //

void initforperiodic();
void initperiodicvalue();
void deleteforperiodic();
void print(float [N][N][N]);
void FFT();
void IRFFT();
void D1_to_3D(float [N][N][N], float []);
#endif
// ---------------------------------------------  //
#ifdef IsoBC
float *isophi;
float *isorho;
float *R;
fftwf_complex *isorhok;
fftwf_complex *Rk; 

void initforisolated();
void initialisorhovalue();
void deleteforisolated();
void isoFFT();
void isoIRFFT();
void iso_D1_to_3D(float [N][N][N], float[]); // just return N*N*N
void isoprint(float[N][N][N]); //print only N*N*N
#endif
//  ----------------------------------------------  //

// ------- define particle mesh global ------------//
//define the const
int method = 2; //(1:NGP method ; 2:CIC method ; 3:TSC method)
float G = 1.0 ;
const int n = 2000; // number of paritcle
//const int N = 4; // number of grid
const int N_total=N+2;  //include boundary(buffer) 
float L = N ;// N of box
float dx=L/N;

float m[n]; //mass of particle
float pos[n][3];//position of particle
float vel[n][3];//velcoity of particle
float x[N_total]={0.0};
float y[N_total]={0.0};
float z[N_total]={0.0};
float total = 0.0;//total mass of particle
float total_array = 0.0;//total mass of grid
float grid_mass[N][N][N] = {0.0}; //Define empty grid mass
float num_mass[N_total][N_total][N_total];
int indexx[n][3][2];//Define relevent index set of CIC scheme
int indexy[n][3][3];//Define relecent index set of TSC scheme
//float phi[N][N][N]={};//Define Potential
float grid_force_x[N][N][N]={};//Define Grid Force of Three Dimension
float grid_force_y[N][N][N]={};
float grid_force_z[N][N][N]={};
float particle_force[n][3]={};//force on every particle
float num_force_x[N_total][N_total][N_total]={};//Define Grid Force with Buffer
float num_force_y[N_total][N_total][N_total]={};
float num_force_z[N_total][N_total][N_total]={};



// -------------- KDK, DKD method global variables ---------------------- //
float F[n][3];         //Force on particles
int tooshort[n][n];  //Recording whether any two particles are too close
float dt = 0.01;       //renew every dt seconds
float T = 0.5;        //total time(unit: second)
float t=0.0 ;        //current time
int edge[n][3]; //判斷該粒子是否碰到邊界，是則為1，否則為0
float acc[n][3]; 
float momentum[3]; //at t=t total momentum
// ----------------------------------------------------------------- //

// --------------------------------  FFT Part ------------------------------ //
// give reference solution to phi
void analytic()
{
    // 1. delta source //
/*    for(int x=0;x<N*N*N;x++)
    {  
        int k=x%(N);
        int j=(x/(N))%(N);
        int i=(x/(N))/(N);
        float dis=sqrt(static_cast<float>((k-N/2)*(k-N/2)+(j-N/2)*(j-N/2)+(i-N/2)*(i-N/2)));
        phians3D[i][j][k]=-100/dis-bias;
    }
    phians3D[N/2][N/2][N/2]=10000000000;
  */
    // 2. sine function source //
#pragma omp parallel
{
#pragma omp for
    for(int x=0;x<N*N*N;x++)
    {  
        int k=x%(N);
        int j=(x/(N))%(N);
        int i=(x/(N))/(N);
        phians3D[i][j][k]=5.0*(sin(k1*k)+sin(k1*j)+sin(k1*i))+10.0*(sin(k2*k)+sin(k2*j)+sin(k2*i));
        //cout<<phians3D[i][j][k]<<endl;
    }
}// end parallel

}

#ifndef IsoBC
// new matrix and give initial value //
void initforperiodic()
{
    phi1D=new float [N*N*N];
    
    phians= new float[N*N*N];
    
    rho=new float [N*N*N];
    
    rhok=(fftwf_complex*) fftwf_malloc(N*N*(N/2+1) *sizeof(fftwf_complex));

    
}

void initperiodicvalue() // give rho(3-D) value to rho1D
{
//  -------- give initial value for rho(ex(1)) --------------//
    //rho[N/2+N*(N/2+N*(N/2))]=100;
    //bias=-100/sqrt(static_cast<float>(3*(N/2)*(N/2)));

    // ------------give initial value for rho ex(2) ------------ //
#pragma omp parallel
{
# pragma omp for
    for(int x=0;x<N*N*N;x++)
    {
        int k=x%(N);
        int j=(x/(N))%(N);
        int i=(x/(N))/(N);
        /* give analytic rho
        rho[x]=(1/(4*M_PI))*(-5.0*k1*k1*(sin(k1*k)+sin(k1*j)+sin(k1*i))-10.0*k2*k2*(sin(k2*k)+sin(k2*j)+sin(k2*i)));
        */
        rho[x]=grid_mass[i][j][k];
    }
    
}//end parallel
}
void deleteforperiodic()
{
    fftwf_free(rhok);
    fftwf_cleanup();

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
    fftwf_plan_with_nthreads(NThread);
    fftwf_plan plan= fftwf_plan_dft_r2c_3d(N,N,N,rho,rhok,FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
#pragma omp parallel
{
#pragma omp for
    for(int x=0;x<N*N*(N/2+1);x++)
    {
                int zN=N/2+1;
                int k=x%zN;
                int j=(x/zN)%N;
                int i=(x/zN)/N;

                float kx = (i<=N/2) ? 2.0*M_PI*i/(N) : 2.0*M_PI*(i-N)/(N);
                float ky = (j<=N/2) ? 2.0*M_PI*j/(N) : 2.0*M_PI*(j-N)/(N);
                float kz = (k<=N/2) ? 2.0*M_PI*k/(N) : 2.0*M_PI*(k-N)/(N);
                               
                float scales=(-kx*kx-ky*ky-kz*kz-0.00000001);
                rhok[k+zN*(j+N*i)][REAL]/=scales;
                rhok[k+zN*(j+N*i)][IMAG]/=scales;
    }
}//end parallel 
    rhok[0][REAL]=0.0;
  
}

//inverse FFT with normalization//
void IRFFT()
{
    fftwf_plan_with_nthreads(NThread);
    fftwf_plan plan2=fftwf_plan_dft_c2r_3d(N,N,N,rhok,phi1D,FFTW_ESTIMATE);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
    fftwf_cleanup_threads();
    //   normalize, show the result   //
    bias= phi1D[0]*scale*4*M_PI;
    //cout<<"bias"<<bias;
#pragma omp parallel
{
# pragma omp for
    for(int x=0;x<(N*N*N);x++)
    {
        phi1D[x]=(phi1D[x])*scale*4.0*M_PI;
        //cout<<phi[x]<<endl;
    }

}//end parallel
}
//  return phi to 3D //
void D1_to_3D(float mat[N][N][N], float mattt[])
{
#pragma omp parallel
{
# pragma omp for
    for(int x=0;x<N*N*N;x++)
    {
        int k=x%(N);
        int j=(x/(N))%(N);
        int i=(x/(N))/(N);

        mat[i][j][k]=mattt[x];
    }
}//end parallel
}
// show the result(3D-matrix) //
void print(float mat[N][N][N])
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            for(int k=0;k<N;k++)
            {
                //if(k>N/2-10 && k<N/2+10 && j>N/2-10 && j<N/2+10 && i>N/2-10 && i<N/2+10 )
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


#ifdef IsoBC

void initforisolated()  // init isophi, isorho, isorhok matrix,  R, Rk(with FFT)
{
//    declare 3-d phi matrix to return      //
    isophi= new float[padding];
//               declare 3-d rho matrix          //
    isorho= new float[padding];
   
//          declare R-matrix(distance matrix)  //
    R= new float [padding];
# pragma omp parallel
{
# pragma omp for
    for(int x=0;x<padding;x++)
    {
        int k=x%(2*N);
        int j=(x/(2*N))%(2*N);
        int i=(x/(2*N))/(2*N);
        int ri = (i<N) ? i: 2*N-i;
        int rj = (j<N) ? j : 2*N-j;
        int rk=  (k<N) ? k : 2*N-k;
              
        R[x]=(-1/sqrt(pow(static_cast<float>(rk),2)+pow(static_cast<float>(rj),2)+pow(static_cast<float>(ri),2)));
        
           
    }
}//end parallel
    R[0]=0.0;

//    initial kspace r-matrix and rhokmatrix  //
    isorhok=(fftwf_complex*) fftwf_malloc((2*N)*(2*N)*(N+1) *sizeof(fftwf_complex));
    Rk=(fftwf_complex*) fftwf_malloc((2*N)*(2*N)*(N+1) *sizeof(fftwf_complex));
    // ------------------- FFT to R ------------------------ //
    fftwf_plan plan2= fftwf_plan_dft_r2c_3d(2*N,2*N,2*N,R,Rk,FFTW_ESTIMATE);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
}
void initialisorhovalue()  // give rho value(3-D) to isorho(1-D)
{
 // ------------- give initial value ex(1):delta mass -------------- //
    //isorho[N/2+N*(N/2+N*(N/2))]=100;

    // -------------- ex(2) sin,cos ------------ //
#pragma omp parallel
{
#pragma omp for
    for(int x=0;x<padding;x++)
    {
        int k=x%(2*N);
        int j=(x/(2*N))%(2*N);
        int i=(x/(2*N))/(2*N);
        
        if(k<N && j<N && i<N)
        {
            //isorho[x]=(1.0/(4.0*M_PI))*(-5.0*k1*k1*(sin(k1*k)+sin(k1*j)+sin(k1*i))-10.0*k2*k2*(sin(k2*k)+sin(k2*j)+sin(k2*i)));   analytic sin's rho
            isorho[x]=grid_mass[i][j][k];
        }
        else
        {
            isorho[x]=0.0;
        }
    }
}
}

void deleteforisolated()
{
    fftwf_free(isorhok);
    fftwf_free(Rk);
    fftwf_cleanup();
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
    fftwf_plan_with_nthreads(NThread);
    fftwf_plan plan= fftwf_plan_dft_r2c_3d(2*N,2*N,2*N,isorho,isorhok,FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
/*
    fftwf_plan plan2= fftwf_plan_dft_r2c_3d(2*N,2*N,2*N,R,Rk,FFTW_ESTIMATE);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
*/ 
#pragma omp parallel
{
#pragma omp for
    //  **** FFT(R)*FFT(rho)**** //
    for(int x=0 ; x<(2*N)*(2*N)*(N+1);x++ )
    {
        float Re=isorhok[x][REAL];
        float Im=isorhok[x][IMAG];
        isorhok[x][REAL]=(Re*Rk[x][REAL]-Im*Rk[x][IMAG]);
        isorhok[x][IMAG]=(Re*Rk[x][IMAG]+Im*Rk[x][REAL]);
        //cout<<"yo"<<isorhok[x][REAL]<<"???"<<isorhok[x][IMAG]<<"i"<<endl;
    }  
}//end parallel
    isorhok[0][REAL]=0.0; //assure DC term=0(no DC-bias) 

}

void isoIRFFT()
{
    fftwf_plan_with_nthreads(NThread);
    fftwf_plan plan3 = fftwf_plan_dft_c2r_3d(2*N,2*N,2*N,isorhok,isophi,FFTW_ESTIMATE);
    fftwf_execute(plan3);
    fftwf_destroy_plan(plan3);
    fftwf_cleanup_threads();

    //   normalize    //
# pragma omp parallel
{
# pragma omp for
    for(int x=0;x<(2*N*2*N*2*N);x++)
    {
        isophi[x]=(isophi[x])*paddingscale;
    }

}
}

// ----------- return 3D phi with only N*N*N  ------- //
void iso_D1_to_3D(float mat[N][N][N], float mattt[])
{
# pragma omp parallel
{
# pragma omp for
    for(int x=0;x<padding;x++)
    {
        int k=x%(2*N);
        int j=(x/(2*N))%(2*N);
        int i=(x/(2*N))/(2*N);
        if(i<N && j<N && k<N)
        {
            mat[i][j][k]=mattt[x];
        }
    }
}//end parallel
}

void isoprint(float mat[N][N][N])
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            for(int k=0;k<N;k++)
            {
                //if(k>N/2-5 && k<N/2+5 && j>N/2-5 && j<N/2+5 && i>N/2-5 && i<N/2+5 )
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

//Define Weighting Function (CIC and TSC) 
float W(float x_p,float x_g ){
    float d=fabs(x_p-x_g);
    float w=0;
    if (method==2){
    	if (d <= dx){
        	w=1-d/dx;
    	} else{
        	w=0;
    	}
    }
    if (method==3) {       
        if (d <= 0.5*dx){
            w=3.0/4.0-pow((d/dx),2);
        } else if ( d <= 1.5*dx){
            w=0.5*pow(3.0/2.0-d/dx,2);
        } else{
            w=0;
        }
    } 
//    printf("%.8f \n",d);
    return w;
}

//Define Periodic function
int re(int i){
    int p=i;
    if (i==0){
        p=N_total-2;
    }if (i==N_total-1){
        p=1;
    }
    return p;
}


void inti()
{
//Define Mass
    for (int i=0;i<n;i++){
        m[i]=1.0;
    }

//Define position
    float r[n];
    float theta[n];
    float psi[n];
    for (int i=0;i<n;i++){
        r[i]=0.5*L * rand() / (RAND_MAX + 1.0) ;
        theta[i]=M_PI*rand() / (RAND_MAX + 1.0);
        psi[i]= 2*M_PI*rand() / (RAND_MAX + 1.0);
        //pos[i][j]=L * rand() / (RAND_MAX + 1.0) + -L/2;  //give random position in box
        pos[i][0]=r[i]*sin(theta[i])*cos(psi[i]);
        pos[i][1]=r[i]*sin(theta[i])*sin(psi[i]);
        pos[i][2]=r[i]*cos(theta[i]);

    }

//Define velocity
    for (int i=0;i<n;i++){
        for (int j=0;j<3;j++){
            //vel[i][j]=(L * rand() / (RAND_MAX + 1.0) + -L/2)/10;
            vel[i][j]=0.0;
        }
		}

//Define Coordinate 
    for (int i=1 ; i<N_total-1 ;i++) 
        {
            x[i] = 0.5*(-L-dx)+i*dx; 
            y[i] = 0.5*(-L-dx)+i*dx;
            z[i] = 0.5*(-L-dx)+i*dx;
        }

//Set Boundary Coordintate
    x[0]= -L/2-dx/2;
    y[0]= -L/2-dx/2;
    z[0]= -L/2-dx/2;
    x[N_total-1]=L/2+dx/2;
    y[N_total-1]=L/2+dx/2;
    z[N_total-1]=L/2+dx/2;	
} 

void total_mass_particle()//Calcaulate the total mass of particle
{
for (int i = 0; i<n; i++){
	total = total+ m[i];
	}
	//printf ("the total mass of particle : %.2f \n " , total);
	//printf ("\n");
}

void total_mass_grid()//Calcaulate the total mass of grid
{
# pragma omp parallel for
for (int i=0 ; i<N;i++){
	for (int j=0 ; j<N;j++){
		for (int k=0 ; k<N;k++){
			total_array = total_array + grid_mass[i][j][k];
			}
		}
	}
  // printf ("the total mass of particle in grid : %.2f \n " , total_array);
  // printf ("\n");	
}

void error_mass()//Calcaulate the mass error
{
float error = total - total_array ;
//	printf ("the error of mass : %.2f \n " ,error );
//	printf ("\n");	
}

void NGP_par_mesh()//NGP for particle mesh
{   

	//printf("method %d : NGP\n",method);

//detect the particle pos
//	# pragma omp parallel for
	for (int i=0; i<n; i++){
    
    //int x_pos =floor((pos[i][0]+L/2-dx/2)/dx);
    //int y_pos =floor((pos[i][1]+L/2-dx/2)/dx);
    //int z_pos =floor((pos[i][2]+L/2-dx/2)/dx);
    int x_pos =floor((pos[i][0]+L/2)/dx);
    int y_pos =floor((pos[i][1]+L/2)/dx);
    int z_pos =floor((pos[i][2]+L/2)/dx);

//	printf ("Grid of the particle %d, x= %d, y= %d, z= %d \n",i, x_pos, y_pos, z_pos);
	
//assign the mass to grid
	grid_mass[x_pos][y_pos][z_pos] = grid_mass[x_pos][y_pos][z_pos]+m[i];

	}
printf("\n");
//show the position 
/*
# pragma omp parallel for
for (int i=0 ; i<n;i++){
	printf( "position of Particle %d\n", i ); 
     for (int j=0 ;j<3;j++){
    printf("  %.8f  ",pos[i][j]);
    }
    printf("\n");
	}
printf("\n");
*/

//show the grid of mass
/*
printf( "grid mass (NGP) for N=%d\n", N );
for (int i=0 ; i<N;i++){
    printf( "X = %d\n", i );
    for (int j=0 ;j<N ;j++){
		for (int k=0 ;k<N ;k++){
		 printf("  %.2f  ",grid_mass[i][j][k]);
			}
		printf("\n");	
		}	
		printf("\n");
   }
*/
}

void CIC_par_mesh()//CIC for particle mesh
{
//printf("method %d : CIC\n",method);

//Mass Distribution of nth particle
//3D Cloud in Cell
#pragma omp parallel
{
	# pragma omp for 
    for (int i=0;i<n;i++)
        for (int j=0; j<3; j++)//find the indexx of relevent eight grid in CIC
        {
            indexx[i][j][0]=floor((pos[i][j]+L/2-dx/2)/dx);
            indexx[i][j][0]=(indexx[i][j][0]+1+N_total)%N_total;
            indexx[i][j][1]=(indexx[i][j][0]+1)%N_total;
        } 
    }//end parallelization
//Mass Assignment
    for (int i=0;i<n;i++){
        for (int j=0; j<2; j++){
            for (int k=0; k<2; k++){
                for (int p=0; p<2; p++){
                    num_mass[indexx[i][0][j]][indexx[i][1][k]][indexx[i][2][p]]=num_mass[indexx[i][0][j]][indexx[i][1][k]][indexx[i][2][p]]+m[i]*W(pos[i][0],x[indexx[i][0][j]])*W(pos[i][1],y[indexx[i][1][k]])*W(pos[i][2],z[indexx[i][2][p]]);
                }
            }
        }
    }
//Periodic Boundary Implement
    for (int i=0; i<N_total ; i++){
        for (int j=0; j<N_total ; j++){
            for (int k=0; k<N_total ; k++){
                if (i==0 || j==0 || k==0|| i==N_total-1||j==N_total-1||k==N_total-1){
                    num_mass[re(i)][re(j)][re(k)]=num_mass[re(i)][re(j)][re(k)]+num_mass[i][j][k];
                    num_mass[i][j][k]=0;
                }
            }
        }        
    }

//output assignment result :grid_mass
#pragma omp parallel
{
	# pragma omp for collapse(3)
    for (int i=1;i<N_total-1;i++){
        for (int j=1;j<N_total-1;j++){
            for (int k=1;k<N_total-1;k++){
                grid_mass[i-1][j-1][k-1]=num_mass[i][j][k];
            }
        }
    }
}
//printf("\n");

/*
//show the position 
for (int i=0 ; i<n;i++){
	printf( "position of Particle %d\n", i ); 
     for (int j=0 ;j<3;j++){
    printf("  %.8f  ",pos[i][j]);
    }
    printf("\n");
	}
printf("\n");

//show the grid of mass
printf( "grid mass (CIC) for N=%d\n", N );
for (int i=0 ; i<N;i++){
    printf( "X = %d\n", i );
    for (int j=0 ;j<N ;j++){
		for (int k=0 ;k<N ;k++){
		 printf("  %.2f  ",grid_mass[i][j][k]);
			}
		printf("\n");	
		}	
		printf("\n");
   }
*/
}
void TSC_par_mesh()//TSC for particle mesh
{
//3D Triangular-Shaped-Cloud
//    printf("method %d : NGP\n",method);
#pragma omp parallel
{
    #pragma omp for
    for (int i=0;i<n;i++){
//find the index of relevent eight grid in TSC
        for (int j=0; j<3; j++){
            indexy[i][j][1]=lround((pos[i][j]+L/2+0.5*dx)/dx);
            if (indexy[i][j][1]==N+1) {indexy[i][j][1]= N; }
            indexy[i][j][0]=(indexy[i][j][1]-1+N_total)%N_total;
            indexy[i][j][2]=(indexy[i][j][1]+1+N_total)%N_total;
        }
    }
}//end parallelization
//Mass Assignment
    for (int i=0;i<n;i++){
        for (int j=0; j<3; j++){
            for (int k=0; k<3; k++){
                for (int p=0; p<3; p++){
                    num_mass[indexy[i][0][j]][indexy[i][1][k]][indexy[i][2][p]]=num_mass[indexy[i][0][j]][indexy[i][1][k]][indexy[i][2][p]]+m[i]*W(pos[i][0],x[indexy[i][0][j]])*W(pos[i][1],y[indexy[i][1][k]])*W(pos[i][2],z[indexy[i][2][p]]);

                }
            }
        }
    }


//Periodic Boundary Implement
    for (int i=0; i<N_total ; i++){
        for (int j=0; j<N_total ; j++){
            for (int k=0; k<N_total ; k++){
                if (i==0 || j==0 || k==0|| i==N_total-1||j==N_total-1||k==N_total-1){
                    num_mass[re(i)][re(j)][re(k)]=num_mass[re(i)][re(j)][re(k)]+num_mass[i][j][k];
                    num_mass[i][j][k]=0;
                }
            }
        }        
    }
#pragma omp parallel
{
//output assignment result :grid_mass
    #pragma omp for collapse(3)
    for (int i=1;i<N_total-1;i++){
        for (int j=1;j<N_total-1;j++){
            for (int k=1;k<N_total-1;k++){
                grid_mass[i-1][j-1][k-1]=num_mass[i][j][k];
            }
        }
    }
}
}


void test_potential()//test potential 
{
	for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                phi[i][j][k]=-G/sqrt(x[i]*x[i]+y[j]*y[j]+z[k]*z[k]+1);
            }
        }
    }
}

void potential_to_force()//change potential to force
{
//x-direction
#pragma omp parallel
{
	# pragma omp for
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                if (i==0) {
                    grid_force_x[i][j][k]=-grid_mass[i][j][k]*(phi[i+1][j][k]-phi[N-1][j][k])/(2*dx);
                } else if (i==(N-1)) {
                    grid_force_x[i][j][k]=-grid_mass[i][j][k]*(phi[0][j][k]-phi[i-1][j][k])/(2*dx);
                } else {
                    grid_force_x[i][j][k]=-grid_mass[i][j][k]*(phi[i+1][j][k]-phi[i-1][j][k])/(2*dx);
                }
                if (j==0) {
                    grid_force_y[i][j][k]=-grid_mass[i][j][k]*(phi[i][j+1][k]-phi[i][N-1][k])/(2*dx);
                } else if (j==(N-1)) {
                    grid_force_y[i][j][k]=-grid_mass[i][j][k]*(phi[i][0][k]-phi[i][j-1][k])/(2*dx);
                } else {
                    grid_force_y[i][j][k]=-grid_mass[i][j][k]*(phi[i][j+1][k]-phi[i][j-1][k])/(2*dx);
                }
                if (k==0) {
                    grid_force_z[i][j][k]=-grid_mass[i][j][k]*(phi[i][j][k+1]-phi[i][j][N-1])/(2*dx);
                } else if (k==(N-1)) {
                    grid_force_z[i][j][k]=-grid_mass[i][j][k]*(phi[i][j][0]-phi[i][j][k-1])/(2*dx);
                } else {
                    grid_force_z[i][j][k]=-grid_mass[i][j][k]*(phi[i][j][k+1]-phi[i][j][k-1])/(2*dx);
                }
            }
        }
    }

}//end parallel
}

void NGP_force()//NGP for return force
{
	for (int i=0;i<n;i++){
		//int x_pos =floor((pos[i][0]+L/2-dx/2)/dx);
        //int y_pos =floor((pos[i][1]+L/2-dx/2)/dx);
        //int z_pos =floor((pos[i][2]+L/2-dx/2)/dx);
		
        int x_pos =floor((pos[i][0]+L/2)/dx);
        int y_pos =floor((pos[i][1]+L/2)/dx);
        int z_pos =floor((pos[i][2]+L/2)/dx);

        particle_force[i][0] = grid_force_x[x_pos][y_pos][z_pos];
		particle_force[i][1] = grid_force_y[x_pos][y_pos][z_pos];
		particle_force[i][2] = grid_force_z[x_pos][y_pos][z_pos];

		//printf("the force one the particle %d\n Fx = %.2f Fy = %.2f Fz = %.2f \n",i ,particle_force[i][0],particle_force[i][1],particle_force[i][2]);
	}
}

void CIC_force()//CIC for return force
{
#pragma omp parallel
{
//Copy the internal part
    #pragma omp for collapse(3)
    for (int i=1; i<1+N; i++){
        for (int j=1; j<1+N; j++){
            for (int k=1; k<1+N; k++){
                num_force_x[i][j][k]=grid_force_x[i-1][j-1][k-1];
                num_force_y[i][j][k]=grid_force_y[i-1][j-1][k-1];
                num_force_z[i][j][k]=grid_force_z[i-1][j-1][k-1];
            }
        }
    }

//assign boundary value
	# pragma omp for collapse(3)
    for (int i=0; i<N_total ; i++){
        for (int j=0; j<N_total ; j++){
            for (int k=0; k<N_total ; k++){
                if (i==0 || j==0 || k==0|| i==N_total-1||j==N_total-1||k==N_total-1){
                    num_force_x[i][j][k]=num_force_x[re(i)][re(j)][re(k)];
                    num_force_y[i][j][k]=num_force_y[re(i)][re(j)][re(k)];
                    num_force_z[i][j][k]=num_force_z[re(i)][re(j)][re(k)];
                }
            }
        }        
    }   
}
//interpolation by inverse CIC
//F_x        
		
    for (int i=0; i<n; i++){
        for (int j=0; j<2; j++){
            for (int k=0; k<2; k++){
                for (int p=0; p<2; p++){
                    particle_force[i][0]=particle_force[i][0]+num_force_x[indexx[i][0][j]][indexx[i][1][k]][indexx[i][2][p]]*(W(pos[i][0],x[indexx[i][0][j]])*W(pos[i][1],y[indexx[i][1][k]])*W(pos[i][2],z[indexx[i][2][p]]));
                    particle_force[i][1]=particle_force[i][1]+num_force_y[indexx[i][0][j]][indexx[i][1][k]][indexx[i][2][p]]*(W(pos[i][0],x[indexx[i][0][j]])*W(pos[i][1],y[indexx[i][1][k]])*W(pos[i][2],z[indexx[i][2][p]]));
                    particle_force[i][2]=particle_force[i][2]+num_force_z[indexx[i][0][j]][indexx[i][1][k]][indexx[i][2][p]]*(W(pos[i][0],x[indexx[i][0][j]])*W(pos[i][1],y[indexx[i][1][k]])*W(pos[i][2],z[indexx[i][2][p]]));    
                
				}
            }
        }
       // printf("the force one the particle %d\n Fx = %.2f Fy = %.2f Fz = %.2f \n",i ,particle_force[i][0],particle_force[i][1],particle_force[i][2]);
    }
}

void reset()
{
    for (int i=0;i<n;i++){
        for (int j=0;j<3;j++){
           particle_force[i][0]=0;
           particle_force[i][1]=0;
           particle_force[i][2]=0; 
        }
    }
    for (int i=0;i<n;i++){
            for (int j=0; j<3; j++){
                for (int k=0; k<3; k++){
                    for (int p=0; p<3; p++){
                        num_mass[indexx[i][0][j]][indexx[i][1][k]][indexx[i][2][p]]=0; 
                    }   
                }
            }
        }
    }
void TSC_force()//TSC for return force
{
# pragma omp parallel
{
//Copy the internal part
    #pragma omp for collapse(3)
    for (int i=1; i<1+N; i++){
        for (int j=1; j<1+N; j++){
            for (int k=1; k<1+N; k++){
                num_force_x[i][j][k]=grid_force_x[i-1][j-1][k-1];
                num_force_y[i][j][k]=grid_force_y[i-1][j-1][k-1];
                num_force_z[i][j][k]=grid_force_z[i-1][j-1][k-1];
            }
        }
    }
//assign boundary value
    # pragma omp for collapse(3)
    for (int i=0; i<N_total ; i++){
        for (int j=0; j<N_total ; j++){
            for (int k=0; k<N_total ; k++){
                if (i==0 || j==0 || k==0|| i==N_total-1||j==N_total-1||k==N_total-1){
                    num_force_x[i][j][k]=num_force_x[re(i)][re(j)][re(k)];
                    num_force_y[i][j][k]=num_force_y[re(i)][re(j)][re(k)];
                    num_force_z[i][j][k]=num_force_z[re(i)][re(j)][re(k)];
                }
            }
        }        
    } 
}//end parallelization

//interpolation by inverse TSC        
    for (int i=0; i<n; i++){
        for (int j=0; j<3; j++){
            for (int k=0; k<3; k++){
                for (int p=0; p<3; p++){
                    particle_force[i][0]=particle_force[i][0]+num_force_x[indexy[i][0][j]][indexy[i][1][k]][indexy[i][2][p]]*(W(pos[i][0],x[indexy[i][0][j]])*W(pos[i][1],y[indexy[i][1][k]])*W(pos[i][2],z[indexy[i][2][p]]));
                    particle_force[i][1]=particle_force[i][1]+num_force_y[indexy[i][0][j]][indexy[i][1][k]][indexy[i][2][p]]*(W(pos[i][0],x[indexy[i][0][j]])*W(pos[i][1],y[indexy[i][1][k]])*W(pos[i][2],z[indexy[i][2][p]]));
                    particle_force[i][2]=particle_force[i][2]+num_force_z[indexy[i][0][j]][indexy[i][1][k]][indexy[i][2][p]]*(W(pos[i][0],x[indexy[i][0][j]])*W(pos[i][1],y[indexy[i][1][k]])*W(pos[i][2],z[indexy[i][2][p]]));    
                }
            }
        }
    }

}




/*
void init() //初始條件，包含位置、受力、質量，以及令初速度為零，合併時不用放
{
pos[0][0]=0.2;
pos[0][1]=0;
pos[0][2]=0;

pos[1][0]=-0.3;
pos[1][1]=0;
pos[1][2]=0;

F[0][0]=-0.2;
F[0][1]=0;
F[0][2]=0;

F[1][0]=0.2;
F[1][1]=0;
F[1][2]=0;

m[0] = 1;
m[1] = 1;


    for (int p=0; p<n; p++)
    {
        for (int d=0; d<3; d++)
        {
            vel[p][d] = 0;
        }
    }
}
*/
void boundery()//periodical 邊界
{
    for (int p=0; p<n; p++)
            {
                for (int d=0; d<3; d++)
                {
                    if(pos[p][d]>=0.5*L)//正邊界
                    {
                        edge[p][d] = 1;
                    }
                    else if(pos[p][d]<=-0.5*L)//負邊界
                    {
                        edge[p][d] = 2;
                    }
                    else
                    {
                        edge[p][d] = 0;
                    }
                }


            }
}

void periodic()
{
    for (int p=0; p<n; p++)
    {
        for (int d=0; d<3; d++)
        {
            if(edge[p][d] == 1)
            {
                pos[p][d]=pos[p][d]-L;
            }
            else if(edge[p][d] == 2)
            {
                pos[p][d]=pos[p][d]+L;
            }
        }
    }

}
void distance()//計算第p個質點和第p+i個質點的距離平方是否小於小於一定值(dd)，陣列tooshort中，1代表距離過近，0代表距離夠遠
{
    #pragma omp parallel
    {
        #pragma omp for 
        for (int p=0; p<n; p++)
        {
            for (int i=1; p+i<=n; i++)
            {
                float dd = abs(pos[p][0]-pos[p+i][0])*abs(pos[p][0]-pos[p+i][0])+
                            abs(pos[p][1]-pos[p+i][1])*abs(pos[p][1]-pos[p+i][1])+
                            abs(pos[p][2]-pos[p+i][2])*abs(pos[p][2]-pos[p+i][2]);
                if (dd<(0.01*L)*(0.01*L))
                {
                    tooshort[p][p+i]=1;
                }

            }

        }
    }


}
void collision()//將過近的兩個質點動量交換(不嚴謹，僅滿足動量守恆)
{
    #pragma omp parallel
    {
        #pragma omp for
        for (int a=0; a<n; a++)
        {
            for (int b=a+1; b<n; b++)
            {
                if (tooshort[a][b]==1)
                {
                    for (int d=0; d<3; d++)
                    {
                        float f = vel[a][d];
                        vel[a][d] = m[b]/m[a]*vel[b][d];
                        vel[b][d] = m[a]/m[b]*f;
                    }
                }
            }
        }
    }

}

#ifdef KD
void KDK()
{    
        # pragma omp parallel
        {
            # pragma omp for
            for (int p=0; p<n; p++)
            {
                for (int d=0; d<3; d++)
                {
                    //acc[p][d] = 0.0001*particle_force[p][d]/m[p];
                    acc[p][d] = particle_force[p][d]/m[p];
                    //KDK
                    vel[p][d] += acc[p][d]*0.5*dt; //K
                    pos[p][d] += vel[p][d]*dt;     //D
                    vel[p][d] += acc[p][d]*0.5*dt;  //K
                    //cout<<pos[p][d]<<endl;
                }


            }

        }

        distance();
        collision();
        boundery();
        periodic();


}
#endif

#ifndef KD
void DKD()//利用DKD更新位置以及速度，再利用distance()和collision()處理距離過近的質點。時間迴圈在合併時應可去掉
{
   
        # pragma omp parallel
        {
            # pragma omp for
            for (int p=0; p<n; p++)
            {
                //# pragma omp for
                for (int d=0; d<3; d++)
//total_mass_particle();


while(t<T)
{
    //NGP method
    if (method == 1)
    {
        NGP_par_mesh();
    }
    //CIC method
    if (method == 2) 
    {
        CIC_par_mesh();
    }
    if (method == 3) 
    {
        //distance();
        //collision();
        boundery();
        periodic();
    
}
#endif

void total_momentum()//計算各方向總動量, cout each direction of total momentum 
{
    float current_momentum[3]={0,0,0};
    int counter=1;
    if(counter==1)
    for (int d=0; d<3; d++)
    {
        for (int p=0; p<n; p++)
        {
            momentum[d]=momentum[d]+m[p]*vel[p][d];
        }
        
    }
    counter++;
    for (int d=0; d<3; d++)
    {
        for (int p=0; p<n; p++)
        {
                    current_momentum[d] = current_momentum[d] + m[p]*vel[p][d];
        }
        //cout<<"increased momentun "<<d<<" = "<<pow(current_momentum[0]*current_momentum[0]+current_momentum[1]*current_momentum[1]+current_momentum[2]*current_momentum[2],0.5) <<endl;

    }
    cout<<"increased momentun "<<" = "<<pow(current_momentum[0]*current_momentum[0]+current_momentum[1]*current_momentum[1]+current_momentum[2]*current_momentum[2],0.5)/n <<endl;
}

//main 
int main( int argc, char *argv[] )
{

//Time counter
float ti, tf;
ti = omp_get_wtime();
// ----------------  set openmp threads ----------------- //
omp_set_num_threads(NThread);

// ---------------- set FFTW threads ----------------- //
fftwf_init_threads();
fftwf_plan_with_nthreads(NThread);

printf("Number of Threads : %d \n \n",NThread );
// ---------------   init particles to axis ------------------ //
ofstream Datafile;
Datafile.open("Data.txt",ios::out);
Datafile<<"x_position"<<"\t"<<"y_position"<<"\t"<<"z_position"<<"\n";

// 1. allocate particle //
inti(); 
//test_potential();
//total_mass_particle();
#ifdef IsoBC
initforisolated() ;
#endif
#ifndef IsoBC
   initforperiodic();
#endif

while(t<T)
{
    //NGP method
    if (method == 1)
    {
        NGP_par_mesh();
    }
    //CIC method
    if (method == 2) 
    {
        CIC_par_mesh();
    }
    if (method == 3) 
    {
        TSC_par_mesh();
    }
    total_mass_grid();
    error_mass();
   

    // -----------FFT to get potential---------------- //
    #ifndef IsoBC
    initperiodicvalue();
    FFT();
    IRFFT();
    D1_to_3D(phi,phi1D);
    //print(phi);
    #endif

    #ifdef IsoBC
        initialisorhovalue();
        isoFFT();
        isoIRFFT();
        iso_D1_to_3D(phi,isophi);
        //isoprint(phi);
    #endif


    //Turn Potental into Force 
    //Grid Potential to Grid Force (Assume Period Potential)
    potential_to_force();

    //Insert Force Back to Particle
    //NGP method
    if(method == 1) NGP_force();
    //CIC method
    if(method == 2) CIC_force();
    //TSC method
    if(method ==3) TSC_force();



    //  ------------------ update particle position by force at particles --------------- //
    
    #ifdef KD
    KDK();
    #endif

    #ifndef KD
    DKD();
    #endif
    /**
    for(int i=0; i<n ; i++)
    {
        cout<< vel[i][0]<<" "<<vel[i][1]<<" "<<vel[i][2]<<endl;

    }
    */
    total_momentum();
    

    //output each particle x,y,z to file//
    
    for(int i=0;i<n;i++)
    {
        //cout<<pos[i][0]<<"\t"<<pos[i][1]<<"\t"<<pos[i][2]<<"\n"<<endl;
        Datafile<< pos[i][0]<<"\t"<<pos[i][1]<<"\t"<<pos[i][2]<<"\n";
    }
    

    t=t+dt;
    //cout<<t<<endl;
    reset();
    

}// end while

// --------------- delete pointer of FFT ----------------------- //
    #ifndef IsoBC
    deleteforperiodic();
    #endif
    #ifdef IsoBC
    deleteforisolated();
    #endif
    //time counter
    tf= omp_get_wtime() ;
    float time = tf-ti ;
    printf("Execution time : %.8f",time);
    //cout<<"execution time "<<time<<endl;
    Datafile.close();
    return EXIT_SUCCESS;

}














