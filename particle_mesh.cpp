#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <omp.h>

//define the const
int method = 1 ; //(1:NGP method ; 2:CIC method)
float G = 1.0 ;
int n = 10; // number of paritcle
int N = 10; // number of grid
int N_total=N+2;  //include boundary(buffer) 
float L = 1.0 ;// size of box
float dx=L/N;


//Define Weighting Function
float W(float x_p,float x_g ){
    float d=fabs(x_p-x_g);
    float w=0;
    if (d <= dx){
        w=1-d/dx;
    } else{
        w=0;
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

int main( int argc, char *argv[] )
{
//Time counter
double ti, tf;
ti = omp_get_wtime();

//Set Number of Threads
const int NThread = 2 ;
omp_set_num_threads( NThread );
printf("Number of Threads : %d \n \n",NThread );

//initializaion
//Define Mass
    double m[n]={0.0};
    for (int i=0;i<n;i++){
        m[i]=1.0;
    }

//Define Position
    float pos[n][3]={0};
    for (int i=0;i<n;i++){
        for (int j=0;j<3;j++){
            pos[i][j]=L * rand() / (RAND_MAX + 1.0) + -L/2;
        }
    }

//Define Velocity
    float vel[n][3]={0};
    for (int i=0;i<n;i++){
        for (int j=0;j<3;j++){
            vel[i][j]=(L * rand() / (RAND_MAX + 1.0) + -L/2)/10;
        }
		}

//Define Coordinate 
    float x[N_total]={0.0};
    float y[N_total]={0.0};
    float z[N_total]={0.0};
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
    
//define error (the total mass of particle)
float total = 0.0;
for (int i = 0; i<n; i++){
	total = total+ m[i];
	}
	printf ("the total mass of particle : %.2f \n " , total);
	printf ("\n");

//Inject the particle mass into grid
	double grid_mass[N][N][N] = {0.0}; //Define empty grid mass
	double num_mass[N_total][N_total][N_total]={};
	
//Define relevent index set
	int index[n][3][2]={};
	
//NGP method
if (method == 1) {
printf("method %d : NGP\n",method);

//detect the particle pos
	# pragma omp parallel for
	for (int i=0; i<n; i++){

	int x_pos = floor(N*pos[i][0]+L*N/2);
	int y_pos = floor(N*pos[i][1]+L*N/2);
	int z_pos = floor(N*pos[i][2]+L*N/2);

//	printf ("Grid of the particle %d, x= %d, y= %d, z= %d \n",i, x_pos, y_pos, z_pos);
	
//assign the mass to grid
	grid_mass[x_pos][y_pos][z_pos] = grid_mass[x_pos][y_pos][z_pos]+m[i];

	}
printf("\n");


//show the position 
/*
# pragma omp parallel for
for (int i=0 ; i<n;i++){
	printf( "Position of Particle %d\n", i ); 
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

//calculate the mass error
float total_array = 0.0;
# pragma omp parallel for
for (int i=0 ; i<N;i++){
	# pragma omp parallel for
	for (int j=0 ; j<N;j++){
		# pragma omp parallel for
		for (int k=0 ; k<N;k++){
			total_array = total_array + grid_mass[i][j][k];
			}
		}
	}
   printf ("the total mass of particle in grid : %.2f \n " , total_array);
   printf ("\n");

float error = total - total_array ;
	printf ("the error of mass : %.2f \n " ,error );
	printf ("\n");

}

//CIC method
if (method == 2) {
printf("method %d : CIC\n",method);

//Mass Distribution of nth particle
//3D Cloud in Cell
	# pragma omp parallel for
    for (int i=0;i<n;i++){
//find the index of relevent eight grid in CIC
		//# pragma omp parallel for
        for (int j=0; j<3; j++){
            index[i][j][0]=floor((pos[i][j]+L/2-dx/2)/dx);
            index[i][j][0]=(index[i][j][0]+1+N_total)%N_total;
            index[i][j][1]=(index[i][j][0]+1)%N_total;
        } 
//Mass Assignment
	//# pragma omp parallel for
        for (int j=0; j<2; j++){
        	//# pragma omp parallel for
            for (int k=0; k<2; k++){
            	//# pragma omp parallel for
                for (int p=0; p<2; p++){
                    num_mass[index[i][0][j]][index[i][1][k]][index[i][2][p]]=num_mass[index[i][0][j]][index[i][1][k]][index[i][2][p]]+m[i]*W(pos[i][0],x[index[i][0][j]])*W(pos[i][1],y[index[i][1][k]])*W(pos[i][2],z[index[i][2][p]]);
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
    for (int i=1;i<N_total-1;i++){
        for (int j=1;j<N_total-1;j++){
            for (int k=1;k<N_total-1;k++){
                grid_mass[i-1][j-1][k-1]=num_mass[i][j][k];
            }
        }
    }
    
printf("\n");

/*
//show the position 
for (int i=0 ; i<n;i++){
	printf( "Position of Particle %d\n", i ); 
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

//calculate the mass error
float total_array = 0.0;
for (int i=0 ; i<N;i++){
	for (int j=0 ; j<N;j++){
		for (int k=0 ; k<N;k++){
			total_array = total_array + grid_mass[i][j][k];
			}
		}
	}
   printf ("the total mass of particle in grid : %.2f \n " , total_array);
   printf ("\n");
   
float error = total - total_array ;
printf ("the error of mass : %.2f \n " ,error );
printf ("\n");
}
















//Turn Potental into Force 


//Define Potential
    float phi[N][N][N]={};

//Testing Potential :Plummer's model
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                phi[i][j][k]=-G/sqrt(x[i]*x[i]+y[j]*y[j]+z[k]*z[k]+1);
            }
        }
    }
//Assume Period Potential
//Define Grid Force of Three Dimension
    float grid_force_x[N][N][N]={};
    float grid_force_y[N][N][N]={};
    float grid_force_z[N][N][N]={};

//Grid Potential to Grid Force
//x-direction
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
            }
        }
    }

//y-direction
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
                if (j==0) {
                    grid_force_y[i][j][k]=-grid_mass[i][j][k]*(phi[i][j+1][k]-phi[i][N-1][k])/(2*dx);
                } else if (j==(N-1)) {
                    grid_force_y[i][j][k]=-grid_mass[i][j][k]*(phi[i][0][k]-phi[i][j-1][k])/(2*dx);
                } else {
                    grid_force_y[i][j][k]=-grid_mass[i][j][k]*(phi[i][j+1][k]-phi[i][j-1][k])/(2*dx);
                }
            }
        }
    }

//z-direction
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){
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

//Insert Force Back to Particle
float particle_force[n][3]={};

//NGP method
if(method == 1){
//	# pragma omp parallel for
	for (int i=0;i<n;i++){

	int x_pos = floor(N*pos[i][0]+L*N/2);
	int y_pos = floor(N*pos[i][1]+L*N/2);
	int z_pos = floor(N*pos[i][2]+L*N/2);

	particle_force[i][0] = grid_force_x[x_pos][y_pos][z_pos];
	particle_force[i][1] = grid_force_y[x_pos][y_pos][z_pos];
	particle_force[i][2] = grid_force_z[x_pos][y_pos][z_pos];

	printf("the force one the particle %d\n Fx = %.2f Fy = %.2f Fz = %.2f \n",i ,particle_force[i][0],particle_force[i][1],particle_force[i][2]);

	}
}

printf ("\n");
tf= omp_get_wtime() ;
double time = tf-ti ;
printf("Take time : %.8f s",time);

    return EXIT_SUCCESS;
}

