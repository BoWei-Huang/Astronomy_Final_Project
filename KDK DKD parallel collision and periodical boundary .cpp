#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<cstdlib>
#include<ctime>
#include<cmath>
#include<omp.h>
#include<time.h>
using namespace std;
const int n = 2;        //number of particles
const int d = 3;        //dimension
double m[n];            //mass of each particle
double Pos[n][d];       //Position of particles
double Vel[n][d];       //Velocity of particles
double Acc[n][d];       //Acceleration of particles
double F[n][d];         //Force on particles
int tooshort[n][n];  //Recording whether any two particles are too close
double dt = 0.01;       //renew every dt seconds
double T = 2;        //total time(unit: second)
double L = 1;
int edge[n][d]; //判斷該粒子是否碰到邊界，是則為1，否則為0


void init() //初始條件，包含位置、受力、質量，以及令初速度為零，合併時不用放
{
Pos[0][0]=0.2;
Pos[0][1]=0;
Pos[0][2]=0;

Pos[1][0]=-0.3;
Pos[1][1]=0;
Pos[1][2]=0;

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
            Vel[p][d] = 0;
        }
    }
}
void boundery()//periodical 邊界
{
    for (int p=0; p<n; p++)
            {
                for (int d=0; d<3; d++)
                {
                    if(Pos[p][d]>=0.5*L)//正邊界
                    {
                        edge[p][d] = 1;
                    }
                    else if(Pos[n][d]<=-0.5*L)//負邊界
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
                Pos[p][d]=Pos[p][d]-L;
                //cout<<Pos[p][d]<<endl;
            }
            else if(edge[p][d] == 2)
            {
                Pos[p][d]=Pos[p][d]+L;
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
                double dd = abs(Pos[p][0]-Pos[p+i][0])*abs(Pos[p][0]-Pos[p+i][0])+
                            abs(Pos[p][1]-Pos[p+i][1])*abs(Pos[p][1]-Pos[p+i][1])+
                            abs(Pos[p][2]-Pos[p+i][2])*abs(Pos[p][2]-Pos[p+i][2]);
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
                        double f = Vel[a][d];
                        Vel[a][d] = m[b]/m[a]*Vel[b][d];
                        Vel[b][d] = m[a]/m[b]*f;
                    }
                }
            }
        }
    }





}
void KDK()//利用KDK更新位置以及速度，再利用distance()和collision()處理距離過近的質點。時間迴圈在合併時應可去掉
{

    for (double t=0; t<T; t=t+dt)
    {
        # pragma omp parallel
        {
            # pragma omp for
            for (int p=0; p<n; p++)
            {

                //# pragma omp for
                for (int d=0; d<3; d++)
                {
                    Acc[p][d] = F[p][d]/m[p];
                    //KDK
                    Vel[p][d] += Acc[p][d]*0.5*dt; //K
                    Pos[p][d] += Vel[p][d]*dt;     //D
                    Vel[p][d] += Acc[p][d]*0.5*dt;  //K
                    //cout<<Pos[p][0]<<endl;
                }


            }

        }

        distance();
        collision();
        boundery();
        periodic();
    }
}
void DKD()//利用DKD更新位置以及速度，再利用distance()和collision()處理距離過近的質點。時間迴圈在合併時應可去掉
{
    for (double t=0; t<T; t=t+dt)
    {
        # pragma omp parallel
        {
            # pragma omp for
            for (int p=0; p<n; p++)
            {
                //# pragma omp for
                for (int d=0; d<3; d++)
                {
                    Acc[p][d] = F[p][d]/m[p];
                    //DKD
                    Pos[p][d] += Vel[p][d]*0.5*dt; //D
                    Vel[p][d] += Acc[p][d]*dt;     //K
                    Pos[p][d] += Vel[p][d]*0.5*dt; //D
                }
            }
        }

        distance();
        collision();
        boundery();
        periodic();
    }
}

int main()//主函式，輸出最後的結果，僅測試用，合併時可直接忽略此部分
{
    init();
    KDK();

     for (int p=0; p<n; p++)
        {
            for (int d=0; d<3; d++)
            {
                cout<<Pos[p][d]<<" ";
            }
            cout<<endl;
        }
}
