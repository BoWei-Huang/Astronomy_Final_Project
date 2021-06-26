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
int edge[n][d]; //�P�_�Ӳɤl�O�_�I����ɡA�O�h��1�A�_�h��0


void init() //��l����A�]�t��m�B���O�B��q�A�H�ΥO��t�׬��s�A�X�֮ɤ��Ω�
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
void boundery()//periodical ���
{
    for (int p=0; p<n; p++)
            {
                for (int d=0; d<3; d++)
                {
                    if(Pos[p][d]>=0.5*L)//�����
                    {
                        edge[p][d] = 1;
                    }
                    else if(Pos[n][d]<=-0.5*L)//�t���
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

void distance()//�p���p�ӽ��I�M��p+i�ӽ��I���Z������O�_�p��p��@�w��(dd)�A�}�Ctooshort���A1�N��Z���L��A0�N��Z������
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
void collision()//�N�L�񪺨�ӽ��I�ʶq�洫(���Y�ԡA�Ⱥ����ʶq�u��)
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
void KDK()//�Q��KDK��s��m�H�γt�סA�A�Q��distance()�Mcollision()�B�z�Z���L�񪺽��I�C�ɶ��j��b�X�֮����i�h��
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
void DKD()//�Q��DKD��s��m�H�γt�סA�A�Q��distance()�Mcollision()�B�z�Z���L�񪺽��I�C�ɶ��j��b�X�֮����i�h��
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

int main()//�D�禡�A��X�̫᪺���G�A�ȴ��եΡA�X�֮ɥi��������������
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
