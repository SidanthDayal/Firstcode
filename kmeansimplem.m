%Sidanth Dayal
%Chemical Engineer in training (Honors)
%University of KwaZulu-Natal
%Faculty of Engineering
%Howard College
% 214500832@stu.ukzn.ac.za
% sidanthdayal@gmail.com
% https://github.com/SidanthDayal
%% Program
%This code uses K-means clustering to separate scattered data (x-y) into  
%k clusters.
clear all;clc
%sample
%x=[42.9000000000000;75.7000000000000;37.7000000000000;57.4000000000000;56.5000000000000;32.4000000000000;52.2000000000000;38.1000000000000;48.7000000000000;35.7000000000000;83.7000000000000;79;72.6000000000000;56.8000000000000;66;61.2000000000000;42;37.2000000000000;65];
%y=[16.8880000000000;32.9040000000000;13.8120000000000;28.1580000000000;40.2050000000000;5.94900000000000;21.4690000000000;23.6810000000000;27.1630000000000;22.2040000000000;66.2630000000000;49.5440000000000;28.4160000000000;12.3910000000000;22.4030000000000;18.0050000000000;15.6770000000000;12.0800000000000;43.1180000000000];
%NOT AUTOMATIC
x=[];y=[]; %enter or import scatter data
%% Main routine
k=2;%number of clusters
cluster1=[];%create more arrays called clusterk for k>2
cluster2=[];
a=1;
cent1x=x(1);cent2x=x(3);cent1y=y(1);cent2y=y(3);%initialize centroids
while a<2000
    cluster1=[cent1x cent1y];
cluster2=[cent2x cent2y];
for index=1:length(x)
   j1=sqrt(((x(index)-cent1x).^2)+((y(index)-cent1y).^2));
   j2=sqrt(((x(index)-cent2x).^2)+((y(index)-cent2y).^2));
   if j1<j2
       cluster1=[cluster1 ;x(index) y(index)];
   end
   if j2<j1
       cluster2=[cluster2 ;x(index) y(index)];
   end
end
a=a+1;
cent1x=mean(cluster1(:,1));cent2x=mean(cluster2(:,1));cent1y=mean(cluster1(:,2));cent2y=mean(cluster2(:,2));
end
figure(2)
plot(cluster1(:,1),cluster1(:,2),'b*')
hold on
plot([cent1x cent2x],[cent1y cent2y],'k')
m=(cent1y- cent2y)/(cent1x- cent2x);
bisectorm=-1/m;
xx=linspace(0,300,121);
midyy=0.5*(cent1y+cent2y);
midxx=0.5*(cent1x+cent2x);
cc=midyy-bisectorm*midxx;
ccc=cent1y-m*cent1x;
yy=bisectorm.*xx+cc;
yyy=m*xx+ccc;
hold on 
plot(xx,yy,'r')
hold on
plot(xx,yyy,'k')
hold on
plot(cluster2(:,1),cluster2(:,2),'m*')
plot([cent1x cent2x],[cent1y cent2y],'r*','LineWidth',4)
xlim([min(x) max(x)])
xlabel('Independent data','Interpreter','Latex','FontSize',14)
ylabel('Dependent data','Interpreter','Latex','FontSize',14)

