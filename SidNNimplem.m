%Sidanth Dayal
%Chemical Engineer in training (Honors)
%University of KwaZulu-Natal
%Faculty of Engineering
%Howard College
% 214500832@stu.ukzn.ac.za
% sidanthdayal@gmail.com
% https://github.com/SidanthDayal
%% Program
% NEURAL NETWORK template that contains the condensed version of the back-
%propagation algorithm with a sigmoid transfer function. The architecture
%is a 3 feedforward Neural network with a topology that the user can
%change.
%By looking at the code, one will see a neural network that is designed to
%learn the dynamic state behaviour of a 2 interacting tank system and
%suggest control settings based on what level setpoint one sets. 
%This file simply is to train the Neural Network on the differential
%equations. THE NETWORK WEIGHTS STORE THE SYSTEM DYNAMICS IN ITS MEMORY 
%% Generate data
% Manipulated variables
F0=[0.0024 0.00245 0.0025 0.0026  0.0027 0.003 0.00305 0.0031 ];
p=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];[F0mesh,pmesh]=meshgrid(F0,p);%p by F0
%% C0NSTANTS
dtank=1;Atank=pi.*0.25.*dtank.^2;L=1;
dpipe=0.05;Apipe=pi.*0.25.*dpipe.*dpipe;eta=Apipe/Atank;g=9.8;
Kg=sqrt(2.*g);Kcv=sqrt(2*g);
f=0.024;
coeff=f*L/dpipe;
%% Containers
[h11,h22]=meshgrid(linspace(0,1,10),linspace(0,1,10));
dh1cell=cell(size(F0mesh));
dh2cell=cell(size(F0mesh));
dh2dh1cell=cell(size(F0mesh));
%% valve allows direction in both ways
for index=1:size(F0mesh,2)
    for jindex=1:size(F0mesh,1)
        dh11dt=(F0mesh(jindex,index)/Atank)-sign(h11-h22).*(Apipe.*pmesh(jindex,index).*Kcv.*(sqrt(abs(h11-h22)./(1+coeff)))./Atank);
        dh22dt=sign(h11-h22).*(Apipe.*pmesh(jindex,index).*Kcv.*(sqrt(abs(h11-h22)./(1+coeff)))./Atank)-(eta.*0.6.*Kg*(sqrt(h22./(1+coeff))));
dh22dh11=dh22dt./dh11dt;
dh1cell{jindex,index}=dh11dt;
dh2cell{jindex,index}=dh22dt;
anglecell{jindex,index}=atand(dh22dt./dh11dt);
magcell{jindex,index}=sqrt((dh11dt.^2)+(dh22dt.^2));
    end 
end

%% sort input data and labels and initialize weights
%weights for 3 layers
m=4;%number of inputs
n=3;%number of hidden neurons
o=2;%number of outputs
LR=0.35;alph=0.15;
alimit=-1;blimit=1;
weights1=(blimit-alimit).*rand(m,n)+alimit;
weights2=(blimit-alimit).*rand(n,o)+alimit;
delw1=zeros(m,n);delw2=zeros(n,o);
dataError=zeros(size(h11));
subSetError=zeros(size(dh2dh1cell));
epochs=20000;
totalNetworkerror=zeros(epochs,1);
%% NN training
for epoch=1
    for index=1:1
        for jindex=1:1
            angleS=anglecell{jindex,index}./(max(max(abs(anglecell{jindex,index}))));%scaled
            magnit=magcell{jindex,index}./(max(max(abs(magcell{jindex,index}))));%scaled
            for subindex=1:1
                for subjindex=3
                    inputData=[h11(subjindex,subindex); h22(subjindex,subindex) ;angleS(subjindex,subindex);magnit(subjindex,subindex)];%scaled
                    labelData=[(F0mesh(jindex,index)/(max(max(F0mesh)))) ;(pmesh(jindex,index)/(max(max(pmesh))))];%scaled
                    % FEEDFORWARD
                    A1H1=1./(1+exp(-1.*inputData));
                    A2in=transpose(weights1)*A1H1;
                    AH1O=1./(1+exp(-1.*A2in));
                    A3in=transpose(weights2)*AH1O;
                    NNoutput=1./(1+exp(-1.*A3in));
                    temp=0.5.*((labelData-NNoutput).^2);
                    dataError(subjindex,subindex)=(sum(temp,1));
                    subSetError(jindex,index)=(sum(sum(dataError,1),2))./(numel(dataError));
                    totalNetworkerror(epoch,1)=(sum(sum(subSetError,1),2))./(numel(subSetError));%plotted versus epochs
                    %BACKPROPAGATION
                    del2=NNoutput.*(1-NNoutput).*(labelData-NNoutput);
                    del1=AH1O.*(1-AH1O).*(weights2*del2);
                    delw2=LR*(AH1O*del2')+alph*(delw2);
                    delw1=LR*(A1H1*del1')+alph*(delw1);
                    weights2=weights2+delw2;
                    weights1=weights1+delw1;
                end
            end
        end
    end        
end
        
        
    






