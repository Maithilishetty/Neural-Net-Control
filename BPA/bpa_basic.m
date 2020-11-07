
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File name: bpa.m                                                        %
% Neural Network Controllers                                              %                                                     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;clear all;close all;

% ----------------------------------------------------------------------- %
%                           Variables Declaration                         %
% ----------------------------------------------------------------------- %

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% m0 = number of neurons in the input layer                               %
% m1 = number of neurons in the hidden layer                              %
% m2 = number of neurons in the output layer                              % 
% w1 = weights to the hidden layer                                        %
% w2 = weights to the output layer                                        %
% eta1 = learning rate parameter                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = 1; b = 1;
m0 = 1;
m1 = 5;
m2 = 1;
w1 = rand([m1 m0+1]);
w2 = rand([m2 m1+1]);
disp('Initial Weight Vectors');
disp(w1);
disp(w2);
disp('----');
bk = 1;
eta1 = 0.1;
train_in = 1:1:1000;
train_in = train_in';
train_out = sin(0.01*train_in);

for j = 1:length(train_in)
    
    x = j;
    x_k = [bk ; x];% 2*1
    v1 = w1*x_k;% 5*1
    
    %FORWARD PASS
    
    phi_v1 = a*tanh(b*v1);
    y1_k = [1 ; phi_v1];
    v2 = w2*y1_k;
    phi_v2 = v2; %linear activation function
    y2_k(j) = v2;
    e_k(j) = train_out(j) - y2_k(j);
    phi_v2_diff = 1;
    phi_v1_diff = (1-phi_v1.^2);
    
    %BACKWARD PASS
    
    delta2 = e_k(j).*phi_v2_diff;
    delta_w2 = eta1*delta2*(y1_k)';
    delta1 = (w2(2:length(w2))'.*delta2).*(phi_v1_diff);
    delta_w1 = eta1*delta1*(x_k)';
    
    %WEIGHT UPDATION
    
    w1 = w1+delta_w1;
    w2 = w2+delta_w2;
    
end

err = mse(e_k);
disp('The error is: ');
disp(err)

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %

figure(1);
subplot(211);
plot(train_in,y2_k,'-r',train_in,train_out,'--g')
legend('Neural Network Output', 'Desired Output');
title('Sine Function Approximation');
subplot(212);
plot(train_in,e_k);
title('Mean Square Error');
axis([0 1000 -3 3])
disp('Final Weight Vectors');
disp(w1);
disp(w2);
    
    

