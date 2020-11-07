clc;
close all;
clear all;

a = 1; 
b = 1;
m0 = 2;
m1 = 18;
m2 = 1;
w1 = rand(m1,m0+1);
w2 = zeros(m2,m1+1);
lambda = 0.001;
P = (1/lambda)*eye(m1+1);
n = 0:1:100;
yp = [0 0];
yphat = [0 0];

for k = 1:length(n)
    
    u = sin(2*pi*k/25);
    
    p = [1;yp(end);yp(end-1)];
    %FORWARD PASS
    v1 = w1*p;
    phi_v1 = a*tanh(b*v1);
    y1 = [1 ; phi_v1];
    v2 = w2*y1;
    y2=v2;
    yphat1 = y2 + u;
    yphat=[yphat yphat1];
    
    yp1 = yp(end)*yp(end-1)*(yp(end) + 2.5)/(1 + (yp(end))^2 + (yp(end-1))^2) + u;
    yp=[yp yp1];
    
    %ERROR CALCULATION
    e = yp1 - yphat1;
    
    %BACKWARD PASS
    P1 = P - (P*y1*y1'*P)/(1 + y1'*P*y1);
    P=P1;
    w2 = w2 + e*y1'*P1;
    
end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %

% K = 0:length(n)+1;
% figure;
% plot(K,yphat,'--r',K,yp,'-g')
plot(1:length(n),yp(3:end),'r-',1:length(n),yphat(3:end),'g--');
legend('Neural Network Output', 'Desired Output');
title('Example 2 Function Approximation - OSLA');
% axis([0 100 -4 4]);



%VAF Calculation
g=var(yp(3:end)-yphat(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-yphat(3:end)).^2)/(length(yp(3:end)));
disp(mse);