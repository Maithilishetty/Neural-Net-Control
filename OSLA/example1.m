clc;
close all;
clear all;

a = 1;
b = 1;
m0 = 1;
m1 = 10;
m2 = 1;
w1 = rand(m1,m0+1);
w2 = zeros(m2,m1+1);
lambda = 0.001;
P = (1/lambda)*eye(m1+1);
n=0:1:1000;
yp = [0 0];
yphat = [0 0];

for k = 1:length(n)
    
    u = sin(2*pi*k/250);
    p = [1;u];

    v1 = w1*p;
    phi_v1 = a*tanh(b*v1);
    y1 = [1 ; phi_v1];
    v2 = w2*y1;
    y2=v2;
    yphat1 = y2 + 0.3*yp(end) + 0.6*yp(end-1);
    yphat = [yphat yphat1];
    
    yp1 = 0.3*yp(end) + 0.6*yp(end-1) +0.6*sin(pi*u) + 0.3*sin(3*pi*u) + 0.1*sin(5*pi*u);
    yp=[yp yp1];
    

    e = yp1 - yphat1;
    
    
    P1 = P - (P*y1*y1'*P)/(1 + y1'*P*y1);
    P=P1;
    w2 = w2 + e*y1'*P1;
    
    
end
% K = 0:length(n)+1;
% figure;
% plot(K,yphat,'-r',K,yp,'-g')
plot(1:length(n),yp(3:end),'r-',1:length(n),yphat(3:end),'g--');
legend('Neural Network Output', 'Desired Output');
title('Example 1 Function Approximation - OSLA');


%VAF Calculation
g=var(yp(3:end)-yphat(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-yphat(3:end)).^2)/(length(yp(3:end)));
disp(mse);