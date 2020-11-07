
clc;
close all;
clear all;
a = 1; 
b = 1;
m0 = 1;
m1 = 18;
m2 = 1;
w1 = rand(m1,m0+1);
w2 = zeros(m2,m1+1);
lambda = 0.001;
P = (1/lambda)*eye(m1+1);
n = 0:1:1000;
yp = [0];
yphat = [0];

for k = 1:length(n)
    
        u=sin(2*pi*k/25);
        p = [1;u];%2*1
        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 1
        yphat1 = v2 + 0.8*yphat(end);
        yphat = [yphat yphat1];
       
        f_u(k)=(u-0.8).*u.*(u+0.5);
        yp1= 0.8*yp(end)+f_u(k);
        yp = [yp yp1];
         
        e_k=yp1-yphat1;
        
        P1 = P - (P*y1_k*(y1_k')*P)/(1+((y1_k')*P*y1_k));
        P = P1;
        w2=w2+e_k*(y1_k')*P1;
end
       
plot(1:length(n)-1,yp(3:end),'r-',1:length(n)-1,yphat(3:end),'g--');
legend('Neural Network Output', 'Desired Output');
title('Example 6 Function Approximation - OSLA');
axis([0 500 -2.5 1])
g=var(yp(3:end)-yphat(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-yphat(3:end)).^2)/(length(yp(3:end)));