clc;close all;clear all;
a = 1;
b = 1;
m0 = 1;
m1 = 10;
m2 = 1;

w1 = rand(m1,m0+1);
w2 = zeros(m2,m1+1);

w3 = rand(m1,m0+1);%20 x 3
w4 = zeros(m2,m1+1);%10 x 21

lam1 = 0.001;
lam2 = 0.001;

P1 = ((1/lam1)*eye(m1+1));
P2 = ((1/lam2)*eye(m1+1));

n=0:1:10000;

yp=[0];
yphat=[0];



for k=1:length(n)
       
       u = sin(2*pi*k/25)+sin(2*pi*k/10) ;
       
       p = [1;yp(end)];
       
       up = u^3;

       %FORWARD PASS
       v1 = w1*p;%20 x 1 
       phi_v1 = a*tanh(b*v1); %20 x 1
       yp_k = [1 ; phi_v1];%21 x 1
       v2 = w2*yp_k; %10 x 1
       y2 = v2 ;
      
       p2 = [1;u];

       %FORWARD PASS
       v3 = w3*p2;%20 x 1 
       phi_v3 = a*tanh(b*v3); %20 x 1
       u_k = [1 ; phi_v3];%21 x 1
       v4 = w4*u_k; %10 x 1
       y4 = v4 ;
        
       yphat1= y2 + y4;
       yphat = [yphat yphat1];
       
       yp1 =(yp(end)/(1 + (yp(end))^2))+ up;
       yp = [yp yp1];
       
       % error calculation
       
       E = yp1 - yphat1;
        
       Ph = P1 - (P1*(yp_k*yp_k')*P1)/(1 + yp_k'*P1*yp_k);
       P1 = Ph;
       w2 = w2 + E*yp_k'*Ph;
       
       %BACKWARD PASS
       Pt = P2 - (P2*(u_k*u_k')*P2)/(1 + u_k'*P2*u_k);
       P2 = Pt;
       w4 = w4 + E*u_k'*Pt;
       
       
        
end
plot(0:length(n)-2,yp(3:end),'r-',0:length(n)-2,yphat(3:end),'g--');
legend('Neural Network Output', 'Desired Output');
title('Example 3 Function Approximation - OSLA');
axis([0 500 -4 4]);
g=var(yp(3:end)-yphat(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-yphat(3:end)).^2)/(length(yp(3:end)));
disp(mse);
