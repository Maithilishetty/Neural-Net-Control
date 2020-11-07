clc;close all;clear all;
a = 1; 
b = 1;
m0 = 5;
m1 = 18;
m2 = 1;
w1 = rand(m1,m0+1);
w2 = zeros(m2,m1+1);
lambda = 0.001;
P = (1/lambda)*eye(m1+1);
n = 0:1:1000;
yp = [0 0 0];
yphat = [0 0 0];


%finding u
for k = 1:length(n)+1
    if(k<=500)
        u(k) = sin(2*pi*k/250);
    else
        u(k) = 0.8*sin(2*pi*k/250) + 0.2*sin(2*pi*k/25);
    end
end

for k=1:length(n)
   
        p = [1;yp(end);yp(end-1);yp(end-2);u(k+1);u(k)];
        
        
        %FORWARD PASS
        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 1
        yphat1 = v2 ;
        yphat = [yphat yphat1];
        
        yp1 = (yp(end)*yp(end-1)*yp(end-2)*u(k)*(yp(end-2) - 1) + u(k+1))/(1 + yp(end-2)^2 + yp(end-1)^2);
        yp = [yp yp1];
        
        %Error Updation
        E = yp1 - yphat1;
        
        %weight updation
        P1 = P - (P*y1_k*y1_k'*P)/(1 + y1_k'*P*y1_k);
        P = P1;
        w2 = w2 + E*y1_k'*P1;
      

end


% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %

plot(1:length(n)+1,yp(3:end),'r-',1:length(n)+1,yphat(3:end),'g--');
legend('Neural Network Output', 'Desired Output');
title('Example 4 Function Approximation - OSLA');
g=var(yp(3:end)-yphat(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-yphat(3:end)).^2)/(length(yp(3:end)));
disp(mse);