clc;close all;clear all;

a = 1;
b = 1;
m0 = 2;
m1 = 18;
m2 = 1;
w1 = rand(m1,m0+1);
w2 = zeros(m2,m1+1);
lambda = 0.0001;
P = (1/lambda)*eye(m1+1);
n=0:1:100000;
yp = [0 0];
ym = [0 0];
u = 0;


for k = 1:length(n)
    
        p = [1;yp(end);yp(end-1)];
        
        r = sin(2*pi*k/25);
        
        yp1 = yp(end-1)*yp(end)*(yp(end) + 2.5)/(1 + (yp(end))^2 + (yp(end-1))^2) + u;

        ym1 = 0.6*ym(end) + 0.2*ym(end-1) + r;
        
        %FORWARD PASS
        
        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); 
        y1 = [1 ; phi_v1];
        v2 = w2*y1;
        y2 = v2;
        yphat = y2 + u;        
        
        e =  yp1 - yphat;
        
        %BACKWARD PASS
    
        P1 = P - (P*y1*y1'*P)/(1 + y1'*P*y1);
        P = P1;
        w2 = w2 + e*y1'*P1;        
    
        u = -y2 + 0.6*yp(end) + 0.2*yp(end-1) + r;
        yp = [yp yp1];
        ym=[ym ym1];
      
end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %


K = 0:length(n)+1;
figure;
plot(K,yp,'-r',K,ym,'--g')
legend('Plant Output', 'Reference Model Output');
title('Example 7 with control - OSLA');
axis([0 500 -4 6]);
g=var(ym(3:end)-yp(3:end));
h=var(ym(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);
