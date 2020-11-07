clc;close all;clear all;
a =1; b = 1;
m0 = 3;
m1 = 20;
m2 = 1;

w1 = 0.001*rand(m1,m0+1);
w2 = zeros(m2,m1+1);
lambda=0.0001;
P = (1/lambda)*eye(m1+1);

yp = [0 0 0];
ym= [0 0 0];
u =[0 0];
n= 0:10000;

for k =1:length(n)
    
        p = [1;yp(end);yp(end-1);yp(end-2)];
        r1 = sin(2*pi*k/25);
        y =(5*yp(end)*yp(end-1)/(1 + (yp(end))^2 + (yp(end-1))^2+ (yp(end-2))^2))+ u(end)+(1.1)*u(end-1);
        ya = 0.32*ym(end) + 0.64*ym(end-1)-0.5*ym(end-2)+ r1;

        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k;
       
        y3 = v2 ;
        E  = y - y3;
        
        
        P1 = P - (P*y1_k*y1_k'*P)/(1 + y1_k'*P*y1_k);
        P = P1;
        w2 = w2 + E*y1_k'*P1;


        u1 = (1/1)*(-y3 -(1.1)*u(end)+0.32*yp(end) + 0.64*yp(end-1)-0.5*yp(end-2)+ r1);
        
        yp=[yp y];
        ym=[ym ya];
        u=[u u1];
end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %
K =0:length(n)+2;
figure;
plot(K,yp,'-r',K,ym,'--g')
axis([0 50  -10 10])
legend('Plant Output', 'Reference Model Output');
title('Example 9 with Control- OSLA');

K=0:length(n)+1;
figure;
plot(K,u)
axis([0 50 -300 300])
title('Example 9 oscillatory input');

g=var(yp(3:end)-ym(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);