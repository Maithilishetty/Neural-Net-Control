clc;
close all;
clear all;

a = 1; b = 0.25;
m0 = 2;
m1 = 30;
m2 = 1;

w11 = 0.1*randn(m1,m0+1);
w21 = 0.1*rand(m2,m1+1);

w12 = 0.1*randn(m1,m0+1);
w22 = 0.1*rand(m2,m1+1);

lambda= 0.001;

P1=(1/lambda)*eye(m1+1);
P2=(1/lambda)*eye(m1+1);

yp1 = 0;
yp2 = 0;
ym1 = 0;
ym2 = 0;
u1 = 0;
u2 = 0;
s = 0:10000;

for k = 1:length(s)
        r1=sin(2*pi*k/25);
        r2=cos(2*pi*k/25);
        
        yp1_k = yp1(end)/(1+yp2(end)^2) + u1;
        ym1_k = 0.6*ym1(end) + 0.2*ym2(end) + r1;
        yp2_k = (yp1(end)*yp2(end))/(1 + (yp2(end))^2) + u2;
        ym2_k = 0.1*ym1(end) - 0.8*ym2(end) + r2;
        
         x_k = [yp1(end);yp2(end)];
         v0_k=[1;x_k];
        
        v11_k=a*tanh(b*w11*v0_k);%20*1
        v11_k=[1;v11_k];%21*1
        yn1=w21*v11_k;%1x1
%         yp1_cap=[yp1_cap;yn1+u1];
       
        v12_k=a*tanh(b*w12*v0_k);%20*1
        v12_k=[1;v12_k];%21*1
        yn2=w22*v12_k;%1x1
%         yp2_cap=[yp2_cap;yn2+u2];
        
        e_k1(k)=yp1(end)/(1+yp2(end)^2)-yn1;
        e_k2(k)=(yp1(end)*yp2(end))/(1 + (yp2(end))^2)-yn2;
        
        P_k1=P1 - (P1*v11_k*(v11_k')*P1)/(1+((v11_k')*P1*v11_k));
        P1=P_k1;
        w21=w21+e_k1(k)*(v11_k')*P_k1;
        
        P_k2=P2 - (P2*v12_k*(v12_k')*P2)/(1+((v12_k')*P2*v12_k));
        P2=P_k2;
        w22=w22+e_k2(k)*(v12_k')*P_k2;
        
   
        u1 = -yn1 + 0.6*yp1(end) + 0.2*yp2(end) + r1;
        u2=  -yn2 + 0.1*yp1(end)-0.8*yp2(end)+ r2;
        

        yp1 = [yp1;yp1_k];
        yp2 = [yp2;yp2_k];
        ym1 = [ym1;ym1_k];
        ym2 = [ym2;ym2_k];
        
          
end
        
% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %
K = 0:length(s);
figure;
plot(K,yp1,'-r',K,ym1,'--g')
legend('Plant Output', 'Reference Model Output');
title('Example 10(a) with control action - OSLA');
axis([0 500 -10 10]);
g=var(yp1(3:end)-ym1(3:end));
h=var(ym1(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp1(3:end)-ym1(3:end)).^2)/(length(yp1(3:end)));
disp(mse);

figure;
plot(K,yp2,'--g',K,ym2,'r')
legend('Plant Output', 'Reference Model Output');
axis([0 500 -3 3]);
title('Example 10 with control action - Single Layer BPA');
g=var(yp2(3:end)-ym2(3:end));
h=var(ym2(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp2(3:end)-ym2(3:end)).^2)/(length(yp2(3:end)));
disp(mse);
