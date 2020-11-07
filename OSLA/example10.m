clc;
close all;
clear all;

a = 1; b = 1;
m0 = 2;
m1 = 30;
m2 = 1;

w11 = rand(m1,m0+1);
w21 = zeros(m2,m1+1);

w12 = rand(m1,m0+1);
w22 = zeros(m2,m1+1);

lambda= 0.001;

P1 = (1/lambda)*eye(m1+1);
P2 = (1/lambda)*eye(m1+1);

yp1 = 0;
yp2 = 0;
ym1 = 0;
ym2 = 0;
u1 = 0;
u2 = 0;
s = 0:100000;

for k = 1:length(s)
    
        r1 = sin(2*pi*k/25);
        r2 = cos(2*pi*k/25);

        ym1_k = 0.6*ym1(end) + 0.2*ym2(end) + r1;
        ym1 = [ym1 ym1_k];
        ym2_k = 0.1*ym1(end) - 0.8*ym2(end) + r2;
        ym2 = [ym2 ym2_k];
          
        x_k = [yp1(end);yp2(end)];
        v0_k = [1;x_k];
        
        v11_k = a*tanh(b*w11*v0_k);
        y1 = [1;v11_k];
        yn1 = w21*y1;
       
        yp1_k = yp1(end)/(1+yp2(end)^2) + u1;   
        yp1 = [yp1 yp1_k];
        yp1_cap = yn1 + u1;
        e_k1 = yp1(end)/(1+yp2(end)^2) - yn1;
        
        P_k1 = P1 - (P1*y1*y1'*P1)/(1+(y1'*P1*y1));
        P1 = P_k1;
        w21 = w21 + e_k1*y1'*P1;
        
        v12_k = a*tanh(b*w12*v0_k);
        y2 = [1;v12_k];
        yn2 = w22*y2;  
        
        yp2_k = (yp1(end)*yp2(end))/(1 + (yp2(end))^2) + u2;
        yp2 = [yp2 yp2_k];
        yp2_cap = yn2 + u2;     
        e_k2 = (yp1(end)*yp2(end))/(1 + (yp2(end))^2) - yn2;
             
        P_k2 = P2 - (P2*y2*y2'*P2)/(1+(y2'*P2*y2));
        P2 = P_k2;
        w22 = w22 + e_k2*y2'*P2;
         
        u1 = -yn1 + 0.6*yp1(end) + 0.2*yp2(end) + r1;
        u2 = -yn2 + 0.1*yp1(end) - 0.8*yp2(end) + r2;
        

end
        
% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %
K = 0:length(s);
figure;
plot(K,yp1,'-r',K,ym1,'--g')
legend('Plant Output', 'Reference Model Output');
title('Example 10 with control action for yp1 - OSLA');
axis([0 500 -10 10]);
g=var(ym1(3:end)-yp1(3:end));
h=var(ym1(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp1(3:end)-ym1(3:end)).^2)/(length(yp1(3:end)));
disp(mse);

figure;
plot(K,yp2,'-r',K,ym2,'--g')
legend('Plant Output', 'Reference Model Output');
axis([0 500 -10 10]);
title('Example 10 with control action for yp2 - OSLA');
g=var(ym2(3:end)-yp2(3:end));
h=var(ym2(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp2(3:end)-ym2(3:end)).^2)/(length(yp2(3:end)));
disp(mse);
