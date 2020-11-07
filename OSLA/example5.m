close all;
clear all;
clc;

a = 1; b =1;
m0 = 2;
m1 = 18;
m2 = 1;

w11 = rand(m1,m0+1);%20 x 3
w12 = zeros(m2,m1+1);%1 x 21

w21 = rand(m1,m0+1);%20 x 3
w22 = zeros(m2,m1+1);%1 x 21

lambda= 0.0001;
P1=(1/lambda)*eye(m1+1);
P2=(1/lambda)*eye(m1+1);
yp1= [0];
yp2= [0];
yp1_cap=0;
yp2_cap=0;

for k=1:10000

    
        u1 = sin(2*pi*k/25);
        u2 = cos(2*pi*k/25);
        

        x_k = [yp1(end);yp2(end)];
        
        v0_k=[1;x_k];
        
        v11_k=a*tanh(b*w11*v0_k);%20*1
        v11_k=[1;v11_k];%21*1
        yn1=w12*v11_k;%1x1
        yp1cap = yn1 + u1;
        yp1_cap=[yp1_cap yp1cap];
       
        y1 = yp1(end)/(1 + (yp2(end))^2) + u1;
        yp1=[yp1 y1];
        
        e_k1=y1-yp1cap;
        
        v12_k=a*tanh(b*w21*v0_k);%20*1
        v12_k=[1;v12_k];%21*1
        yn2=w22*v12_k;%1x1
        
        yp2cap = yn2+u2;
        yp2_cap=[yp2_cap yp2cap];
        
        
        
        y2 =(yp1(end)*yp2(end))/(1 + (yp2(end))^2) + u2 ;
        yp2=[yp2 y2];

        e_k2=y2-yp2cap;
        
        P_k1=P1 - (P1*v11_k*(v11_k')*P1)/(1+((v11_k')*P1*v11_k));
        P1 = P_k1;
        w12=w12+e_k1*(v11_k')*P_k1;
        
        P_k2=P2- (P2*v12_k*(v12_k')*P2)/(1+((v12_k')*P2*v12_k));
        P2= P_k2;
        w22=w22+e_k2*(v12_k')*P_k2;
        
end
K=1:1:k+1;
figure;
plot(K,yp1_cap,'-r',K,yp1,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 5 Function Approximation for yp1 - OSLA');
axis([0 500 -5 4])
g=var(yp1(3:end)-yp1_cap(3:end));
h=var(yp1(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp1(3:end)-yp1_cap(3:end)).^2)/(length(yp1(3:end)));
disp(mse);

figure;
plot(K,yp2_cap,'-r',K,yp2,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 5 Function Approximation for yp2 - OSLA');
axis([9900 10000 -5 5])
g=var(yp2(3:end)-yp2_cap(3:end));
h=var(yp2(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp2(3:end)-yp2_cap(3:end)).^2)/(length(yp2(3:end)));
disp(mse);





