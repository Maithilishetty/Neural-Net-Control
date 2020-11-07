% function ym1=neural_net_5a(yp1,yp2)
clc;close all;clear all;
a = 1; b = 1;

m0 = 2;
m1 = 20;
m2 = 10;
m3 = 1;

w11 = 0.1*rand(m1,m0+1);%20 x 3
w21 = rand(m2,m1+1);%10 x 21
w31 = rand(m3,m2+1);%1 x 11

w12 = 0.1*rand(m1,m0+1);%20 x 3
w22 = 0.1*rand(m2,m1+1);%10 x 21
w32 = 0.1*rand(m3,m2+1);%1 x 11

eta1 = 0.01;


yp1= 0;
yp2= 0;
yp1_hat=0;
yp2_hat=0;

for k=1:100000

        u1 = sin(2*pi*k/25);
        u2 = cos(2*pi*k/25);

        p = [1;yp1(end);yp2(end)];
    

        %FORWARD PASS
        v11 = w11*p;%20 x 1 
        phi_v11 = a*tanh(b*v11); %20 x 1
        y11_k = [1 ; phi_v11];%21 x 1
        v21 = w21*y11_k; %10 x 1
        phi_v21 = a*tanh(b*v21);
        %linear activation function
        y21_k = [1 ; phi_v21];%11 x 1
        v31 = w31*y21_k; %1 x 1
        y31 = v31 ;
        
        y1 = yp1(end)/(1 + (yp2(end))^2) + u1;
        yp1=[yp1 y1];
        
        yp1_hat1 = y31 + u1;
        yp1_hat=[yp1_hat yp1_hat1];
        
        E1 = y1 - yp1_hat1;
        
        phi_v31_diff = 1;
        phi_v21_diff = (a/b)*(a^2 - phi_v21.^2); %10 x 1
        phi_v11_diff = (a/b)*(a^2 - phi_v11.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta31 = E1.*phi_v31_diff; % 1 x 1   
        delta_w31 = eta1.*delta31*(y21_k)'; % 1 x 11
        delta21 = (w31(2:length(w31))'.*delta31).*(phi_v21_diff); 
        delta_w21 = eta1.*delta21*(y11_k)';
        delta11 = (w21(:,2:length(w21))'*delta21).*(phi_v11_diff);
        delta_w11 = eta1.*delta11*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
         
        w11 = w11 + delta_w11;
        w21 = w21 + delta_w21;
        w31 = w31 + delta_w31;
        
        %FORWARD PASS
        
        v12 = w12*p;%20 x 1 
        phi_v12 = a*tanh(b*v12); %20 x 1
        y12_k = [1 ; phi_v12];%21 x 1
        v22 = w22*y12_k; %10 x 1
        phi_v22 = a*tanh(b*v21);
        %linear activation function
        y22_k = [1 ; phi_v22];%11 x 1
        v32 = w32*y22_k; %1 x 1
        y32 = v32 ;
        
        y2 =(yp1(end)*yp2(end))/(1 + (yp2(end))^2) + u2 ;
        yp2=[yp2 y2];
        
        yp2_hat1 = y32 + u2;
        yp2_hat=[yp2_hat yp2_hat1];      
        
        E2 = y2 - yp2_hat1;
        
        phi_v32_diff = (1);
        phi_v22_diff = (a/b)*(a^2 - phi_v22.^2); %10 x 1
        phi_v12_diff = (a/b)*(a^2 - phi_v12.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta32 = E2.*phi_v32_diff; % 1 x 1   
        delta_w32 = eta1.*delta32*(y22_k)'; % 1 x 11
        delta22 = (w32(2:length(w32))'.*delta32).*(phi_v22_diff); 
        delta_w22 = eta1.*delta22*(y12_k)';
        delta12 = (w22(:,2:length(w22))'*delta22).*(phi_v12_diff);
        delta_w12 = eta1.*delta12*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
         
        w12 = w12 + delta_w12;
        w22 = w22 + delta_w22;
        w32 = w32 + delta_w32;
        
        
end

K=1:1:k+1;
figure;
plot(K,yp1_hat,'-r',K,yp1,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 5 Function Approximation for yp1 - Two Layer BPA');
axis([0 500 -5 5]);
g=var(yp1(3:end)-yp1_hat(3:end));
h=var(yp1(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp1(3:end)-yp1_hat(3:end)).^2)/(length(yp1(3:end)));
disp(mse);
figure;
plot(K,yp2_hat,'-r',K,yp2,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 5 Function Approximation for yp2 - Two Layer BPA');
axis([0 500 -5 5]);
g=var(yp2(2:end)-yp2_hat(2:end));
h=var(yp2(2:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp2(2:end)-yp2_hat(2:end)).^2)/(length(yp2(2:end)));
disp(mse);





