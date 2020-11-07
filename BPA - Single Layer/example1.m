clc;close all;clear all;
a = 1; b = 1;
m0 = 1;
m1 = 10;
m2 = 1;
w1 = rand(m1,m0+1);%20 x 3
w2 = rand(m2,m1+1);%10 x 21
eta1 = 0.1;
n = 0:1000;

for k=1:length(n)

    
        u = -1 + 2.*rand(1);
        p = [1;u];
          
        %FORWARD PASS
        v1 = w1*p;%20 x 1 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 1
        y2 = v2 ;
        E = 0.6*sin(pi*u) + 0.3*sin(3*pi*u) + 0.1*sin(5*pi*u)- y2;
        phi_v2_diff = 1;
        phi_v1_diff = a*(1 - phi_v1.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta2 = E.*phi_v2_diff; % 1 x 1   
        delta_w2 = eta1.*delta2*(y1_k)'; % 1 x 11
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff); 
        delta_w1 = eta1.*delta1*(p)';
        %delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        %delta_w1 = eta1.*delta1*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
        
        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;

end

yp = [0;0];
ym = [0;0];

for k = 1:length(n)
    
        u = sin(2*pi*k/250);
        p = [1;u];

        y1 = 0.3*yp(end) + 0.6*yp(end-1) +0.6*sin(pi*u) + 0.3*sin(3*pi*u) + 0.1*sin(5*pi*u);
        
        %FORWARD PASS
        v1 = w1*p;%20 x 1 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 
        y2 = (v2)+ 0.3*yp(end) + 0.6*yp(end-1);
        
        yp = [yp;y1];
        ym = [ym;y2];

end

K = 0:length(n)+1;
figure;
plot(K,ym,'-r',K,yp,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example1 Function Approximation - BPA Single Layer');
g=var(yp(3:end)-ym(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);