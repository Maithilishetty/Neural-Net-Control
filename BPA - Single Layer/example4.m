clc;
close all;
clear all;
a = 1; b = 1;
m0 = 5;
m1 = 20;
m2 = 1;
w1 = rand(m1,m0+1);%20 x 3
w2 = rand(m2,m1+1);%10 x 21
eta1 = 0.1;
yp = [0;0;0];
u = -1 + 2.*rand(50000,1);

for k=1:99998
    if k<50000
        p = [1;yp(end);yp(end-1);yp(end-2);u(k+1);u(k)];
        y = (yp(end)*yp(end-1)*yp(end-2)*u(k)*(yp(end-2) - 1) + u(k+1))/(1 + yp(end-2)^2 + yp(end-1)^2);
        
        %FORWARD PASS
        v1 = w1*p;%20 x 1 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 1
        y2= v2 ;
        E = (yp(end)*yp(end-1)*yp(end-2)*u(k)*(yp(end-2) - 1) + u(k+1))/(1 + yp(end-2)^2 + yp(end-1)^2) - y2;
        phi_v2_diff = (1);
        phi_v1_diff = a*(1 - phi_v1.^2); % 20 x 1
   
        %BACKWARD PASS
    
        delta2 = E.*phi_v2_diff; % 1 x 1   
        delta_w2 = eta1.*delta2*(y1_k)'; % 1 x 11
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta1.*delta1*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION
        
        
        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;
        
        yp = [yp;y];
end
end

yp = [0;0;0];
ym = [0;0;0];

for k = 1:800
    if(k<=500)
        u(k) = sin(2*pi*k/250);
    else
        u(k) = 0.8*sin(2*pi*k/250) + 0.2*sin(2*pi*k/25);
    end
end

for k = 1:800
        p = [1;yp(end);yp(end-1);yp(end-2);u(k+1);u(k)];
        y1 = (yp(end)*yp(end-1)*yp(end-2)*u(k)*(yp(end-2) - 1) + u(k+1))/(1 + yp(end-2)^2 + yp(end-1)^2);
        
        %FORWARD PASS
        v1 = w1*p;%20 x 1 
        phi_v1 = a*tanh(b*v1); %20 x 1
        y1_k = [1 ; phi_v1];%21 x 1
        v2 = w2*y1_k; %10 x 1
        %linear activation function
        y2 = (v2);
        yp = [yp;y1];
        ym = [ym;y2];

end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %

K = 1:803;
figure;
plot(K,ym,'-r',K,yp,'--g')
legend('Neural Network Output', 'Desired Output');
title('Example 4 Function Approximation - Single Layer BPA');


%VAF Calculation
g=var(yp(3:end)-ym(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf)
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);
