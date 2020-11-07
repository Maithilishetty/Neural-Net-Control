clc;close all;clear all;
a = 1; b = 0.75;
m0 = 2;
m1 = 50;
m2 = 10;
m3 = 1;

w1 = rand(m1,m0+1);
w2 = rand(m2,m1+1);
w3 = zeros(m3,m2+1);
eta1 = 0.1;
s = 0:100000;
yp=[0;0];
ym=[0;0];
u = 0;

for k = 1:length(s)
    
        p = [1;yp(end);yp(end-1)];
        r = sin(2*pi*k/25);
        yp1 = yp(end-1)*yp(end)*(yp(end) + 2.5)/(1 + (yp(end))^2 + (yp(end-1))^2) + u;
        ym1 = 0.6*ym(end) + 0.2*ym(end-1) + r;
       
        %FORWARD PASS
        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k;
        phi_v2 = a*tanh(b*v2);
        y2_k = [1 ; phi_v2];
        v3 = w3*y2_k; 
        y3 = v3 ;
        E =  yp1 - u - y3;
        phi_v3_diff = (1);
        phi_v2_diff =  (b/a)*(a^2 - (phi_v2.^2)); 
        phi_v1_diff =  (b/a)*(a^2 - (phi_v1.^2)); 
   
        %BACKWARD PASS
    
        delta3 = E*phi_v3_diff;  
        delta_w3 = eta1.*delta3*(y2_k)'; 
        delta2 = (w3(2:length(w3))'.*delta3).*(phi_v2_diff); 
        delta_w2 = eta1.*delta2*(y1_k)';
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta1.*delta1*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION

        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;
        w3 = w3 + delta_w3;          
    
        u = -y3 + 0.6*yp(end) + 0.2*yp(end-1) + r;
        yp=[yp;yp1];
        ym=[ym;ym1];
        
end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% -----------------------------------------------------------------------

K = 0:length(s)+1;
figure;
plot(K,yp,'-r',K,ym,'--g')
legend('Plant Output', 'Reference Model Output');
title('Example 7 with control - Two Layer BPA');
axis([0 500 -4 6]);

g=var(yp(3:end)-ym(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);
